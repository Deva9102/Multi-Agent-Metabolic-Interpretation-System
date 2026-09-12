"""
pipeline.py

LangGraph-based orchestrator for the multi-agent pipeline:

    Router Agent
        -> {Glucose, Meal, Bloodwork, Gut} Agents (only the ones the
           Router selected, run in parallel via LangGraph's
           conditional fan-out)
        -> Coordinator Agent (LLM synthesis, LangGraph fan-in)
        -> Critic Agent (guardrails + LLM critic)

Setup:
    pip install langgraph

State design note: each specialist agent writes to its OWN key
(glucose_result, meal_result, bloodwork_result, gut_result). Because
the Router can fan out to several specialists in parallel, giving each
one a distinct state key avoids any need for a custom merge/reducer
function — LangGraph merges independent keys from parallel branches
for free. The Meal Agent still recomputes its own spike-time fallback
internally rather than depending on Glucose Agent's node output, which
is what makes true parallel execution of Glucose and Meal safe here.

"""

from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END

from agents.router_agent import route
from agents.glucose_agent import analyze as glucose_analyze, interpret as glucose_interpret
from agents.meal_agent import analyze as meal_analyze, interpret as meal_interpret
from agents.bloodwork_agent import analyze as bloodwork_analyze, interpret as bloodwork_interpret
from agents.gut_agent import analyze as gut_analyze, interpret as gut_interpret
from agents.coordinator_agent import synthesize
from agents.critic_agent import evaluate, build_ground_truth_text

# State

class PipelineState(TypedDict, total=False):
    subject_id: int
    question: str
    router_plan: dict
    glucose_result: Optional[dict]
    meal_result: Optional[dict]
    bloodwork_result: Optional[dict]
    gut_result: Optional[dict]
    answer: str
    critic: dict

# Nodes — one per agent. Each reads only what it needs from state and
# returns a partial state update (LangGraph merges it in).

def router_node(state: PipelineState) -> dict:
    plan = route(state["question"])
    return {"router_plan": plan}


def glucose_node(state: PipelineState) -> dict:
    result = glucose_analyze(state["subject_id"])
    result["interpretation"] = glucose_interpret(result)
    return {"glucose_result": result}


def meal_node(state: PipelineState) -> dict:
    spike_time = None
    glucose_probe = glucose_analyze(state["subject_id"])
    if glucose_probe.get("spike_count", 0) > 0:
        spike_time = glucose_probe["largest_spike"]["peak_time"]

    if spike_time is None:
        return {"meal_result": None}

    mention_glucose_spike = bool(state.get("router_plan", {}).get("glucose"))

    result = meal_analyze(state["subject_id"], spike_time)
    result["interpretation"] = meal_interpret(result, mention_glucose_spike=mention_glucose_spike)
    return {"meal_result": result}


def bloodwork_node(state: PipelineState) -> dict:
    result = bloodwork_analyze(state["subject_id"])
    result["interpretation"] = bloodwork_interpret(result)
    return {"bloodwork_result": result}


def gut_node(state: PipelineState) -> dict:
    result = gut_analyze(state["subject_id"])
    result["interpretation"] = gut_interpret(result)
    return {"gut_result": result}


def _extract_narrations(state: PipelineState) -> dict:
    """Rebuild the domain -> plain-English-text mapping directly from
    the state's already-declared *_result keys. Both coordinator_node
    and critic_node call this, so the two nodes can never disagree
    about which domains were actually analyzed — there's no separate
    "smuggled" key that could get dropped or go stale between them."""
    narrations = {}
    if state.get("glucose_result"):
        narrations["glucose"] = state["glucose_result"]["interpretation"]
    if state.get("meal_result"):
        narrations["meal"] = state["meal_result"]["interpretation"]
    if state.get("bloodwork_result"):
        narrations["bloodwork"] = state["bloodwork_result"]["interpretation"]
    if state.get("gut_result"):
        narrations["gut"] = state["gut_result"]["interpretation"]["text"]
    return narrations


def coordinator_node(state: PipelineState) -> dict:
    narrations = _extract_narrations(state)
    answer = synthesize(state["question"], narrations)
    return {"answer": answer}


def critic_node(state: PipelineState) -> dict:
    narrations = _extract_narrations(state)
    provided_domains = list(narrations.keys())

    # Ground truth for the Critic comes from the agents' structured,
    # code-computed result dicts  NOT their LLM narrations. Checking
    # the Coordinator's answer against another LLM's prose is circular:
    # an error already baked into a specialist's narration would always
    # look "faithful". Comparing against the real numbers/timestamps
    # lets the Critic catch a value attributed to the wrong entity.
    raw_results = {
        "glucose": state.get("glucose_result"),
        "meal": state.get("meal_result"),
        "bloodwork": state.get("bloodwork_result"),
        "gut": state.get("gut_result"),
    }
    raw_results = {k: v for k, v in raw_results.items() if v}
    raw_findings_text = build_ground_truth_text(raw_results)

    critic_result = evaluate(state["answer"], raw_findings_text, state["question"], provided_domains)
    return {"critic": critic_result}

# Conditional routing — after the Router Agent decides the plan, fan out
# to only the specialist nodes it selected. Returning a list of node
# names from a conditional edge function is how LangGraph expresses
# "invoke these branches in parallel."

def select_specialists(state: PipelineState):
    plan = state["router_plan"]
    selected = []
    if plan.get("glucose"):
        selected.append("glucose_node")
    if plan.get("meal"):
        selected.append("meal_node")
    if plan.get("bloodwork"):
        selected.append("bloodwork_node")
    if plan.get("gut"):
        selected.append("gut_node")

    if not selected:
        # Router selected nothing relevant — skip straight to the
        # Coordinator, which will report "no relevant findings."
        return ["coordinator_node"]
    return selected

# Graph assembly

def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("router_node", router_node)
    graph.add_node("glucose_node", glucose_node)
    graph.add_node("meal_node", meal_node)
    graph.add_node("bloodwork_node", bloodwork_node)
    graph.add_node("gut_node", gut_node)
    graph.add_node("coordinator_node", coordinator_node)
    graph.add_node("critic_node", critic_node)

    graph.set_entry_point("router_node")

    graph.add_conditional_edges(
        "router_node",
        select_specialists,
        {
            "glucose_node": "glucose_node",
            "meal_node": "meal_node",
            "bloodwork_node": "bloodwork_node",
            "gut_node": "gut_node",
            "coordinator_node": "coordinator_node",
        },
    )
    graph.add_edge("glucose_node", "coordinator_node")
    graph.add_edge("meal_node", "coordinator_node")
    graph.add_edge("bloodwork_node", "coordinator_node")
    graph.add_edge("gut_node", "coordinator_node")

    graph.add_edge("coordinator_node", "critic_node")
    graph.add_edge("critic_node", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run(subject_id, question):
    """Main entry point. Runs the full graph for one subject/question
    pair and returns a clean result dict — this is what app.py,
    run_batch.py, and test_pipeline.py all import and call."""
    graph = get_graph()
    final_state = graph.invoke({"subject_id": subject_id, "question": question})

    raw_findings = {
        "glucose": final_state.get("glucose_result"),
        "meal": final_state.get("meal_result"),
        "bloodwork": final_state.get("bloodwork_result"),
        "gut": final_state.get("gut_result"),
    }
    raw_findings = {k: v for k, v in raw_findings.items() if v}

    return {
        "subject_id": subject_id,
        "question": question,
        "router_plan": final_state["router_plan"],
        "raw_findings": raw_findings,
        "answer": final_state["answer"],
        "critic": final_state["critic"],
    }


if __name__ == "__main__":
    result = run(11, "Why did I spike after lunch on October 6th?")

    print("ROUTER PLAN:", result["router_plan"])
    print("\nANSWER:\n", result["answer"])
    print("\nCRITIC OVERALL PASSED:", result["critic"]["overall_passed"])
    if result["critic"]["guardrails"]["flags"]:
        print("Guardrail flags:", result["critic"]["guardrails"]["flags"])
    print("Critic explanation:", result["critic"]["critic"].get("explanation"))