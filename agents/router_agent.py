"""
router_agent.py

ROUTER AGENT — reads the user's question and decides which specialist
agents are relevant before any of them run. This is what makes the
system genuinely agentic rather than a fixed pipeline: it chooses its
own investigation path per question.

"""

import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from llm_client import create_completion


def route(question):
    """Decide which specialist agents are relevant to this question.
    Fails safe: if parsing fails, defaults to running ALL agents
    rather than silently skipping something that might matter."""

    prompt = (
        f"A user asked this question about their health data: "
        f"\"{question}\"\n\n"
        f"Four specialist agents are available:\n"
        f"- GLUCOSE: analyzes blood sugar spikes and patterns over time\n"
        f"- MEAL: analyzes logged meals and their macros (carbs, fiber, "
        f"fat) around a specific time\n"
        f"- BLOODWORK: analyzes lab values (A1c, fasting glucose, BMI) "
        f"compared to a cohort\n"
        f"- GUT: analyzes gut microbiome scores compared to a cohort\n\n"
        f"Decide which of these are relevant to answering the question. "
        f"Only include an agent if the question genuinely relates to what "
        f"it analyzes. For example, a question only about a specific meal "
        f"and a glucose spike does not need GUT. A general question about "
        f"overall health risk might need all four.\n\n"
        f"IMPORTANT — questions unrelated to this patient's metabolic "
        f"health data should get ALL FOUR set to false. This includes: "
        f"general chit-chat, weather, other people's or animals' health "
        f"(e.g. a pet), requests for a specific meal recommendation with "
        f"no data grounding, or any topic these four agents don't cover. "
        f"Do NOT select an agent 'just in case' if the question doesn't "
        f"actually reference this patient's own glucose, meals, labs, or "
        f"gut data. Examples: \"What's the weather today?\" -> all false. "
        f"\"Tell me about my dog's health\" -> all false (this system only "
        f"analyzes the human patient's own data, not a pet's).\n\n"
        f'Respond ONLY with valid JSON: {{"glucose": true or false, '
        f'"meal": true or false, "bloodwork": true or false, '
        f'"gut": true or false, "reasoning": "one sentence explaining '
        f'the choice"}}'
    )

    response = create_completion(
        max_tokens=300,
        reasoning_effort="low",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.choices[0].message.content.strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        plan = json.loads(raw_text)
        for key in ["glucose", "meal", "bloodwork", "gut"]:
            if key not in plan:
                raise ValueError(f"Missing key: {key}")
        plan["parse_error"] = False
        return plan
    except (json.JSONDecodeError, ValueError):
        return {
            "glucose": True, "meal": True, "bloodwork": True, "gut": True,
            "reasoning": "Router output could not be parsed — defaulting to "
                         "running all agents to avoid skipping something relevant.",
            "parse_error": True,
        }