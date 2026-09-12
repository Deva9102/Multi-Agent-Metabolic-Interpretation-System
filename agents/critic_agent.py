"""
critic_agent.py

CRITIC AGENT — the final quality-control step. Validates the
Coordinator's answer against the raw findings.

Two layers, on purpose:
  1. Guardrails (run_guardrails) — pure code, no LLM, no judgment
     calls. Every check here is a simple, explicit regex rule. This
     is the last line of defense: even if the LLM critic call fails,
     is unavailable, or is inconsistent, guardrails still runs and
     blocks hard violations.
  2. LLM critic (run_critic) — an actual model call that checks
     faithfulness, hallucination, unbacked medical claims, domain
     violations, and overclaiming. Catches subtler issues guardrails'
     regexes can't (e.g. a rephrased causal claim, a subtly invented
     number).

NOTE: this file replaces BOTH the old critic.py and the old
guardrails.py, which contained a duplicate copy of the same
run_guardrails() logic. There is now exactly one guardrails
implementation.

LLM call: yes (1, in run_critic()).

Usage:
    from agents.critic_agent import evaluate

    result = evaluate(final_answer, raw_findings_text, question, provided_domains)
"""

import json
import os
import re
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from llm_client import create_completion

CONFIDENT_LANGUAGE = [
    "clearly shows", "clearly indicates",
    "definitely", "certainly", "proves that", "confirms that",
]
# NOTE: "very low"/"very high" were removed from this list. They're
# meant to catch overconfident DIAGNOSTIC claims ("your risk is very
# high"), but they also fire on completely legitimate factual
# statements ("your score is at the very low end of the percentile
# range") — a real 0th-percentile value IS "very low" by definition.
# Naive phrase-matching can't tell the difference, so keeping these
# in a blocking list produced false failures on correct answers.

CAUSAL_LANGUAGE = [
    "caused by", "causes", "triggered by", "triggers", "led to",
    "because of", "due to", "resulted in", "results in",
]

PRESCRIPTIVE_LANGUAGE = [
    "you should avoid", "you should eat", "you should stop",
    "you need to", "you must", "i recommend", "we recommend",
    "avoid eating", "stop eating",
]


def build_ground_truth_text(raw_results):
    """Build the Critic's comparison text from genuinely structured,
    code-computed fields — NOT from any agent's LLM narration.

    This matters because comparing the final answer against another
    LLM's prose is circular: if a specialist agent's own narration
    already contains an error (e.g. attributing a glucose spike's peak
    time to the meal instead of the meal's own logged time), checking
    the Coordinator's answer against that narration will always look
    "faithful" — the error just propagates silently. Every timestamp
    and number here is pulled directly from each agent's analyze()
    output, with explicit labels naming which entity each value
    belongs to, so the Critic can actually catch a swapped/misattributed
    value instead of just checking self-consistency between two LLM
    outputs.

    `raw_results` is a dict like {"glucose": glucose_result_dict,
    "meal": meal_result_dict, ...} — the structured dicts returned by
    each agent's analyze(), not the .interpret()/["interpretation"] text.
    """
    lines = []

    glucose = raw_results.get("glucose")
    if glucose and glucose.get("spike_count", 0) > 0:
        spike = glucose["largest_spike"]
        all_spikes = glucose.get("all_spikes", [])
        other_peaks = ", ".join(
            f"{s['peak_value']} mg/dL at {s['peak_time']}"
            for s in all_spikes if s is not spike
        )
        lines.append(
            f"GLUCOSE (ground truth, code-computed): baseline={glucose.get('baseline_glucose')} mg/dL; "
            f"TOTAL SPIKE COUNT={glucose['spike_count']} spike episode(s) above 180 mg/dL total. "
            f"The LARGEST of these {glucose['spike_count']} episode(s): PEAK VALUE={spike['peak_value']} mg/dL "
            f"at PEAK TIME={spike['peak_time']}, duration={spike['duration_minutes']} minutes."
            + (f" Other episode(s): {other_peaks}." if other_peaks else "")
            + f" IMPORTANT: {spike['peak_time']} is the GLUCOSE SPIKE's peak time — it is NOT a meal timestamp. "
            f"A statement mentioning '{glucose['spike_count']} spikes/episodes' is CORRECT and faithful — "
            f"do not flag the total spike count as invented, it is given right here."
        )
    elif glucose:
        lines.append(
            f"GLUCOSE (ground truth, code-computed): baseline={glucose.get('baseline_glucose')} mg/dL; "
            f"no spike episodes found."
        )

    meal = raw_results.get("meal")
    if meal and meal.get("closest_meal"):
        m = meal["closest_meal"]
        lines.append(
            f"MEAL (ground truth, code-computed): the meal's own LOGGED TIMESTAMP={m.get('timestamp')} "
            f"(this is DIFFERENT from and must not be confused with the glucose spike's peak time above), "
            f"meal_type={m.get('meal_type')}, calories={m.get('calories')} kcal, carbs={m.get('carbs')}g, "
            f"fiber={m.get('fiber')}g, fat={m.get('fat')}g. "
            f"IMPORTANT: {m.get('timestamp')} is the MEAL's own logged time — it is NOT the glucose spike's peak time."
        )
    elif meal:
        lines.append("MEAL (ground truth, code-computed): no meal found within the lookback window.")

    bloodwork = raw_results.get("bloodwork")
    if bloodwork:
        field_lines = []
        for field, info in bloodwork.get("fields", {}).items():
            sig = info.get("significance_test") or {}
            field_lines.append(
                f"{info['label']}={info['value']} (clinically classified as "
                f"'{info['clinical_classification']}'; statistically_significant="
                f"{sig.get('statistically_significant')})"
            )
        lines.append("BLOODWORK (ground truth, code-computed): " + "; ".join(field_lines))

    gut = raw_results.get("gut")
    if gut:
        field_lines = []
        for field, c in gut.get("comparisons", {}).items():
            sig = c.get("significance_test") or {}
            field_lines.append(
                f"{field}={c['value']} ({c['percentile']}th percentile of {c['cohort_size']}; "
                f"statistically_significant={sig.get('statistically_significant')})"
            )
        lines.append(
            "GUT (ground truth, code-computed, EXPLORATORY — direction of scale unconfirmed): "
            + "; ".join(field_lines)
        )

    return "\n\n".join(lines)


def run_guardrails(final_answer, provided_domains=None):
    """Deterministic checks against the final answer. Returns a report
    — never silently passes without listing what was checked, so
    failures are always traceable.

    Each flag carries a severity:
      - "blocking": a real rule violation. Any blocking flag fails the
        guardrail check outright.
      - "advisory": worth a human's attention but not proof of an
        error by itself (e.g. a 12-hour time that MIGHT be a bad
        conversion, or might just be correctly noon-hour). Advisory
        flags are always returned so nothing is hidden, but they do
        NOT flip `passed` to False on their own.
    """

    text_lower = final_answer.lower()
    flags = []

    for phrase in CONFIDENT_LANGUAGE:
        if phrase in text_lower:
            flags.append({"rule": "no_overconfident_language", "phrase_found": phrase, "severity": "blocking"})

    for phrase in CAUSAL_LANGUAGE:
        if phrase in text_lower:
            flags.append({"rule": "no_causal_language", "phrase_found": phrase, "severity": "blocking"})

    # Cues that mean the phrase is being used to DECLINE or reference
    # the question, not to actually give the advice — e.g. "...does not
    # answer whether you should stop eating carbs" is a refusal, not
    # prescriptive advice, even though "you should stop" appears in it.
    NEGATION_CUES = [
        "whether", "not provide", "does not", "doesn't", "cannot",
        "can't", "no direct", "won't", "unable to", "not able to",
        "not give", "not offer",
    ]

    def _is_negated(phrase, text):
        idx = text.find(phrase)
        if idx == -1:
            return False
        window_start = max(0, idx - 40)
        preceding_window = text[window_start:idx]
        return any(cue in preceding_window for cue in NEGATION_CUES)

    for phrase in PRESCRIPTIVE_LANGUAGE:
        if phrase in text_lower and not _is_negated(phrase, text_lower):
            flags.append({"rule": "no_prescriptive_advice", "phrase_found": phrase, "severity": "blocking"})

    if re.search(r"\d{1,2}:\d{2}\s*(am|pm)", text_lower):
        flags.append({
            "rule": "time_format_review_needed",
            "note": "Contains a 12-hour AM/PM time — verify it matches "
                    "the original 24-hour timestamp.",
            "severity": "advisory",
        })

    # Narrow, precise backstop: catches "the/your <domain> analysis/data/
    # results was/were/is/are ..." — this specific sentence structure
    # reliably means a finding is being STATED about that domain, not
    # just disclosed as absent. Only fires for domains NOT provided.
    if provided_domains is not None:
        all_domains = ["glucose", "meal", "bloodwork", "gut"]
        for domain in all_domains:
            if domain in provided_domains:
                continue
            pattern = rf"(the|your)\s+{domain}\s+(analysis|data|results|findings|scores?)\s+(was|were|is|are|show|shows|showed)"
            match = re.search(pattern, text_lower)
            if match:
                flags.append({
                    "rule": "domain_finding_stated_but_not_analyzed",
                    "domain": domain,
                    "matched_text": match.group(0),
                    "note": f"Sentence structure implies a '{domain}' finding "
                            f"exists, but '{domain}' was not in the analyzed "
                            f"domains ({provided_domains}).",
                    "severity": "blocking",
                })

    blocking_flags = [f for f in flags if f.get("severity") == "blocking"]

    # Full checklist for display purposes — includes checks that PASSED
    # silently, not just ones that fired. `flags` above only ever
    # contains violations; a UI showing "every metric we check" needs
    # to know about the checks with nothing to report too.
    fired_rules = {f["rule"] for f in flags}
    checks = {
        "no_overconfident_language": "no_overconfident_language" not in fired_rules,
        "no_causal_language": "no_causal_language" not in fired_rules,
        "no_prescriptive_advice": "no_prescriptive_advice" not in fired_rules,
        "no_undisclosed_domain_findings": "domain_finding_stated_but_not_analyzed" not in fired_rules,
        "time_format_ok": "time_format_review_needed" not in fired_rules,  # advisory, not blocking
    }

    return {"passed": len(blocking_flags) == 0, "flags": flags, "checks": checks}


def run_critic(final_answer, raw_findings_text, question, provided_domains):
    """LLM critic call. Checks the Coordinator's answer against the
    raw findings for hallucination, unbacked claims, and overclaiming."""

    prompt = (
        f"You are a strict clinical fact-checker. A user asked: "
        f"\"{question}\"\n\n"
        f"Only these domains were actually analyzed: {provided_domains}. "
        f"No other domain has any data — mentioning a lack of data for "
        f"another domain is fine, but stating an actual finding about it "
        f"is a violation.\n\n"
        f"RAW FINDINGS (the only source of truth — pay close attention to "
        f"any 'IMPORTANT' notes marking which value belongs to which "
        f"entity; a common failure mode is attributing one entity's "
        f"timestamp or number to a different entity, e.g. reporting a "
        f"glucose spike's peak time as if it were a meal's logged time):"
        f"\n{raw_findings_text}\n\n"
        f"ANSWER TO CHECK:\n{final_answer}\n\n"
        f"Evaluate the answer against these findings:\n"
        f"1. faithful: does every number/claim trace exactly to the raw "
        f"findings above, AND is each value attributed to the correct "
        f"entity? (allow reasonable rounding, but a correct number "
        f"attributed to the wrong entity — e.g. a meal 'logged at' a "
        f"time that is actually the glucose peak time — is NOT faithful)\n"
        f"2. hallucination_detected: did it invent any fact, number, or "
        f"domain not present in the raw findings?\n"
        f"3. unbacked_medical_claims: any medical assertion (diagnosis, "
        f"recommendation, causation) not directly supported by the data?\n"
        f"4. domain_violation: does it state an actual finding for a "
        f"domain outside {provided_domains}? (honestly saying 'no data "
        f"for X' is NOT a violation)\n"
        f"5. overclaim: does it treat an exploratory or "
        f"statistically-non-significant finding as certain/notable?\n\n"
        f"Respond ONLY with valid JSON, no other text, exactly matching:\n"
        f'{{"faithful": true/false, "hallucination_detected": true/false, '
        f'"unbacked_medical_claims": true/false, "domain_violation": '
        f'true/false, "overclaim": true/false, "verdict": "pass" or '
        f'"fail", "explanation": "one or two sentences explaining any '
        f'violations found, or why it passed"}}'
    )

    try:
        response = create_completion(
            max_tokens=600,
            reasoning_effort="low",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.choices[0].message.content.strip()
    except Exception as e:
        return {
            "error": str(e),
            "faithful": False, "hallucination_detected": True,
            "unbacked_medical_claims": True, "domain_violation": True,
            "overclaim": True, "verdict": "fail",
            "explanation": f"Critic call failed, defaulting to fail-safe: {e}",
        }

    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {
            "error": "Could not parse critic response as JSON",
            "raw_response": raw_text,
            "faithful": False, "hallucination_detected": True,
            "unbacked_medical_claims": True, "domain_violation": True,
            "overclaim": True, "verdict": "fail",
            "explanation": "Critic output was unparseable, defaulting to fail-safe.",
        }


def evaluate(final_answer, raw_findings_text, question="", provided_domains=None):
    """Full Critic Agent pass: deterministic guardrails + LLM critic call."""

    if not provided_domains:
        guardrail_result = run_guardrails(final_answer)
        return {
            "guardrails": guardrail_result,
            "critic": {
                "verdict": "pass",
                "explanation": "No domains were relevant — nothing to check.",
            },
            "overall_passed": guardrail_result["passed"],
        }

    guardrail_result = run_guardrails(final_answer, provided_domains=provided_domains)
    critic_result = run_critic(final_answer, raw_findings_text, question, provided_domains)

    critic_passed = critic_result.get("verdict") == "pass"

    return {
        "guardrails": guardrail_result,
        "critic": critic_result,
        # Guardrails is a hard gate — if the regex backstop catches the
        # exact bug the LLM critic missed, overall_passed must still be False.
        "overall_passed": guardrail_result["passed"] and critic_passed,
    }


if __name__ == "__main__":
    final_answer = (
        "Your glucose spiked to 208 mg/dL on October 6th, staying elevated "
        "for about 101 minutes, and the lunch at 12:01 AM on that day is "
        "the meal most closely associated with that rise. That lunch "
        "contained 94 g of carbohydrates, 5 g of fiber, and 13 g of fat. "
        "Your blood work shows a BMI of 29.2 (overweight) and an A1c of "
        "5.7% (pre-diabetes range), with a fasting glucose of 109 mg/dL "
        "(pre-diabetic). Gut health scores are exploratory; "
        "they are not yet linked to clear symptoms and should be "
        "interpreted with caution."
    )
    raw_findings = (
        "Glucose: Baseline 105 mg/dL, largest spike 208 mg/dL at "
        "2020-10-06 12:51:00, lasting 101 minutes.\n"
        "Meal: Lunch at 2020-10-06 12:01:00, 94g carbs, 5g fiber, 13g fat.\n"
        "Bloodwork: BMI 29.16 (overweight), A1c 5.7% (prediabetic), "
        "fasting glucose 109 (prediabetic).\n"
        "Gut: Gut lining health score 1.0, inflammatory activity score "
        "1.0. Direction of scale not confirmed."
    )

    result = evaluate(final_answer, raw_findings,
                       question="Why did I spike after lunch on October 6th?",
                       provided_domains=["glucose", "meal", "bloodwork", "gut"])

    print("Guardrails:", result["guardrails"])
    print("\nCritic result:", json.dumps(result["critic"], indent=2))
    print("\nOverall passed:", result["overall_passed"])