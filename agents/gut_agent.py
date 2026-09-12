"""
gut_agent.py

GUT AGENT — computes where a subject's gut scores rank against the
cohort (percentile + statistical significance, both in code), then
asks an LLM to narrate the comparative finding.

The gut microbiome fields have an unconfirmed scale direction: we do
not know whether a higher or lower percentile is "good." Percentile
rank and a significance test are neutral factual statements regardless
of direction, so those are computed in code. A rule-based guardrail
backstops the LLM narration in case it slips into directional language
anyway ("healthy", "concerning", etc.) — if it does, a safe
code-generated fallback replaces the LLM's text entirely.

Always flagged confidence="exploratory": even a real, statistically
significant percentile difference doesn't tell us whether it's
favorable, since the scale's meaning is unconfirmed.

LLM call: yes (1), with a code guardrail on the output.

Usage:
    from agents.gut_agent import analyze, interpret

    result = analyze(subject_id=11)
    result["interpretation"] = interpret(result)
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cgm_data import get_subject_data
from cohort_stats import get_percentile, get_statistical_outlier_test

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-20b"

FIELDS_TO_COMPARE = [
    "gut_lining_health", "inflammatory_activity",
    "microbiome_induced_stress", "gut_microbiome_health",
]

BANNED_DIRECTIONAL_WORDS = [
    "healthy", "unhealthy", "good", "bad", "optimal", "suboptimal",
    "better", "worse", "improved", "worsened", "concerning", "reassuring",
]


def check_for_directional_language(text):
    text_lower = text.lower()
    return [w for w in BANNED_DIRECTIONAL_WORDS if w in text_lower]


def analyze(subject_id):
    """Compute this subject's real cohort percentile AND statistical
    significance for each gut field — all in code."""
    data = get_subject_data(subject_id)
    if data is None:
        return {"error": f"No data found for subject {subject_id}"}

    comparisons = {}
    for field in FIELDS_TO_COMPARE:
        pct = get_percentile(subject_id, field, category="gut")
        sig = get_statistical_outlier_test(subject_id, field, category="gut")
        if pct:
            comparisons[field] = {**pct, "significance_test": sig}

    if not comparisons:
        return {"error": f"No comparable gut data found for subject {subject_id}"}

    return {
        "subject_id": subject_id,
        "confidence": "exploratory",
        "comparisons": comparisons,
    }


def interpret(analysis_result):
    """Ask the LLM to narrate the comparative finding — without
    judging whether the percentile is good or bad. Guardrail-checked;
    falls back to a safe, fully factual summary if directional
    language slips through."""

    comparisons = analysis_result["comparisons"]
    comparisons_text = "\n".join(
        f"- {field}: this subject's score is {c['value']}, at the "
        f"{c['percentile']}th percentile among {c['cohort_size']} subjects "
        f"(cohort range {c['cohort_min']}-{c['cohort_max']}); "
        f"statistically_significant={c['significance_test'].get('statistically_significant') if c.get('significance_test') else 'unknown'}"
        for field, c in comparisons.items()
    )

    prompt = (
        f"Here is how one patient's gut microbiome scores compare to a "
        f"real cohort of other subjects:\n\n{comparisons_text}\n\n"
        f"IMPORTANT: We do NOT know whether a higher or lower percentile "
        f"is better or worse for any of these measures. Report ONLY the "
        f"percentile rankings and whether each is statistically "
        f"significant, factually (e.g. 'this subject ranks at the Nth "
        f"percentile, which is/is not a statistically significant "
        f"difference from the cohort'). Do NOT say 'good', 'bad', "
        f"'healthy', 'optimal', 'better', or 'worse'. Do not invent "
        f"numbers not given above. State clearly that percentile rank "
        f"alone does not indicate whether this is favorable, since the "
        f"scale's meaning is unconfirmed. Keep your answer to 3-4 "
        f"sentences."
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=400,
        reasoning_effort="low",
        messages=[{"role": "user", "content": prompt}],
    )

    llm_output = response.choices[0].message.content.strip()
    flagged_words = check_for_directional_language(llm_output)

    if flagged_words:
        safe_fallback = "\n".join(
            f"{field}: {c['value']} ({c['percentile']}th percentile among "
            f"{c['cohort_size']} subjects; statistically significant: "
            f"{c['significance_test'].get('statistically_significant') if c.get('significance_test') else 'unknown'})"
            for field, c in comparisons.items()
        ) + (
            "\nPercentile rank is reported factually only. Whether a "
            "higher or lower rank is favorable is not confirmed for "
            "these measures."
        )
        return {
            "text": safe_fallback,
            "guardrail_triggered": True,
            "guardrail_reason": f"LLM output contained directional language: {flagged_words}",
        }

    return {
        "text": llm_output,
        "guardrail_triggered": False,
    }


if __name__ == "__main__":
    result = analyze(11)
    print(f"Subject {result['subject_id']}")
    print("\nComparisons (code-computed):")
    for field, c in result["comparisons"].items():
        print(f"  {field}: {c}")

    interpretation = interpret(result)
    print(f"\nGuardrail triggered: {interpretation['guardrail_triggered']}")
    if interpretation.get("guardrail_reason"):
        print(f"Reason: {interpretation['guardrail_reason']}")
    print("\nGut Agent narration:")
    print(f"  {interpretation['text']}")