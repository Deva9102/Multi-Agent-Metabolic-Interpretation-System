"""
bloodwork_agent.py

BLOODWORK AGENT — classifies a subject's one-time labs (A1c, fasting
glucose, BMI) against standard clinical thresholds AND against the
real cohort, then asks an LLM to explain the result in plain language.

Two-layer design (same pattern as every agent in this system):
  1. Deterministic math (classify) — fixed clinical thresholds, plus
     a real one-sample statistical significance test against the
     cohort.
  2. LLM narration (interpret) — explains the already-computed
     classifications and significance results in plain language.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cgm_data import get_subject_data
from cohort_stats import get_statistical_outlier_test

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-20b"

FIELDS = {
    "a1c_pdl_lab": "A1c",
    "fasting_glu___pdl_lab": "fasting glucose",
    "bmi": "BMI",
}

THRESHOLDS = {
    "a1c_pdl_lab": [(5.7, "normal"), (6.5, "prediabetic"), (float("inf"), "diabetic range")],
    "fasting_glu___pdl_lab": [(100, "normal"), (126, "prediabetic"), (float("inf"), "diabetic range")],
    "bmi": [(25, "normal"), (30, "overweight"), (float("inf"), "obese")],
}


def _classify_value(field, value):
    if value is None:
        return None
    for cutoff, label in THRESHOLDS[field]:
        if value < cutoff:
            return label
    return THRESHOLDS[field][-1][1]


def analyze(subject_id):
    """Classifies a subject's one-time labs (A1c, fasting glucose, BMI)
    against standard clinical thresholds AND against the real cohort.
    Returns a dict with the raw labs, the code-computed classifications,
    and the statistical significance test results (p-value, cohort mean, and whether the difference is statistically significant)."""
    data = get_subject_data(subject_id)
    if data is None:
        return {"error": f"No data found for subject {subject_id}"}

    bio = data["bio"]

    raw_labs = {
        "age": bio.get("age"),
        "gender": bio.get("gender"),
        "bmi": bio.get("bmi"),
        "a1c": bio.get("a1c_pdl_lab"),
        "fasting_glucose": bio.get("fasting_glu___pdl_lab"),
    }

    fields_result = {}
    for field, label in FIELDS.items():
        value = bio.get(field)
        classification = _classify_value(field, value)
        sig_test = get_statistical_outlier_test(subject_id, field, category="bio")
        fields_result[field] = {
            "label": label,
            "value": value,
            "clinical_classification": classification,
            "significance_test": sig_test, 
        }

    return {
        "subject_id": subject_id,
        "raw_labs": raw_labs,
        "fields": fields_result,
    }


def interpret(analysis_result):
    """Take the already-computed classifications and significance
    tests and ask an LLM to explain them in plain language. The LLM
    is given only these computed results, never asked to apply
    thresholds or do a comparison itself."""

    lines = []
    for field, info in analysis_result["fields"].items():
        line = f"- {info['label']}: {info['value']}, clinically classified as '{info['clinical_classification']}'"
        sig = info.get("significance_test")
        if sig:
            line += (
                f"; vs. cohort mean {sig.get('cohort_mean')}, "
                f"statistically_significant={sig.get('statistically_significant')} "
                f"(p={sig.get('p_value')})"
            )
        lines.append(line)
    facts_text = "\n".join(lines)

    prompt = (
        f"Here are a patient's lab results, already classified against "
        f"standard clinical thresholds and already tested for "
        f"statistical significance against a real cohort:\n\n{facts_text}\n\n"
        f"In 3-4 sentences, explain these results in plain, patient-"
        f"friendly language. Rules:\n"
        f"1. State the exact numbers and classifications given above — "
        f"do not invent, recompute, or re-classify anything yourself.\n"
        f"2. Only describe a lab value as notably different from the "
        f"cohort if statistically_significant is true; otherwise "
        f"describe it as within normal variation for the cohort.\n"
        f"3. Do not give medical advice or recommendations."
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=400,
        reasoning_effort="low",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    result = analyze(11)
    print(f"Subject {result['subject_id']}")
    print(f"Raw labs: {result['raw_labs']}")
    print(f"\nFields (code-computed):")
    for field, info in result["fields"].items():
        print(f"  {info['label']}: {info}")
    print(f"\nBloodwork Agent interpretation:")
    print(f"  {interpret(result)}")