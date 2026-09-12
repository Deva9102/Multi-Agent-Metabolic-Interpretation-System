"""
glucose_agent.py

GLUCOSE AGENT — finds spikes and patterns in a subject's glucose
timeseries.

Two-layer design (this is the pattern every specialist agent in this
system follows):
  1. Deterministic math (find_spikes, analyze) — computes exact,
     guaranteed-correct numbers from the data. No LLM involved here,
     on purpose: arithmetic on a timeseries should never be left to
     an LLM to "read" and risk a wrong number.
  2. LLM interpretation (interpret) — takes those already-computed
     numbers and adds a short natural-language read on them. The LLM
     never sees the raw timeseries and never computes a number itself
     — it only narrates around numbers that are already known correct.

LLM call: yes (1, in interpret()).

Usage:
    from agents.glucose_agent import analyze, interpret

    result = analyze(subject_id=11)
    result["interpretation"] = interpret(result)
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cgm_data import get_subject_data

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-20b"

SPIKE_THRESHOLD = 180  # mg/dL — standard clinical hyperglycemia threshold


def find_spikes(timeseries, glucose_col="libre_gl", threshold=SPIKE_THRESHOLD):
    """Find every distinct spike episode where glucose crosses above
    the threshold. Groups consecutive high readings into one episode
    rather than counting every single minute separately."""
    ts = timeseries.dropna(subset=[glucose_col]).sort_values("timestamp").reset_index(drop=True)
    above = ts[glucose_col] > threshold

    episodes = []
    in_spike = False
    start_idx = None

    for i, is_above in enumerate(above):
        if is_above and not in_spike:
            in_spike = True
            start_idx = i
        elif not is_above and in_spike:
            in_spike = False
            episode = ts.iloc[start_idx:i]
            episodes.append(_summarize_episode(episode, glucose_col))

    if in_spike:  # spike still ongoing at the end of the data
        episode = ts.iloc[start_idx:]
        episodes.append(_summarize_episode(episode, glucose_col))

    return episodes


def _summarize_episode(episode_df, glucose_col):
    peak_row = episode_df.loc[episode_df[glucose_col].idxmax()]
    return {
        "start_time": str(episode_df["timestamp"].iloc[0]),
        "end_time": str(episode_df["timestamp"].iloc[-1]),
        "peak_value": float(peak_row[glucose_col]),
        "peak_time": str(peak_row["timestamp"]),
        "duration_minutes": len(episode_df),  # ~1 row per minute
    }


def analyze(subject_id, near_time=None):
    """Main entry point. Returns a dict of glucose findings for one subject.

    If near_time is given (a string like '2020-05-03 19:00:00'), returns
    the spike closest to that time. Otherwise returns all spikes found,
    with the largest one highlighted.
    """
    data = get_subject_data(subject_id)
    if data is None:
        return {"error": f"No data found for subject {subject_id}"}

    ts = data["timeseries"]
    baseline = float(ts["libre_gl"].dropna().median())

    spikes = find_spikes(ts)

    if not spikes:
        return {
            "subject_id": subject_id,
            "baseline_glucose": baseline,
            "spike_count": 0,
            "spikes": [],
            "note": f"No readings above {SPIKE_THRESHOLD} mg/dL found for this subject.",
        }

    largest_spike = max(spikes, key=lambda s: s["peak_value"])

    result = {
        "subject_id": subject_id,
        "baseline_glucose": round(baseline, 1),
        "spike_count": len(spikes),
        "largest_spike": largest_spike,
        "all_spikes": spikes,
    }

    if near_time:
        import pandas as pd
        target = pd.Timestamp(near_time)
        closest = min(spikes, key=lambda s: abs(pd.Timestamp(s["peak_time"]) - target))
        result["closest_to_question"] = closest

    return result


def interpret(analysis_result):
    """Take the already-computed numbers from analyze() and ask an LLM
    for a short, plain-English read on them. The LLM is given ONLY the
    computed numbers below — never the raw timeseries — so it cannot
    invent or misread a value; it can only comment on what's already
    verified correct."""

    if analysis_result.get("spike_count", 0) == 0:
        prompt = (
            f"A patient's glucose baseline (median) was "
            f"{analysis_result['baseline_glucose']} mg/dL, with no readings "
            f"above {SPIKE_THRESHOLD} mg/dL during the monitoring period. "
            f"In one short sentence, describe what this means for their "
            f"glucose stability. Do not add any numbers not given above."
        )
    else:
        spike = analysis_result["largest_spike"]
        prompt = (
            f"A patient's glucose baseline (median) was "
            f"{analysis_result['baseline_glucose']} mg/dL. They had "
            f"{analysis_result['spike_count']} spike episode(s) above "
            f"{SPIKE_THRESHOLD} mg/dL. The largest spike peaked at "
            f"{spike['peak_value']} mg/dL at {spike['peak_time']}, lasting "
            f"{spike['duration_minutes']} minutes. In one short sentence, "
            f"describe this pattern in plain language. Do not add any "
            f"numbers not given above, and do not guess at causes."
        )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=400,  # gpt-oss spends some tokens on hidden reasoning first
        reasoning_effort="low",  # we only need a one-sentence answer, not deep reasoning
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    result = analyze(11)
    print(f"Subject {result['subject_id']}")
    print(f"Baseline (median) glucose: {result['baseline_glucose']} mg/dL")
    print(f"Number of spike episodes (>{SPIKE_THRESHOLD} mg/dL): {result['spike_count']}")
    print(f"\nLargest spike:")
    for k, v in result["largest_spike"].items():
        print(f"  {k}: {v}")

    print(f"\nAll spikes found:")
    for i, spike in enumerate(result["all_spikes"], 1):
        print(f"  {i}. Peak {spike['peak_value']} mg/dL at {spike['peak_time']} "
              f"(lasted {spike['duration_minutes']} min)")

    print(f"\nLLM interpretation:")
    print(f"  {interpret(result)}")