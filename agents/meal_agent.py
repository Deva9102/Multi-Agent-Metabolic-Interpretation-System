"""
meal_agent.py

MEAL AGENT — identifies which logged meal most likely preceded a
glucose spike and summarizes it.

This replaces the old nutrition_agent.py design, which handed the LLM
the ENTIRE meal list as text and asked it to find the closest one
itself — that's a lookup task, and lookup tasks should never be left
to an LLM to "read" a table and risk picking the wrong row or
mis-transcribing a number.

Two-layer design (same pattern as glucose_agent.py):
  1. Deterministic math (find_closest_meal) — pandas timestamp
     comparison. Guaranteed correct, no LLM involved.
  2. LLM narration (interpret) — takes the already-found meal and
     explains it in plain language, reasoning about the macros. The
     LLM never sees the full meal list and never picks the meal
     itself — it only narrates around a match that's already verified
     correct.

LLM call: yes (1, in interpret()).

Usage:
    from agents.meal_agent import analyze, interpret

    result = analyze(subject_id=11, spike_time="2020-10-06 12:51:00")
    result["interpretation"] = interpret(result)
"""

import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cgm_data import get_subject_data

from llm_client import create_completion

MAX_HOURS_BEFORE = 4  # a meal logged more than this long before a spike is not a candidate


def find_closest_meal(meals, spike_time, max_hours_before=MAX_HOURS_BEFORE):
    """Deterministically find the meal logged closest to (but before)
    the spike time, within a reasonable window. Returns None if no
    meal qualifies — this is reported honestly, never guessed."""
    meals = meals.copy()
    meals["timestamp"] = pd.to_datetime(meals["timestamp"])
    spike_ts = pd.Timestamp(spike_time)

    before = meals[meals["timestamp"] <= spike_ts].copy()
    if len(before) == 0:
        return None

    before["minutes_before"] = (spike_ts - before["timestamp"]).dt.total_seconds() / 60
    closest = before.loc[before["minutes_before"].idxmin()]

    if closest["minutes_before"] > max_hours_before * 60:
        return None

    return closest.to_dict()


def analyze(subject_id, spike_time):
    """Main entry point. Returns the deterministically-found closest
    meal (or None) for one subject/spike-time pair."""
    data = get_subject_data(subject_id)
    if data is None:
        return {"error": f"No data found for subject {subject_id}"}

    meals = data["meals"]
    if len(meals) == 0:
        return {
            "subject_id": subject_id,
            "spike_time": spike_time,
            "closest_meal": None,
            "note": "No meals logged for this subject.",
        }

    closest_meal = find_closest_meal(meals, spike_time)

    return {
        "subject_id": subject_id,
        "spike_time": spike_time,
        "closest_meal": closest_meal,
    }


def interpret(analysis_result, mention_glucose_spike=True):
    """Take the already-found meal (or lack thereof) and ask an LLM to
    narrate it in plain language. The LLM is given ONLY the matched
    meal's fields — never the full meal list — so it cannot invent a
    different match or misreport a value.

    mention_glucose_spike: whether the narration is allowed to
    disclose the glucose spike's time/value that was used internally
    to find this meal. This agent ALWAYS needs the spike time to
    determine which meal is closest — that lookup is unaffected. But
    if the Router didn't select the glucose domain for this question,
    narrating "your glucose spiked at X" is a real glucose FINDING
    leaking out through the meal domain's answer, which the Critic
    correctly flags as a domain violation. Pass False in that case so
    the meal is described on its own, without asserting anything
    about glucose."""

    meal = analysis_result.get("closest_meal")

    if meal is None:
        if mention_glucose_spike:
            prompt = (
                f"A patient's glucose spiked at {analysis_result['spike_time']}, "
                f"but no meal was logged within a reasonable window "
                f"(within {MAX_HOURS_BEFORE} hours) before that time. "
                f"In one short sentence, state plainly that no meal could be "
                f"linked to this spike — do not guess at what might have "
                f"caused it."
            )
        else:
            prompt = (
                f"No meal was logged within a reasonable lookback window "
                f"before the time being asked about. In one short sentence, "
                f"state plainly that no relevant meal was found in the "
                f"patient's log for this question. Do not mention blood "
                f"glucose, blood sugar, or any spike — describe only the "
                f"absence of a logged meal."
            )
    else:
        if mention_glucose_spike:
            prompt = (
                f"A patient's glucose spiked at {analysis_result['spike_time']}. "
                f"The meal logged closest before that spike was:\n"
                f"- Timestamp: {meal.get('timestamp')}\n"
                f"- Meal type: {meal.get('meal_type')}\n"
                f"- Calories: {meal.get('calories')} kcal\n"
                f"- Carbs: {meal.get('carbs')}g\n"
                f"- Fiber: {meal.get('fiber')}g\n"
                f"- Fat: {meal.get('fat')}g\n\n"
                f"In 2-3 sentences, describe this meal and note it as the "
                f"meal most closely associated with (not proven to have "
                f"caused) the spike. Report the numbers exactly as given "
                f"above — do not invent, round unusually, or add any "
                f"number not listed."
            )
        else:
            prompt = (
                f"Here is a meal logged in a patient's data:\n"
                f"- Timestamp: {meal.get('timestamp')}\n"
                f"- Meal type: {meal.get('meal_type')}\n"
                f"- Calories: {meal.get('calories')} kcal\n"
                f"- Carbs: {meal.get('carbs')}g\n"
                f"- Fiber: {meal.get('fiber')}g\n"
                f"- Fat: {meal.get('fat')}g\n\n"
                f"In 2-3 sentences, describe this meal using ONLY the "
                f"numbers given above — its timestamp, type, and macros. "
                f"Do NOT mention or imply blood glucose, blood sugar, a "
                f"spike, or any other health metric — describe only the "
                f"meal itself. Do not invent, round unusually, or add any "
                f"number not listed."
            )

    response = create_completion(
        max_tokens=400,
        reasoning_effort="low",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    result = analyze(11, "2020-10-06 12:51:00")
    print(f"Subject {result['subject_id']}")
    print(f"Spike time: {result['spike_time']}")
    print(f"Closest meal (code-matched): {result['closest_meal']}")
    print(f"\nMeal Agent interpretation:")
    print(f"  {interpret(result)}")