"""
cohort_stats.py

Shared helper for comparative analysis across all 45 subjects.
Loads every subject's bio.json and gut_health.json and computes where a given subject ranks relative to the cohort
for any numeric field.

This is what turns Bloodwork/Gut Agents from static threshold lookups
into genuine comparative detection — instead of "A1c > 5.7", they can
now say "this person's A1c is higher than 38 of the other 44 subjects".

"""

import sys
import os

sys.path.append(os.path.dirname(__file__))
from cgm_data import get_subject_data, list_subjects
from scipy import stats

_cache = {"bio": None, "gut": None}


def load_all_bio():
    """Load every subject's bio dict once, cached after first call."""
    if _cache["bio"] is None:
        all_bio = {}
        for sid in list_subjects():
            data = get_subject_data(sid)
            if data and data["bio"]:
                all_bio[sid] = data["bio"]
        _cache["bio"] = all_bio
    return _cache["bio"]


def load_all_gut():
    """Load every subject's gut_health dict once, cached after first call."""
    if _cache["gut"] is None:
        all_gut = {}
        for sid in list_subjects():
            data = get_subject_data(sid)
            if data and data["gut_health"]:
                all_gut[sid] = data["gut_health"]
        _cache["gut"] = all_gut
    return _cache["gut"]


def get_percentile(subject_id, field, category):
    """Return where subject_id ranks for `field` among the cohort.

    category must be "bio" or "gut".
    Returns a dict with the subject's value, percentile (0-100), and
    cohort size — or None if the field/subject isn't found.
    """
    all_data = load_all_bio() if category == "bio" else load_all_gut()

    if subject_id not in all_data:
        return None

    subject_value = all_data[subject_id].get(field)
    if subject_value is None:
        return None

    all_values = [
        v[field] for v in all_data.values()
        if v.get(field) is not None
    ]

    if len(all_values) < 2:
        return None

    lower_count = sum(1 for v in all_values if v < subject_value)
    percentile = round(100 * lower_count / (len(all_values) - 1), 1)

    return {
        "subject_id": subject_id,
        "field": field,
        "value": subject_value,
        "percentile": percentile,
        "cohort_size": len(all_values),
        "cohort_min": min(all_values),
        "cohort_max": max(all_values),
        "cohort_mean": round(sum(all_values) / len(all_values), 2),
    }


def get_statistical_outlier_test(subject_id, field, category):
    """The real 'Analyst' step: is this subject's value statistically
    distinguishable from the rest of the cohort, or just normal
    variation? Uses a one-sample t-test comparing the subject's value
    against the distribution of everyone else (leave-one-out).

    This is a genuinely stronger claim than percentile alone —
    percentile just says "where you rank," this says "is that rank
    likely to be meaningful or just noise."
    """
    all_data = load_all_bio() if category == "bio" else load_all_gut()

    if subject_id not in all_data:
        return None

    subject_value = all_data[subject_id].get(field)
    if subject_value is None:
        return None

    # Leave-one-out: compare subject's value against everyone else
    other_values = [
        v[field] for sid, v in all_data.items()
        if sid != subject_id and v.get(field) is not None
    ]

    if len(other_values) < 3:
        return None  # not enough data for a meaningful test

    # One-sample t-test: is subject_value an unlikely draw from the
    # distribution of the rest of the cohort?
    t_stat, p_value = stats.ttest_1samp(other_values, subject_value)

    is_significant = p_value < 0.05  # standard threshold

    return {
        "subject_id": subject_id,
        "field": field,
        "value": subject_value,
        "cohort_mean_excluding_subject": round(sum(other_values) / len(other_values), 2),
        "cohort_std_excluding_subject": round(stats.tstd(other_values), 2),
        "t_statistic": round(float(t_stat), 3),
        "p_value": round(float(p_value), 4),
        "statistically_significant": bool(is_significant),
        "note": (
            "Statistically distinguishable from the rest of the cohort "
            "(p < 0.05)" if is_significant else
            "Within normal variation for the cohort (p >= 0.05) — the "
            "difference could plausibly be due to chance"
        ),
    }