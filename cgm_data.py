"""
cgm_data.py 

Data access layer for the pipeline.

Reads from the small, pre-extracted per-subject files in
data_agent/processed/subject_<id>/ (created by running
extract_subject_data.py first).

This file's only job is: given a subject_id, load their 4 small files
and hand back one clean bundle for the agents to use.
"""

import json
import os

import pandas as pd

# Extracted data lives inside data_agent/, not the top-level data/ folder
PROCESSED_BASE = "data_agent/processed/"


def list_subjects():
    """Return all subject IDs that have been extracted (i.e. that have
    a data_agent/processed/subject_<id>/ folder)."""
    if not os.path.exists(PROCESSED_BASE):
        return []
    folders = [f for f in os.listdir(PROCESSED_BASE) if f.startswith("subject_")]
    ids = sorted(int(f.replace("subject_", "")) for f in folders)
    return ids


def get_subject_data(subject_id):
    """Load one subject's pre-extracted data. Returns None if not found.

    Returns a dict:
        {
            "subject_id": int,
            "timeseries": DataFrame (minute-level glucose/activity),
            "meals": DataFrame (logged meals with macros),
            "bio": dict (one-time labs, age, BMI),
            "gut_health": dict (microbiome-derived scores),
        }
    """
    subject_dir = os.path.join(PROCESSED_BASE, f"subject_{subject_id}")

    if not os.path.exists(subject_dir):
        return None

    ts = pd.read_parquet(os.path.join(subject_dir, "timeseries.parquet"))
    meals = pd.read_parquet(os.path.join(subject_dir, "meals.parquet"))

    with open(os.path.join(subject_dir, "bio.json")) as f:
        bio = json.load(f)

    with open(os.path.join(subject_dir, "gut_health.json")) as f:
        gut_health = json.load(f)

    return {
        "subject_id": subject_id,
        "timeseries": ts,
        "meals": meals,
        "bio": bio,
        "gut_health": gut_health,
    }


def explore_subject(subject_id):
    """Print a human-readable summary of one subject's data."""
    data = get_subject_data(subject_id)
    if data is None:
        print(f"Subject {subject_id} not found. Did you run extract_subject_data.py first?")
        return

    print("=" * 60)
    print(f"SUBJECT {subject_id}")
    print("=" * 60)

    ts = data["timeseries"]
    print(f"\nTimeseries: {len(ts)} readings")
    if len(ts) > 0:
        print(f"  Date range: {ts['timestamp'].min()} to {ts['timestamp'].max()}")
        print(f"  Glucose (libre_gl) range: {ts['libre_gl'].min():.0f}-{ts['libre_gl'].max():.0f} mg/dL")
        print(f"  Glucose (libre_gl) mean: {ts['libre_gl'].mean():.1f} mg/dL")

    meals = data["meals"]
    print(f"\nMeals logged: {len(meals)}")
    if len(meals) > 0:
        print(meals[["timestamp", "meal_type", "calories", "carbs", "fiber"]].head(5))

    bio = data["bio"]
    print(f"\nBio / labs:")
    for key in ["age", "gender", "bmi", "a1c_pdl_lab", "fasting_glu___pdl_lab"]:
        print(f"  {key}: {bio.get(key)}")

    gut = data["gut_health"]
    print(f"\nGut health scores:")
    for key in ["gut_lining_health", "inflammatory_activity", "microbiome_induced_stress", "gut_microbiome_health"]:
        print(f"  {key}: {gut.get(key)}")