"""
app.py

Streamlit UI for the Metabolic Detective multi-agent system.

Run with:
    streamlit run app.py
"""

import streamlit as st

from cgm_data import list_subjects
from pipeline import run

st.set_page_config(page_title="Metabolic Detective", layout="centered")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Source Serif 4', serif;
        font-weight: 600;
        color: #1F2A24;
    }
    .app-subtitle {
        color: #5B6B63;
        font-size: 0.95rem;
        margin-top: -0.6rem;
        margin-bottom: 1.4rem;
    }

    /* Domain pills */
    .domain-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.6rem 0 0.2rem 0; }
    .domain-pill {
        padding: 0.3rem 0.8rem; border-radius: 8px; font-size: 0.85rem; font-weight: 500;
    }
    .domain-active { background: #E3F2EE; color: #0E6B5C; border: 1px solid #B6E0D6; }
    .domain-skipped { background: #F2F2EF; color: #9A9A92; border: 1px solid #E5E5E0; }

    /* Answer card — the hero element */
    .answer-card {
        background: #FFFFFF;
        border-left: 4px solid #0E6B5C;
        border-radius: 6px;
        padding: 1.4rem 1.6rem;
        margin: 0.8rem 0 1.2rem 0;
        line-height: 1.6;
        font-size: 1.02rem;
        color: #24312B;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }

    /* Verdict banner */
    .verdict-pass { background: #E3F2EE; color: #0E6B5C; border: 1px solid #B6E0D6; }
    .verdict-fail { background: #FBEAEA; color: #A23B3B; border: 1px solid #F0C7C7; }
    .verdict-banner {
        padding: 0.7rem 1rem; border-radius: 8px; font-weight: 600; font-size: 0.95rem;
        margin-bottom: 0.6rem;
    }

    /* Metric rows inside the evaluation panel */
    .metric-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.55rem 0.9rem; border-radius: 6px; margin-bottom: 0.4rem;
        border-left: 3px solid transparent;
    }
    .metric-pass  { background: #F3FAF8; border-left-color: #0E6B5C; }
    .metric-fail  { background: #FDF3F3; border-left-color: #A23B3B; }
    .metric-advisory { background: #FDF8EE; border-left-color: #B8860B; }
    .metric-label { font-size: 0.92rem; color: #24312B; }
    .metric-badge { font-size: 0.78rem; font-weight: 600; padding: 0.15rem 0.6rem; border-radius: 999px; }
    .badge-pass { background: #0E6B5C; color: white; }
    .badge-fail { background: #A23B3B; color: white; }
    .badge-advisory { background: #B8860B; color: white; }
    .metric-detail { font-size: 0.82rem; color: #6B776F; margin-top: 0.15rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## Metabolic detective")
st.markdown(
    '<div class="app-subtitle">Ask a question about a subject\'s glucose, meals, bloodwork, or gut health.</div>',
    unsafe_allow_html=True,
)

# Input row
subjects = list_subjects()
subject_id = st.selectbox("Subject", subjects, index=subjects.index(11) if 11 in subjects else 0)

question = st.text_input(
    "Ask a question",
    placeholder="Why did I spike after lunch on October 6th?",
)

run_button = st.button("Ask", type="primary")

if run_button and question.strip():
    with st.spinner("Running the multi-agent pipeline..."):
        try:
            result = run(subject_id, question)
            st.session_state["last_result"] = result
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()
elif run_button:
    st.warning("Enter a question first.")

# Result display 
result = st.session_state.get("last_result")

if result:
    plan = result["router_plan"]
    labels = {"glucose": "Glucose", "meal": "Meal", "bloodwork": "Bloodwork", "gut": "Gut"}

    pills_html = '<div class="domain-row">'
    for key, label in labels.items():
        cls = "domain-active" if plan.get(key) else "domain-skipped"
        pills_html += f'<span class="domain-pill {cls}">{label}</span>'
    pills_html += "</div>"
    st.markdown(pills_html, unsafe_allow_html=True)
    st.caption(plan.get("reasoning", ""))

    st.markdown(f'<div class="answer-card">{result["answer"]}</div>', unsafe_allow_html=True)

    critic = result["critic"]
    overall_passed = critic["overall_passed"]
    verdict_cls = "verdict-pass" if overall_passed else "verdict-fail"
    verdict_text = "Passed evaluation" if overall_passed else "Failed evaluation — see details below"
    st.markdown(f'<div class="verdict-banner {verdict_cls}">{verdict_text}</div>', unsafe_allow_html=True)

    # On-demand evaluation metrics panel
    with st.expander("View evaluation metrics"):
        guardrails = critic["guardrails"]
        checks = guardrails.get("checks", {})
        flags_by_rule = {}
        for f in guardrails.get("flags", []):
            flags_by_rule.setdefault(f["rule"], []).append(f)

        st.markdown("**Guardrails** (rule-based, no LLM)")

        guardrail_display = [
            ("no_overconfident_language", "No overconfident language", "blocking"),
            ("no_causal_language", "No causal language (correlation only)", "blocking"),
            ("no_prescriptive_advice", "No prescriptive medical advice", "blocking"),
            ("no_undisclosed_domain_findings", "No findings stated for un-analyzed domains", "blocking"),
            ("time_format_ok", "12-hour time format review", "advisory"),
        ]

        for check_key, label, severity in guardrail_display:
            passed = checks.get(check_key, True)
            if passed:
                row_cls, badge_cls, badge_text = "metric-pass", "badge-pass", "PASS"
            elif severity == "advisory":
                row_cls, badge_cls, badge_text = "metric-advisory", "badge-advisory", "REVIEW"
            else:
                row_cls, badge_cls, badge_text = "metric-fail", "badge-fail", "FAIL"

            detail = ""
            fired_rule = "domain_finding_stated_but_not_analyzed" if check_key == "no_undisclosed_domain_findings" else (
                "time_format_review_needed" if check_key == "time_format_ok" else check_key
            )
            if not passed and fired_rule in flags_by_rule:
                phrases = [f.get("phrase_found") or f.get("note") or f.get("matched_text", "") for f in flags_by_rule[fired_rule]]
                detail = f'<div class="metric-detail">{"; ".join(p for p in phrases if p)}</div>'

            st.markdown(
                f'<div class="metric-row {row_cls}">'
                f'<span class="metric-label">{label}{detail}</span>'
                f'<span class="metric-badge {badge_cls}">{badge_text}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("**Critic judgments** (LLM check against real computed data)")

        critic_result = critic["critic"]
        critic_display = [
            ("faithful", "Faithful to the data", True),
            ("hallucination_detected", "No hallucinated facts", False),
            ("unbacked_medical_claims", "No unbacked medical claims", False),
            ("domain_violation", "No domain violations", False),
            ("overclaim", "No overclaiming exploratory findings", False),
        ]

        for key, label, good_value in critic_display:
            if key not in critic_result:
                continue
            value = critic_result[key]
            passed = (value == good_value)
            row_cls, badge_cls, badge_text = ("metric-pass", "badge-pass", "PASS") if passed else ("metric-fail", "badge-fail", "FAIL")
            st.markdown(
                f'<div class="metric-row {row_cls}">'
                f'<span class="metric-label">{label}</span>'
                f'<span class="metric-badge {badge_cls}">{badge_text}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if critic_result.get("explanation"):
            st.caption(f"Critic explanation: {critic_result['explanation']}")
        if critic_result.get("raw_response"):
            st.caption("⚠️ Critic response could not be parsed as JSON — showing raw text:")
            st.code(critic_result["raw_response"])

    with st.expander("Raw findings (ground truth data)"):
        for domain, data in result["raw_findings"].items():
            if data:
                st.markdown(f"**{domain.capitalize()}**")
                st.json(data, expanded=False)