"""
coordinator_agent.py

COORDINATOR AGENT — takes the outputs from whichever specialist
agents the Router decided to run (Glucose, Meal, Bloodwork, Gut) and
synthesizes them into one final, plain-English answer to the user's
original question.

Confidence handling:
  - Glucose, Meal, Bloodwork findings are treated as solid — the
    Coordinator can state them plainly (they're either pure code or
    code-computed-then-LLM-narrated, per each agent's own guardrails).
  - Gut Agent's finding is always passed in with its "exploratory"
    flag, and the Coordinator is instructed to hedge it accordingly,
    never stating it as equally certain.
  - Only domains the Router actually ran are ever mentioned. A domain
    that wasn't analyzed gets no claim at all, not even "no
    significant changes" because that's still a claim about a
    domain with zero data behind it.
"""

import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-20b"


def _extract_text(domain, result):
    """Pull the plain English text out of whatever shape each agent returns."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return result.get("text") or result.get("interpretation") or result.get("analysis") or str(result)
    return str(result)


def synthesize(question, findings):
    """Combine whichever specialist findings were provided into one
    final plain English answer. findings is a dict keyed by domain
    name."""

    sections = []
    provided_domains = []
    for domain in ["glucose", "meal", "bloodwork", "gut"]:
        if domain not in findings or findings[domain] is None:
            continue
        provided_domains.append(domain.upper())
        text = _extract_text(domain, findings[domain])
        if domain == "gut":
            sections.append(
                f"GUT FINDING (EXPLORATORY — treat as much less certain "
                f"than the other domains, hedge this specific part "
                f"clearly):\n{text}"
            )
        else:
            sections.append(f"{domain.upper()} FINDING (solid):\n{text}")

    if not sections:
        return "No relevant findings were available to answer this question."

    findings_text = "\n\n".join(sections)

    prompt = (
            f"A user asked this question about their health data: "
            f"\"{question}\"\n\n"
            f"Here is what specialist analyses found. ONLY these domains "
            f"were analyzed: {', '.join(provided_domains)}. No other domain "
            f"has ANY data — not even 'no significant changes' or 'within "
            f"normal variation'. Those are still claims about a domain, and "
            f"stating them for a domain with no data is exactly as wrong as "
            f"inventing a number.\n\n"
            f"EXAMPLE OF WHAT NOT TO DO: if only GLUCOSE and MEAL were "
            f"analyzed, do NOT write \"Bloodwork and gut findings did not "
            f"show any statistically significant changes\" — this is "
            f"fabricated, because bloodwork/gut were never run at all. The "
            f"correct approach is to simply not mention them, or say "
            f"plainly they weren't analyzed for this question.\n\n"
            f"{findings_text}\n\n"
                    f"Write ONE short, plain-English answer (3-5 sentences) to the "
            f"user's question, combining ONLY the findings above. Rules:\n"
            f"1. State non-gut findings directly and confidently — they are "
            f"well-established.\n"
            f"2. Mention the Gut finding (if present) only briefly, and "
            f"explicitly flag it as exploratory/less certain — never present "
            f"it with the same confidence as the other findings.\n"
            f"3. Use correlation language, not causation — say 'associated "
            f"with' or 'occurred around the same time as', never 'caused "
            f"by'. This ONLY applies to a specific meal's timestamp being "
            f"close to a specific glucose spike's timestamp — that temporal "
            f"proximity is real and was actually computed. It does NOT "
            f"license saying a lab value (A1c, fasting glucose, BMI) or a "
            f"gut score is 'associated with', 'related to', or 'explains' "
            f"glucose spikes or eating patterns — no analysis in this system "
            f"ever compared those domains to each other, so any such "
            f"statement is invented, not correlation.\n"
            f"4. Do not invent any numbers not present in the findings above.\n"
            f"5. Do not give prescriptive medical advice (e.g. 'you should "
            f"avoid X') — describe patterns only, not recommendations.\n"
            f"6. Never state or imply ANY finding — including 'normal', 'no "
            f"change', or 'not significant' — about a domain not in this "
            f"list: {', '.join(provided_domains)}.\n"
            f"7. If more than one domain is present, describe each domain's "
            f"findings as its own separate observation. Do not draw a "
            f"connection, explanation, or relationship between two "
            f"different domains (e.g. bloodwork explaining glucose spikes, "
            f"or gut scores relating to diet) — only the meal agent's own "
            f"logged proximity to a specific spike counts as a legitimate "
            f"association, and even that must stay within the meal/glucose "
            f"pairing itself."
        )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=500,
        reasoning_effort="low",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip()