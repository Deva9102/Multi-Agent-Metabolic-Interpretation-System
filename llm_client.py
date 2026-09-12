"""
llm_client.py

Every specialist agent, the Router, Coordinator, and Critic call 
create_completion() from here instead oftalking to Cerebras' SDK directly 
— one shared client, one shared retry policy.

Setup:
    pip install openai python-dotenv

    # .env:
    CEREBRAS_API_KEY=your_key_here
"""

import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = "https://api.cerebras.ai/v1"
DEFAULT_MODEL = "gpt-oss-120b"


def _model_supports_reasoning_effort(model_name):
    name = model_name.lower()
    return "gpt-oss" in name or "glm" in name


_api_key = os.environ.get("CEREBRAS_API_KEY")
if not _api_key:
    raise RuntimeError(
        "CEREBRAS_API_KEY is not set in the environment (.env file). "
        "Get a free key at https://cloud.cerebras.ai and add it to your .env."
    )

_client = OpenAI(base_url=BASE_URL, api_key=_api_key)

MAX_RETRIES = 6
DEFAULT_BACKOFF_SECONDS = 10
SAFETY_MARGIN_SECONDS = 1.0


def _extract_wait_seconds(error_message):
    match = re.search(r"try again in ([\d.]+)\s*s", error_message)
    if match:
        return float(match.group(1))
    return None


def _is_rate_limit_error(e):
    status_code = getattr(e, "status_code", None)
    if status_code == 429:
        return True
    return "rate_limit" in str(e).lower() or "429" in str(e)


def create_completion(**kwargs):
    """Wraps Cerebras' chat.completions.create() with automatic retry
    on 429 rate-limit errors. Every agent should call this instead of
    instantiating its own client."""

    kwargs.setdefault("model", DEFAULT_MODEL)

    if not _model_supports_reasoning_effort(kwargs["model"]):
        kwargs.pop("reasoning_effort", None)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _client.chat.completions.create(**kwargs)
        except Exception as e:
            last_error = e
            if not _is_rate_limit_error(e) or attempt == MAX_RETRIES:
                raise

            wait_seconds = _extract_wait_seconds(str(e))
            if wait_seconds is None:
                wait_seconds = DEFAULT_BACKOFF_SECONDS
            wait_seconds += SAFETY_MARGIN_SECONDS

            print(
                f"  [llm_client:cerebras] Rate limited (attempt {attempt}/{MAX_RETRIES}) "
                f"— waiting {wait_seconds:.1f}s before retrying..."
            )
            time.sleep(wait_seconds)

    raise last_error