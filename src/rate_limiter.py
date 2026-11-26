import logging
import time

import requests

from ai_explainer import _get_secret

logger = logging.getLogger(__name__)

PER_VISITOR_LIMIT = 5      # Llama requests per visitor...
PER_VISITOR_WINDOW = 60    # ...in this many seconds
DAILY_LIMIT = 200          # Llama requests for the whole app per day


def check_visitor(state) -> tuple:
    """Limit for one visitor, kept in their Streamlit session.
    Returns (allowed, seconds_to_wait)."""
    now = time.time()
    recent = [t for t in state.get("llama_calls", []) if now - t < PER_VISITOR_WINDOW]
    if len(recent) >= PER_VISITOR_LIMIT:
        state["llama_calls"] = recent
        wait = int(PER_VISITOR_WINDOW - (now - recent[0])) + 1
        return False, wait
    recent.append(now)
    state["llama_calls"] = recent
    return True, 0


def check_daily() -> bool:
    """Limit for the whole app, counted in Redis so it is shared by every visitor.
    If Redis is not set up or fails, we let the request through."""
    kv_url = _get_secret("KV_REST_API_URL")
    kv_token = _get_secret("KV_REST_API_TOKEN")
    if not (kv_url and kv_token):
        return True
    key = "vibefinder_llama_calls_" + time.strftime("%Y-%m-%d", time.gmtime())
    try:
        r = requests.post(
            f"{kv_url}/pipeline",
            headers={"Authorization": f"Bearer {kv_token}"},
            json=[["INCR", key], ["EXPIRE", key, "172800"]],
            timeout=5,
        )
        count = int(r.json()[0]["result"])
        if count > DAILY_LIMIT:
            logger.warning(f"Daily Llama limit reached ({count}/{DAILY_LIMIT})")
            return False
        return True
    except Exception as e:
        logger.warning(f"Rate limit check failed, allowing request: {e}")
        return True
