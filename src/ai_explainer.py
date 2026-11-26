import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Plan A: our own Llama 3.2 3B running in Google Colab (link saved in Redis).
# Plan B: Meta's Llama on Cloudflare Workers AI (free daily limit).
REDIS_KEY = "vibefinder_llama_url"
LLAMA_MODELS = [
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "@cf/meta/llama-3.1-8b-instruct",
    "@cf/meta/llama-3.2-3b-instruct",
]
_working_model = None
_colab_cache = {"url": None, "checked": 0.0}
last_source = "none"  # which Llama answered last: "Colab (self-hosted)" or "Cloudflare"


def _get_secret(name: str) -> Optional[str]:
    # Works on your computer (.env) and on Streamlit Cloud (Secrets)
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def _has_cloudflare() -> bool:
    return bool(_get_secret("CLOUDFLARE_API_TOKEN") and _get_secret("CLOUDFLARE_ACCOUNT_ID"))


def _has_keys() -> bool:
    return _has_cloudflare() or bool(_colab_url())


def _colab_url() -> Optional[str]:
    """Reads the Colab link from Redis. Checks at most every 30 seconds."""
    kv_url = _get_secret("KV_REST_API_URL")
    kv_token = _get_secret("KV_REST_API_TOKEN")
    if not (kv_url and kv_token):
        return None
    if time.time() - _colab_cache["checked"] < 30:
        return _colab_cache["url"]
    try:
        r = requests.post(kv_url, headers={"Authorization": f"Bearer {kv_token}"},
                          json=["GET", REDIS_KEY], timeout=5)
        _colab_cache["url"] = r.json().get("result")
    except Exception as e:
        logger.warning(f"Could not read Colab link from Redis: {e}")
        _colab_cache["url"] = None
    _colab_cache["checked"] = time.time()
    return _colab_cache["url"]


def _ask_colab(prompt: str, max_tokens: int) -> str:
    url = _colab_url()
    if not url:
        raise RuntimeError("Colab Llama is off")
    r = requests.post(f"{url}/chat", json={
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }, timeout=45)
    r.raise_for_status()
    return r.json()["response"].strip()


def _ask_cloudflare(prompt: str, max_tokens: int) -> str:
    global _working_model
    account = _get_secret("CLOUDFLARE_ACCOUNT_ID")
    token = _get_secret("CLOUDFLARE_API_TOKEN")
    models = ([_working_model] if _working_model else []) + [
        m for m in LLAMA_MODELS if m != _working_model
    ]
    last_error = None
    for model in models:
        try:
            r = requests.post(
                f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                },
                timeout=60,
            )
            data = r.json()
            if not r.ok or not data.get("success"):
                raise RuntimeError(f"HTTP {r.status_code}: {data.get('errors')}")
            answer = data["result"]["response"]
            if not isinstance(answer, str):
                answer = json.dumps(answer)
            _working_model = model
            logger.info(f"Cloudflare answered using {model}")
            return answer.strip()
        except Exception as e:
            logger.warning(f"{model} failed: {e}")
            last_error = e
    raise RuntimeError(f"No Cloudflare model answered. Last error: {last_error}")


def _ask_llama(prompt: str, max_tokens: int = 300) -> str:
    """Tries our own Colab Llama first, then falls back to Cloudflare automatically."""
    global last_source
    try:
        answer = _ask_colab(prompt, max_tokens)
        last_source = "Colab (self-hosted)"
        logger.info("Colab Llama answered")
        return answer
    except Exception as e:
        logger.info(f"Colab not used ({e}), falling back to Cloudflare")
    if not _has_cloudflare():
        raise RuntimeError("Colab is off and Cloudflare keys are missing")
    answer = _ask_cloudflare(prompt, max_tokens)
    last_source = "Cloudflare"
    return answer


def _read_json(text: str) -> Dict:
    # Pulls the {...} part out of the model's reply
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in reply: {text[:200]}")
    return json.loads(match.group(0))


def parse_request(text: str, valid_genres, valid_moods) -> Tuple[Optional[Dict], str]:
    """Step 1 for Llama: turn plain words like 'chill worship music' into
    genre, mood, energy and likes_acoustic for the scoring engine."""
    if not _has_keys():
        return None, "Set the Cloudflare keys (or start the Colab Llama) to use plain-words search."

    genres = sorted(valid_genres)
    moods = sorted(valid_moods)
    prompt = (
        "You turn a music request into settings for a song search engine.\n"
        f"Request: \"{text}\"\n\n"
        f"Pick genre from this list only: {genres}\n"
        f"Pick mood from this list only: {moods}\n"
        "energy is a number from 0.0 (very calm) to 1.0 (very intense).\n"
        "likes_acoustic is true or false.\n"
        "Worship or Christian music means genre \"worship\". Christmas music means genre \"christmas\".\n"
        "Answer with JSON only and nothing else, like: "
        "{\"genre\": \"gospel\", \"mood\": \"chill\", \"energy\": 0.3, \"likes_acoustic\": true}"
    )
    try:
        data = _read_json(_ask_llama(prompt, max_tokens=120))
    except Exception as e:
        logger.error(f"Llama parse error: {e}")
        return None, f"Llama couldn't read that request: {e}"

    # Never trust the model blindly: check every value before using it
    genre = str(data.get("genre", "pop")).strip().lower().replace("-", " ")
    if genre not in valid_genres:
        genre = "pop"
    mood = str(data.get("mood", "chill")).strip().lower()
    if mood not in valid_moods:
        mood = "chill"
    try:
        energy = min(1.0, max(0.0, float(data.get("energy", 0.5))))
    except (TypeError, ValueError):
        energy = 0.5
    likes_acoustic = str(data.get("likes_acoustic", False)).lower() == "true"

    prefs = {"genre": genre, "mood": mood, "energy": energy, "likes_acoustic": likes_acoustic}
    logger.info(f"Llama ({last_source}) read '{text}' as {prefs}")
    return prefs, ""


def get_ai_explanation(user_prefs: Dict, top_songs: List[Tuple[Dict, float, str]]) -> str:
    """Step 2 for Llama: explain why these songs fit (RAG: only uses the retrieved songs)."""
    if not _has_keys():
        return "[AI explanation unavailable: set the Cloudflare keys or start the Colab Llama]"

    song_lines = []
    for i, (song, score, _) in enumerate(top_songs, 1):
        song_lines.append(
            f"{i}. \"{song['title']}\" by {song['artist']} "
            f"(genre: {song['genre']}, mood: {song['mood']}, "
            f"energy: {song['energy']:.2f}, acousticness: {song['acousticness']:.2f}, "
            f"match score: {score:.2f})"
        )
    prompt = (
        "You are a friendly music recommendation assistant.\n\n"
        f"Listener wants: genre {user_prefs.get('genre')}, mood {user_prefs.get('mood')}, "
        f"energy {user_prefs.get('energy', 0.5):.1f}/1.0, likes acoustic: {user_prefs.get('likes_acoustic', False)}\n\n"
        "The system picked these songs:\n" + "\n".join(song_lines) + "\n\n"
        "Write 3 to 4 warm sentences explaining why these songs fit. "
        "Only use the details listed above. Do not invent anything."
    )
    try:
        return _ask_llama(prompt, max_tokens=300)
    except Exception as e:
        logger.error(f"Llama explanation error: {e}")
        return f"[AI explanation unavailable: {e}]"
