import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Meta's Llama, run through Hugging Face Inference Providers.
# We try these in order and remember the first one that answers.
LLAMA_MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]
_working_model = None


def _get_key() -> Optional[str]:
    # Works on your computer (.env) and on Streamlit Cloud (Secrets)
    key = os.getenv("HF_TOKEN")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("HF_TOKEN")
    except Exception:
        return None


def _ask_llama(prompt: str, max_tokens: int = 300) -> str:
    global _working_model
    from huggingface_hub import InferenceClient

    client = InferenceClient(api_key=_get_key())
    models = ([_working_model] if _working_model else []) + [
        m for m in LLAMA_MODELS if m != _working_model
    ]
    last_error = None
    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            _working_model = model
            logger.info(f"Llama answered using {model}")
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"{model} failed: {e}")
            last_error = e
    raise RuntimeError(f"No Llama model answered. Last error: {last_error}")


def _read_json(text: str) -> Dict:
    # Pulls the {...} part out of the model's reply
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in reply: {text[:200]}")
    return json.loads(match.group(0))


def parse_request(text: str, valid_genres, valid_moods) -> Tuple[Optional[Dict], str]:
    """Step 1 for Llama: turn plain words like 'chill worship music'
    into genre, mood, energy and likes_acoustic for the scoring engine."""
    if not _get_key():
        return None, "Set HF_TOKEN to use plain-words search."

    genres = sorted(valid_genres)
    moods = sorted(valid_moods)
    prompt = (
        "You turn a music request into settings for a song search engine.\n"
        f"Request: \"{text}\"\n\n"
        f"Pick genre from this list only: {genres}\n"
        f"Pick mood from this list only: {moods}\n"
        "energy is a number from 0.0 (very calm) to 1.0 (very intense).\n"
        "likes_acoustic is true or false.\n"
        "Worship or Christian music means genre \"gospel\".\n"
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
    logger.info(f"Llama read '{text}' as {prefs}")
    return prefs, ""


def get_ai_explanation(user_prefs: Dict, top_songs: List[Tuple[Dict, float, str]]) -> str:
    """Step 2 for Llama: explain why these songs fit (RAG: only uses the retrieved songs)."""
    if not _get_key():
        return "[AI explanation unavailable: set HF_TOKEN to enable this feature]"

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
