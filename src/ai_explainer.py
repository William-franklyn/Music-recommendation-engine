import json
import logging
import os
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Meta's Llama, run for free by Groq.
# Groq retires old model names, so we ask Groq which ones are live and pick a Llama.
PREFERRED_MODELS = [
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
_chosen_model = None


def _get_key() -> Optional[str]:
    # Works on your computer (.env) and on Streamlit Cloud (Secrets)
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


def _pick_model(client) -> str:
    global _chosen_model
    if _chosen_model:
        return _chosen_model
    forced = os.getenv("GROQ_MODEL")
    if forced:
        _chosen_model = forced
        return forced
    live = [m.id for m in client.models.list().data]
    for name in PREFERRED_MODELS:
        if name in live:
            _chosen_model = name
            break
    else:
        llamas = [m for m in live if "llama" in m.lower() and "guard" not in m.lower()]
        if not llamas:
            raise RuntimeError("Groq has no Llama chat model available right now")
        _chosen_model = llamas[0]
    logger.info(f"Using Groq model {_chosen_model}")
    return _chosen_model


def _ask_llama(prompt: str, max_tokens: int = 300, json_mode: bool = False) -> str:
    from groq import Groq

    client = Groq(api_key=_get_key())
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = client.chat.completions.create(
        model=_pick_model(client),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
        **extra,
    )
    return response.choices[0].message.content.strip()


def parse_request(text: str, valid_genres, valid_moods) -> Tuple[Optional[Dict], str]:
    """Step 1 for Llama: turn plain words like 'chill worship music'
    into genre, mood, energy and likes_acoustic for the scoring engine."""
    if not _get_key():
        return None, "Set GROQ_API_KEY to use plain-words search."

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
        "Answer with JSON only, like: "
        "{\"genre\": \"gospel\", \"mood\": \"chill\", \"energy\": 0.3, \"likes_acoustic\": true}"
    )

    try:
        data = json.loads(_ask_llama(prompt, max_tokens=120, json_mode=True))
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
        return "[AI explanation unavailable: set GROQ_API_KEY to enable this feature]"

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
