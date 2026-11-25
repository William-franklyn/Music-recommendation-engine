import os
from typing import Tuple

import pandas as pd

# The original genres, kept so older tests and songs.csv still work
BASE_GENRES = {"pop", "lofi", "rock", "ambient", "jazz", "synthwave", "indie pop"}
VALID_MOODS = {"happy", "chill", "intense", "relaxed", "moody", "focused"}

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")


def _load_dataset_genres() -> set:
    # Reads every genre in the real Spotify dataset ("hip-hop" becomes "hip hop")
    try:
        genres = pd.read_csv(DATASET_PATH, usecols=["track_genre"])["track_genre"]
        return {str(g).strip().lower().replace("-", " ") for g in genres.unique()}
    except Exception:
        return set()


DATASET_GENRES = _load_dataset_genres()
VALID_GENRES = BASE_GENRES | DATASET_GENRES


def validate_user_prefs(user_prefs: dict) -> Tuple[bool, str]:
    """Returns (is_valid, error_message). error_message is empty when valid."""
    genre = user_prefs.get("genre", "").strip().lower().replace("-", " ")
    mood = user_prefs.get("mood", "").strip().lower()
    energy = user_prefs.get("energy", -1)

    if not genre:
        return False, "Genre is required."
    if genre not in VALID_GENRES:
        return False, f"Unknown genre '{genre}'. Valid options: {sorted(VALID_GENRES)}"
    if not mood:
        return False, "Mood is required."
    if mood not in VALID_MOODS:
        return False, f"Unknown mood '{mood}'. Valid options: {sorted(VALID_MOODS)}"
    if not isinstance(energy, (int, float)) or not (0.0 <= float(energy) <= 1.0):
        return False, f"Energy must be a number between 0.0 and 1.0, got: {energy!r}"

    return True, ""
