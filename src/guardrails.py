import os
from collections import Counter
from typing import Tuple

import pandas as pd

# The original genres, kept so older tests and songs.csv still work
BASE_GENRES = {"pop", "lofi", "rock", "ambient", "jazz", "synthwave", "indie pop"}
VALID_MOODS = {"happy", "chill", "intense", "relaxed", "moody", "focused"}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PARQUET_PATH = os.path.join(DATA_DIR, "songs.parquet")
OLD_CSV_PATH = os.path.join(DATA_DIR, "dataset.csv")

# A genre goes in the menu (and in Llama's list) if at least this many songs have it
MIN_SONGS_PER_GENRE = 1500

# Smaller genres people often ask for, added even though they have fewer songs
EXTRA_GENRES = {
    "christmas", "worship", "ccm", "christian music", "gospel", "christian hip hop",
    "afrobeat", "afropop", "amapiano", "bongo flava", "lo fi", "ambient", "house",
    "techno", "dubstep", "drum and bass", "indie pop", "indie folk", "k pop",
    "anime", "video game music", "children's music", "meditation", "sleep",
}


def _load_dataset_genres() -> set:
    try:
        if os.path.exists(PARQUET_PATH):
            # Every song has a list of genres like "worship, ccm, christian music"
            tags = pd.read_parquet(PARQUET_PATH, columns=["genre"])["genre"]
            counts = Counter(g for line in tags for g in line.split(", "))
            big = {g for g, n in counts.items() if n >= MIN_SONGS_PER_GENRE}
            extras = {g for g in EXTRA_GENRES if counts.get(g, 0) > 0}
            return big | extras
        if os.path.exists(OLD_CSV_PATH):
            genres = pd.read_csv(OLD_CSV_PATH, usecols=["track_genre"])["track_genre"]
            return {str(g).strip().lower().replace("-", " ") for g in genres.unique()}
    except Exception:
        pass
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
