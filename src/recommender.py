import logging
from typing import List, Dict, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# The Kaggle Spotify file uses different column names, so we rename them
# to the names the rest of the app already uses.
KAGGLE_COLUMNS = {
    "track_name": "title",
    "artists": "artist",
    "track_genre": "genre",
    "tempo": "tempo_bpm",
}


def normalize_genre(name) -> str:
    # "indie-pop" and "Indie Pop" both become "indie pop"
    return str(name).strip().lower().replace("-", " ")


def mood_from_features(energy: float, valence: float, instrumentalness: float = 0.0) -> str:
    # Spotify has no mood column, so we work it out from the numbers.
    # valence = how happy a song sounds (0 to 1), energy = how intense (0 to 1)
    if instrumentalness >= 0.5 and energy < 0.6:
        return "focused"
    if energy >= 0.7:
        return "happy" if valence >= 0.5 else "intense"
    if energy < 0.4:
        return "relaxed" if valence >= 0.5 else "moody"
    return "chill" if valence >= 0.4 else "moody"


def song_genres(song: Dict) -> List[str]:
    # A song can have many genres, stored like "hip-hop, gospel"
    return [normalize_genre(g) for g in str(song["genre"]).split(",")]


def genre_matches(song: Dict, user_prefs: Dict) -> bool:
    return normalize_genre(user_prefs.get("genre", "")) in song_genres(song)


def _score_song(song: Dict, user_prefs: Dict) -> float:
    genre_score = 1.0 if genre_matches(song, user_prefs) else 0.0
    mood_score = 1.0 if str(song["mood"]).lower() == str(user_prefs.get("mood", "")).lower() else 0.0
    energy_score = 1.0 - abs(song["energy"] - user_prefs.get("energy", 0.5))
    acoustic_raw = song["acousticness"]
    acoustic_score = acoustic_raw if user_prefs.get("likes_acoustic", False) else (1.0 - acoustic_raw)

    return (0.35 * genre_score +
            0.30 * mood_score +
            0.25 * energy_score +
            0.10 * acoustic_score)


def _build_rule_explanation(song: Dict, user_prefs: Dict, score: float) -> str:
    parts = []
    if genre_matches(song, user_prefs):
        parts.append(f"matches your favorite genre ({user_prefs.get('genre')})")
    if str(song["mood"]).lower() == str(user_prefs.get("mood", "")).lower():
        parts.append(f"fits your '{song['mood']}' mood")
    energy_diff = abs(song["energy"] - user_prefs.get("energy", 0.5))
    if energy_diff < 0.1:
        parts.append("has very close energy to your target")
    elif energy_diff < 0.2:
        parts.append("has similar energy to your target")
    if not parts:
        parts.append("is a reasonable match for your overall taste")
    confidence = "High" if score >= 0.7 else "Medium" if score >= 0.45 else "Low"
    return f"[Confidence: {confidence}] " + "; ".join(parts).capitalize() + "."


def load_songs(csv_path: str) -> List[Dict]:
    """Loads either the small songs.csv or the real Kaggle Spotify dataset."""
    df = pd.read_csv(csv_path)

    if "track_genre" in df.columns:
        # Real Kaggle data: rename columns and drop empty rows
        df = df.rename(columns=KAGGLE_COLUMNS)
        df = df.dropna(subset=["title", "artist"])

        # The same song can appear once per genre. Keep one row per song,
        # but collect ALL of its genres, so "hip-hop" + "gospel" stays both.
        all_genres = df.groupby("track_id")["genre"].agg(lambda g: ", ".join(dict.fromkeys(g)))
        df = df.drop_duplicates(subset="track_id").copy()
        df["genre"] = df["track_id"].map(all_genres)
        df["artist"] = df["artist"].str.replace(";", ", ")
        df["tempo_bpm"] = df["tempo_bpm"].round().astype(int)
        df["mood"] = [
            mood_from_features(e, v, i)
            for e, v, i in zip(df["energy"], df["valence"], df["instrumentalness"])
        ]

    songs = df.to_dict(orient="records")
    logger.info(f"Loaded {len(songs)} songs from {csv_path}")
    return songs


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    if not songs:
        logger.warning("recommend_songs called with empty song list")
        return []

    k = max(1, min(k, len(songs)))

    scored = []
    for song in songs:
        score = _score_song(song, user_prefs)
        scored.append((song, round(score, 4)))

    # Best score first; popular songs win ties
    scored.sort(key=lambda x: (x[1], x[0].get("popularity", 0)), reverse=True)
    top = [(song, score, _build_rule_explanation(song, user_prefs, score)) for song, score in scored[:k]]
    logger.info(f"Top result: '{top[0][0]['title']}' score={top[0][1]}")
    return top
