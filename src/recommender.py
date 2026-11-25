import logging
import re
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The older Kaggle file (114k songs) uses different column names,
# so we rename them to the names the rest of the app uses.
KAGGLE_COLUMNS = {
    "track_name": "title",
    "artists": "artist",
    "track_genre": "genre",
    "tempo": "tempo_bpm",
}

# The scoring weights (they add up to 1.0)
WEIGHTS = {"genre": 0.35, "mood": 0.25, "energy": 0.20, "acoustic": 0.05, "popularity": 0.15}

# Where each mood sits on the (energy, valence) map. valence = how happy a song sounds.
MOOD_TARGETS = {
    "happy": (0.80, 0.80),
    "intense": (0.85, 0.25),
    "chill": (0.45, 0.55),
    "relaxed": (0.30, 0.65),
    "moody": (0.40, 0.20),
    "focused": (0.40, 0.45),
}
MOOD_RANGE = 0.6  # songs farther than this from the mood's spot get 0 mood points


def normalize_genre(name) -> str:
    # "Hip-Hop" becomes "hip hop", and "lofi" becomes "lo fi" (how Spotify writes it)
    text = str(name).strip().lower().replace("-", " ")
    return re.sub(r"\blofi\b", "lo fi", text)


def _genre_pattern(genre: str) -> str:
    # Matches the genre as whole words: "rock" finds "classic rock" but "rap" does not find "trap"
    return r"(?<![a-z0-9])" + re.escape(normalize_genre(genre)) + r"(?![a-z0-9])"


def mood_from_features(energy: float, valence: float, instrumentalness: float = 0.0) -> str:
    # A simple label for each song, shown on screen
    if instrumentalness >= 0.5 and energy < 0.6:
        return "focused"
    if energy >= 0.7:
        return "happy" if valence >= 0.5 else "intense"
    if energy < 0.4:
        return "relaxed" if valence >= 0.5 else "moody"
    return "chill" if valence >= 0.4 else "moody"


def _moods_for_table(df: pd.DataFrame) -> np.ndarray:
    # Same rules as mood_from_features, but for every song at once (much faster)
    e, v = df["energy"], df["valence"]
    i = df["instrumentalness"] if "instrumentalness" in df else 0.0
    return np.select(
        [(i >= 0.5) & (e < 0.6), (e >= 0.7) & (v >= 0.5), e >= 0.7, (e < 0.4) & (v >= 0.5), e < 0.4, v >= 0.4],
        ["focused", "happy", "intense", "relaxed", "moody", "chill"],
        default="moody",
    )


def mood_closeness(energy, valence, mood: str):
    """NEW: partial mood points. 1.0 = right on the mood, 0.0 = far away.
    Before, a song got all the mood points or none."""
    target = MOOD_TARGETS.get(str(mood).lower())
    if target is None:
        return 0.0 * energy
    distance = np.sqrt((energy - target[0]) ** 2 + (valence - target[1]) ** 2)
    return np.clip(1.0 - distance / MOOD_RANGE, 0.0, 1.0)


def genre_matches(song: Dict, user_prefs: Dict) -> bool:
    return re.search(_genre_pattern(user_prefs.get("genre", "")), normalize_genre(song["genre"])) is not None


def _build_rule_explanation(song: Dict, user_prefs: Dict, score: float) -> str:
    parts = []
    if genre_matches(song, user_prefs):
        parts.append(f"matches your favorite genre ({user_prefs.get('genre')})")
    closeness = float(mood_closeness(song["energy"], song.get("valence", 0.5), user_prefs.get("mood", "")))
    if closeness >= 0.7:
        parts.append(f"fits your '{user_prefs.get('mood')}' mood")
    elif closeness >= 0.4:
        parts.append(f"is close to your '{user_prefs.get('mood')}' mood")
    energy_diff = abs(song["energy"] - user_prefs.get("energy", 0.5))
    if energy_diff < 0.1:
        parts.append("has very close energy to your target")
    elif energy_diff < 0.2:
        parts.append("has similar energy to your target")
    if song.get("popularity", 0) >= 60:
        parts.append("is a popular song")
    if not parts:
        parts.append("is a reasonable match for your overall taste")
    confidence = "High" if score >= 0.7 else "Medium" if score >= 0.45 else "Low"
    return f"[Confidence: {confidence}] " + "; ".join(parts).capitalize() + "."


def load_songs(path: str) -> Union[List[Dict], pd.DataFrame]:
    """Loads the songs.

    - data/songs.parquet: the big Spotify file (about 350,000 songs), returned as a table
    - any .csv file: the small sample or the older Kaggle file, returned as a list of dicts
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        df["mood"] = _moods_for_table(df)
        logger.info(f"Loaded {len(df)} songs from {path}")
        return df

    df = pd.read_csv(path)
    if "track_genre" in df.columns:
        # Older Kaggle data: rename columns and drop empty rows
        df = df.rename(columns=KAGGLE_COLUMNS)
        df = df.dropna(subset=["title", "artist"])
        # The same song can appear once per genre. Keep one row per song, with ALL its genres.
        all_genres = df.groupby("track_id")["genre"].agg(lambda g: ", ".join(dict.fromkeys(g)))
        df = df.drop_duplicates(subset="track_id").copy()
        df["genre"] = df["track_id"].map(all_genres)
        df["artist"] = df["artist"].str.replace(";", ", ")
        df["tempo_bpm"] = df["tempo_bpm"].round().astype(int)
        df["mood"] = _moods_for_table(df)

    songs = df.to_dict(orient="records")
    logger.info(f"Loaded {len(songs)} songs from {path}")
    return songs


def score_table(user_prefs: Dict, df: pd.DataFrame) -> pd.Series:
    """The weighted sum, done for every song at once."""
    genre_score = (
        df["genre"].map(normalize_genre).str.contains(_genre_pattern(user_prefs.get("genre", "")), regex=True)
    ).astype(float)
    valence = df["valence"] if "valence" in df else 0.5
    mood_score = mood_closeness(df["energy"], valence, user_prefs.get("mood", ""))
    energy_score = 1.0 - (df["energy"] - user_prefs.get("energy", 0.5)).abs()
    acoustic = df["acousticness"]
    acoustic_score = acoustic if user_prefs.get("likes_acoustic", False) else 1.0 - acoustic
    popularity_score = (df["popularity"] / 100.0).clip(0, 1) if "popularity" in df else 0.0

    return (WEIGHTS["genre"] * genre_score +
            WEIGHTS["mood"] * mood_score +
            WEIGHTS["energy"] * energy_score +
            WEIGHTS["acoustic"] * acoustic_score +
            WEIGHTS["popularity"] * popularity_score).round(4)


def recommend_songs(
    user_prefs: Dict, songs: Union[List[Dict], pd.DataFrame], k: int = 5
) -> List[Tuple[Dict, float, str]]:
    df = songs if isinstance(songs, pd.DataFrame) else pd.DataFrame(songs)
    if df.empty:
        logger.warning("recommend_songs called with no songs")
        return []

    k = max(1, min(k, len(df)))
    scores = score_table(user_prefs, df)

    top = []
    seen = set()
    for idx in scores.sort_values(ascending=False, kind="stable").index:
        song = df.loc[idx].to_dict()
        # Skip the same song twice (for example two versions of "Last Christmas")
        key = (str(song["title"]).lower(), str(song["artist"]).lower())
        if key in seen:
            continue
        seen.add(key)
        score = float(scores[idx])
        top.append((song, score, _build_rule_explanation(song, user_prefs, score)))
        if len(top) == k:
            break

    logger.info(f"Top result: '{top[0][0]['title']}' score={top[0][1]}")
    return top
