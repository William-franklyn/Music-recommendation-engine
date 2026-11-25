import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import streamlit.components.v1 as components

from ai_explainer import get_ai_explanation
from guardrails import VALID_GENRES, VALID_MOODS, validate_user_prefs
from logger_config import setup_logging
from recommender import load_songs, recommend_songs

setup_logging()

# Use the real Spotify dataset if it's there, otherwise the small sample file
DATA_PATH = "data/dataset.csv" if os.path.exists("data/dataset.csv") else "data/songs.csv"


@st.cache_data(show_spinner="Loading songs...")
def get_songs(path):
    # Loads the songs once, then reuses them on every click
    return load_songs(path)


st.set_page_config(page_title="VibeFinder", page_icon="🎵", layout="wide")
st.title("VibeFinder — AI Music Recommender")
st.caption("Powered by a weighted scoring engine on real Spotify data + Claude AI explanations (RAG)")

songs = get_songs(DATA_PATH)

with st.sidebar:
    st.header("Your Taste Profile")
    st.caption(f"Searching {len(songs):,} songs")
    genres = sorted(VALID_GENRES)
    genre = st.selectbox("Favorite Genre", genres, index=genres.index("pop") if "pop" in genres else 0)
    mood = st.selectbox("Favorite Mood", sorted(VALID_MOODS))
    energy = st.slider(
        "Target Energy Level", 0.0, 1.0, 0.7, step=0.05,
        help="0 = very calm, 1 = very intense"
    )
    likes_acoustic = st.checkbox("I prefer acoustic sounds", value=False)
    k = st.slider("Number of recommendations", 1, 10, 5)
    use_ai = st.checkbox("Generate AI explanation with Claude", value=True)
    run_btn = st.button("Find My Songs", type="primary")

if run_btn:
    user_prefs = {
        "genre": genre,
        "mood": mood,
        "energy": energy,
        "likes_acoustic": likes_acoustic,
    }

    is_valid, err = validate_user_prefs(user_prefs)
    if not is_valid:
        st.error(f"Input error: {err}")
        st.stop()

    results = recommend_songs(user_prefs, songs, k=k)

    st.subheader(f"Top {len(results)} Recommendations for You")

    for rank, (song, score, rule_explanation) in enumerate(results, 1):
        confidence = "High" if score >= 0.7 else "Medium" if score >= 0.45 else "Low"
        badge_color = {"High": "green", "Medium": "orange", "Low": "red"}[confidence]

        with st.expander(
            f"#{rank} — {song['title']} by {song['artist']}  |  Score: {score:.2f}",
            expanded=(rank == 1),
        ):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**Genre:** {song['genre']}  |  **Mood:** {song['mood']}")
                st.write(
                    f"**Energy:** {song['energy']:.2f}  |  "
                    f"**Acousticness:** {song['acousticness']:.2f}"
                )
                st.write(
                    f"**Tempo:** {song['tempo_bpm']} BPM  |  "
                    f"**Danceability:** {song['danceability']:.2f}"
                )
                st.write(f"**Why this song:** {rule_explanation}")
                if song.get("track_id"):
                    # Free Spotify player: plays a preview, or the full song if logged in
                    components.iframe(
                        f"https://open.spotify.com/embed/track/{song['track_id']}",
                        height=80,
                    )
            with col2:
                st.markdown(f"**Confidence:** :{badge_color}[{confidence}]")
                st.metric("Match Score", f"{score:.2f}")

    if use_ai:
        st.divider()
        st.subheader("Claude's Take on Your Playlist")
        with st.spinner("Asking Claude..."):
            ai_text = get_ai_explanation(user_prefs, results)
        st.info(ai_text)
else:
    st.info("Set your preferences in the sidebar and click **Find My Songs** to start.")
