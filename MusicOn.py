import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from ytmusicapi import YTMusic
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit.components.v1 as components
import gdown

# --- Add background with blur and animated music icon ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, rgba(245, 245, 245, 0.5), rgba(220, 220, 220, 0.5));
        backdrop-filter: blur(8px);
    }

    .floating-icon {
        position: fixed;
        top: 10%;
        left: 10%;
        font-size: 60px;
        color: #ff4b4b;
        animation: float 3s ease-in-out infinite;
        opacity: 0.2;
        z-index: -1;
    }

    @keyframes float {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(15deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }
    </style>

    <div class="floating-icon">🎵</div>
""", unsafe_allow_html=True)

# --- Load Dataset from Google Drive ---
@st.cache_data
def load_data():
    file_id = "1hQXhG_HG5g1EkZAXex0Y0DQ4p4X0Qu7e"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "SpotifyFeatures.csv"
    gdown.download(url, output, quiet=False)
    df = pd.read_csv(output)
    return df

df = load_data()

# --- Feature Engineering ---
features = df[["danceability", "energy", "loudness", "speechiness",
               "acousticness", "instrumentalness", "liveness", "valence", "tempo"]]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# --- ML Models ---
nn_model = NearestNeighbors(n_neighbors=6, algorithm="ball_tree")
nn_model.fit(scaled_features)

kmeans = KMeans(n_clusters=10, random_state=42)
df["cluster"] = kmeans.fit_predict(scaled_features)

# --- YouTube + Spotify Auth ---
ytmusic = YTMusic()

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id="23bb7ccddcc34607ae9c923bd05320d6",
        client_secret="3352827e3adf4155b23e8354eec3e5f9"
    )
)

# --- Embed YouTube Video ---
def embed_youtube_video(video_id):
    youtube_url = f"https://www.youtube.com/embed/{video_id}"
    components.html(
        f"""
        <iframe width="100%" height="315" 
        src="{youtube_url}" 
        frameborder="0" allow="accelerometer; autoplay; 
        clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
        allowfullscreen></iframe>
        """,
        height=360,
    )

# --- Streamlit UI ---
st.set_page_config(page_title="🎵 Smart Music Player")
st.title("Data Science Project")
st.title("🎵 Smart Music Player")
st.markdown("Search for a song and get smart YouTube music recommendations 🎧")
st.text("Mentor: Himanshu Sardana Sir")

# --- Song Input ---
song_name = st.text_input("🎶 Enter a song name or singer name")

if song_name:
    results = ytmusic.search(song_name, filter="songs")
    if results:
        song = results[0]
        title = song["title"]
        artist = song["artists"][0]["name"]
        video_id = song["videoId"]

        st.markdown("## 🔍 You Searched")
        st.markdown(f"### 🎧 {title} - {artist}")
        embed_youtube_video(video_id)
    else:
        st.warning("No songs found.")

    try:
        index = df[df["track_name"].str.lower() == song_name.lower()].index[0]

        st.markdown("## 🔁 Recommended Songs")

        # --- Nearest Neighbors (2 songs) ---
        st.subheader("🧠 Similar by Audio Features")
        distances, indices = nn_model.kneighbors([scaled_features[index]])
        nn_count = 0
        for i in indices[0][1:]:
            if nn_count >= 2:
                break
            track = df.iloc[i]
            st.markdown(f"**{track['track_name']} - {track['artist_name']}**")
            try:
                yt_results = ytmusic.search(f"{track['track_name']} {track['artist_name']}", filter="songs")
                if yt_results:
                    video_id = yt_results[0]["videoId"]
                    embed_youtube_video(video_id)
                    nn_count += 1
            except Exception:
                st.warning("🎧 Error embedding video.")

        # --- KMeans Cluster (2 songs) ---
        st.subheader("🎯 Songs from Same Cluster")
        cluster_id = df.loc[index, "cluster"]
        similar_songs = df[df["cluster"] == cluster_id].sample(6)

        kmeans_count = 0
        for _, row in similar_songs.iterrows():
            if row["track_name"].lower() == song_name.lower():
                continue
            if kmeans_count >= 2:
                break
            st.markdown(f"**{row['track_name']} - {row['artist_name']}**")
            try:
                yt_results = ytmusic.search(f"{row['track_name']} {row['artist_name']}", filter="songs")
                if yt_results:
                    video_id = yt_results[0]["videoId"]
                    embed_youtube_video(video_id)
                    kmeans_count += 1
            except Exception:
                st.warning("🎧 Error embedding video.")

    except IndexError:
        st.error("⚠️ Song not found in dataset. Try a different title.")



