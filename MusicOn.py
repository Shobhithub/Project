import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from ytmusicapi import YTMusic
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit.components.v1 as components

# Load dataset
import gdown

@st.cache_data
def load_data():
    file_id = "1hQXhG_HG5g1EkZAXex0Y0DQ4p4X0Qu7e"
# replace with your actual ID
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "SpotifyFeatures.csv"
    gdown.download(url, output, quiet=False)
    df = pd.read_csv(output)
    return df

df = load_data()


# Feature selection
features = df[
    ["danceability", "energy", "loudness", "speechiness", "acousticness",
     "instrumentalness", "liveness", "valence", "tempo"]
]

# Scaling features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Nearest Neighbors model
nn_model = NearestNeighbors(n_neighbors=6, algorithm="ball_tree")
nn_model.fit(scaled_features)

# KMeans clustering
kmeans = KMeans(n_clusters=10, random_state=42)
df["cluster"] = kmeans.fit_predict(scaled_features)

# Spotify auth (optional)
sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id="23bb7ccddcc34607ae9c923bd05320d6",
        client_secret="3352827e3adf4155b23e8354eec3e5f9"
    )
)

ytmusic = YTMusic()

# Function to embed YouTube video
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

# Streamlit UI
st.title("🎥 YouTube Music Player (Embed Version)")
song_name = st.text_input("Enter song name")

if song_name:
    results = ytmusic.search(song_name, filter="songs")
    if results:
        song = results[0]
        title = song["title"]
        artist = song["artists"][0]["name"]
        video_id = song["videoId"]

        st.markdown(f"### 🎧 {title} - {artist}")
        embed_youtube_video(video_id)
    else:
        st.warning("No songs found.")

    try:
        index = df[df["track_name"].str.lower() == song_name.lower()].index[0]

        st.subheader("🔁 Nearest Neighbors Recommendations")
        distances, indices = nn_model.kneighbors([scaled_features[index]])
        for i in indices[0][1:]:
            track = df.iloc[i]
            st.markdown(f"**{track['track_name']} - {track['artist_name']}**")

            try:
                yt_results = ytmusic.search(f"{track['track_name']} {track['artist_name']}", filter="songs")
                if yt_results:
                    video_id = yt_results[0]["videoId"]
                    embed_youtube_video(video_id)
                else:
                    st.info("No YouTube result found.")
            except Exception:
                st.warning("🎧 Error embedding video.")

        st.subheader("🎯 KMeans Cluster Recommendations")
        cluster_id = df.loc[index, "cluster"]
        similar_songs = df[df["cluster"] == cluster_id].sample(5)
        for _, row in similar_songs.iterrows():
            st.markdown(f"**{row['track_name']} - {row['artist_name']}**")
            try:
                yt_results = ytmusic.search(f"{row['track_name']} {row['artist_name']}", filter="songs")
                if yt_results:
                    video_id = yt_results[0]["videoId"]
                    embed_youtube_video(video_id)
                else:
                    st.info("No YouTube result.")
            except Exception:
                st.warning("🎧 Error embedding video.")

    except IndexError:
        st.error("Song not found in dataset. Try a different name.")

