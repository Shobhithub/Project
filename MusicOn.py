import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from ytmusicapi import YTMusic
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit.components.v1 as components
import gdown

# --- Load CSV from Google Drive ---
@st.cache_data
def load_data():
    file_id = "1hQXhG_HG5g1EkZAXex0Y0DQ4p4X0Qu7e"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "SpotifyFeatures.csv"
    gdown.download(url, output, quiet=False)
    df = pd.read_csv(output)
    return df

df = load_data()

# --- Feature Processing ---
features = df[
    ["danceability", "energy", "loudness", "speechiness", "acousticness",
     "instrumentalness", "liveness", "valence", "tempo"]
]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# --- Models ---
nn_model = NearestNeighbors(n_neighbors=6, algorithm="ball_tree")
nn_model.fit(scaled_features)

kmeans = KMeans(n_clusters=10, random_state=42)
df["cluster"] = kmeans.fit_predict(scaled_features)

# --- Spotify API Auth ---
sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id="23bb7ccddcc34607ae9c923bd05320d6",
        client_secret="3352827e3adf4155b23e8354eec3e5f9"
    )
)

ytmusic = YTMusic()

# --- Helper: Embed YouTube Video ---
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
st.title("🎧 Smart YouTube Music Recommender")

# --- Genre & Mood Filters ---
genres = sorted(df['genre'].dropna().unique())
selected_genre = st.selectbox("🎼 Filter by Genre (Optional):", ["All"] + genres)

mood = st.slider("🎭 Mood (Valence):", 0.0, 1.0, (0.0, 1.0))

# --- Filter DataFrame ---
filtered_df = df.copy()
if selected_genre != "All":
    filtered_df = filtered_df[filtered_df['genre'] == selected_genre]
filtered_df = filtered_df[(filtered_df["valence"] >= mood[0]) & (filtered_df["valence"] <= mood[1])]

# --- Song Selection with Suggestions ---
filtered_song_list = sorted(filtered_df.apply(lambda x: f"{x['track_name']} - {x['artist_name']}", axis=1).unique().tolist())
song_choice = st.selectbox("🔍 Search and select a song:", filtered_song_list)

# --- Random Button ---
if st.button("🎲 Surprise Me With a Random Song!"):
    song_choice = random.choice(filtered_song_list)
    st.success(f"Random pick: {song_choice}")

# --- Extract track and artist ---
try:
    song_name, artist_name = song_choice.split(" - ", 1)
except:
    song_name = song_choice
    artist_name = ""

# --- Search and Embed ---
if song_choice:
    search_query = f"{song_name} {artist_name}"
    results = ytmusic.search(search_query, filter="songs")
    if results:
        song = results[0]
        video_id = song["videoId"]
        st.markdown(f"### ▶️ {song_name} - {artist_name}")
        embed_youtube_video(video_id)
    else:
        st.warning("No YouTube Music result found.")

    try:
        index = df[
            (df["track_name"].str.lower() == song_name.lower()) &
            (df["artist_name"].str.lower() == artist_name.lower())
        ].index[0]

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
            except:
                st.warning("Error embedding video.")

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
            except:
                st.warning("Error embedding video.")

    except IndexError:
        st.error("⚠️ Song not found in dataset. Try another.")


