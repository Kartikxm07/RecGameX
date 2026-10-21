# Respawned 🎮 — NLP-Powered Game Discovery & Recommendation

**Respawned** is a game discovery and recommendation application that uses Natural Language Processing (NLP) techniques to identify games with similar characteristics. By analyzing game descriptions and metadata, it helps users explore titles related to their interests.

The system uses a Kaggle dataset of **26,000+ games** and combines TF-IDF feature extraction with cosine similarity to find and rank games based on metadata similarity. An interactive Gradio interface makes it easy to search for a game and discover related titles.

## ✨ Key Features

- **Similar Game Suggestions:** Recommends games with metadata similar to a selected title.
- **Text-Based Feature Extraction:** Uses TF-IDF to transform game attributes into numerical representations.
- **Similarity Ranking:** Applies cosine similarity to identify the closest matches.
- **Large Dataset Support:** Works with a collection of over 26,000 games.
- **Interactive Search Interface:** Provides a user-friendly web interface powered by Gradio.
- **Game Artwork Integration:** Displays game header images with fallback handling when images are unavailable.
- **Gaming-Inspired UI:** Features a customized interface with gaming-themed styling and CSS.

## ⚙️ How the System Works

1. **Metadata Preparation:** Relevant attributes, including genres, categories, tags, and developer information, are combined into a unified text representation for each game.
2. **Text Vectorization:** A custom semicolon-based analyzer is used with `TfidfVectorizer` to convert the combined metadata into numerical feature vectors.
3. **Similarity Calculation:** Cosine similarity is used to measure the closeness between game representations.
4. **Game Search:** When a user enters a title, the application identifies the best matching game using partial title matching.
5. **Recommendation Generation:** Candidate games are ranked by their similarity scores, and the top-N results are displayed along with their images.

## 🛠️ Technologies Used

- **Python** — Core implementation
- **Pandas** — Dataset processing and manipulation
- **Scikit-learn** — TF-IDF vectorization and cosine similarity
- **Gradio** — Interactive web application
- **HTML & CSS** — Interface customization and visual styling

## 📊 Dataset Overview

The project uses a game metadata dataset sourced from Kaggle, containing **26,000+ game records**.

The dataset includes the following information:

- Game titles
- Genre information
- Game categories
- SteamSpy tags
- Developer names
- Header image URLs

These attributes provide the textual information required to calculate game similarity and generate recommendations.

## 🎯 Project Objective

The goal of Respawned is to make game discovery easier by helping users find titles related to games they already enjoy. Instead of relying on user ratings or collaborative filtering, the system focuses on similarities between game metadata using NLP-based feature extraction.

---

*Respawned demonstrates how traditional NLP techniques can be applied to build a practical content-based recommendation system.*