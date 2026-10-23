import hashlib
import html
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "augmented_with_images_alternative_api.csv"
DB_PATH = BASE_DIR / "respawned_users.db"
REQUIRED_COLUMNS = ["appid", "name", "genres", "categories", "steamspy_tags", "developer", "header_image_url"]
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_]{3,24}$")
POSITIVE_REVIEW_WORDS = {"good", "great", "amazing", "fun", "love", "best", "enjoy", "awesome", "smooth"}
NEGATIVE_REVIEW_WORDS = {"bad", "boring", "buggy", "hate", "worst", "slow", "broken", "lag", "poor"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_appid(raw_value, fallback_index: int) -> str:
    if pd.isna(raw_value):
        return f"row-{fallback_index}"

    try:
        return str(int(float(raw_value)))
    except (TypeError, ValueError):
        normalized = str(raw_value).strip()
        return normalized if normalized else f"row-{fallback_index}"


def guest_state() -> dict:
    return {"user_id": None, "username": "", "display_name": "Guest"}


# ------------------------------
# Load data
# ------------------------------
if not CSV_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at: {CSV_PATH}")

df = pd.read_csv(CSV_PATH, encoding="utf-8")

for col in REQUIRED_COLUMNS:
    if col not in df.columns:
        df[col] = ""

for col in ["name", "genres", "categories", "steamspy_tags", "developer", "header_image_url"]:
    df[col] = df[col].fillna("")

raw_appids = [normalize_appid(value, idx) for idx, value in enumerate(df["appid"].tolist())]
seen_appids = set()
normalized_appids = []
for idx, appid in enumerate(raw_appids):
    deduped_appid = appid if appid not in seen_appids else f"{appid}-{idx}"
    seen_appids.add(deduped_appid)
    normalized_appids.append(deduped_appid)

df["appid"] = normalized_appids

# ------------------------------
# Helpers / Cleaning
# ------------------------------
def clean_developer_name(dev_string):
    if pd.isna(dev_string):
        return ""
    if not isinstance(dev_string, str):
        dev_string = str(dev_string)
    dev_string = dev_string.replace("|", ";").replace(",", ";").replace("/", ";")
    parts = [re.sub(r"\s+", "", p.strip()) for p in dev_string.split(";") if p.strip()]
    return ";".join(parts)

df["developer_tags"] = df["developer"].apply(clean_developer_name)

df["name"] = df["name"].astype(str).str.replace(r"[^a-zA-Z:]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
df["metadata"] = (df["genres"] + ";" + df["categories"] + ";" + df["steamspy_tags"] + ";" + df["developer_tags"])

# ------------------------------
# TF-IDF & similarity
# ------------------------------
def semicolon_analyzer(text):
    return [token for token in {token.strip().lower() for token in str(text).split(";")} if token]

tfidf_vectorizer = TfidfVectorizer(analyzer=semicolon_analyzer)
tfidf_matrix = tfidf_vectorizer.fit_transform(df["metadata"])
cosine_sim = cosine_similarity(tfidf_matrix, dense_output=False)

INDEX_TO_APPID = df["appid"].to_dict()
APPID_TO_INDEX = {appid: idx for idx, appid in INDEX_TO_APPID.items()}
APPID_TO_NAME = df.set_index("appid")["name"].to_dict()
GAME_TOKENS = [set(semicolon_analyzer(metadata_text)) for metadata_text in df["metadata"].tolist()]


# ------------------------------
# User data store
# ------------------------------
def get_db_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def init_db():
    with get_db_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS saved_games (
                user_id INTEGER NOT NULL,
                appid TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (user_id, appid),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS ratings (
                user_id INTEGER NOT NULL,
                appid TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, appid),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                appid TEXT NOT NULL,
                review_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS interaction_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                appid TEXT,
                details TEXT,
                rating INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """
        )


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def get_user_id(user_state: dict) -> int | None:
    if not user_state:
        return None
    return user_state.get("user_id")


def is_authenticated(user_state: dict | None) -> bool:
    return get_user_id(user_state or {}) is not None


def shell_visibility_updates(user_state: dict):
    logged_in = is_authenticated(user_state)
    return gr.update(visible=not logged_in), gr.update(visible=logged_in)


def user_badge_text(user_state: dict) -> str:
    if get_user_id(user_state):
        safe_display = html.escape(user_state.get("display_name", "Player"))
        safe_username = html.escape(user_state.get("username", ""))
        return f"### Active Profile: {safe_display} (@{safe_username})"
    return "### Active Profile: Guest (anonymous browsing)"


def feedback_hint_text(user_state: dict) -> str:
    if get_user_id(user_state):
        return "You are signed in. Save, rate, and review games to personalize recommendations."
    return "Create a profile or sign in to save games, submit ratings, and build a personalized feedback loop."


def log_event(user_id: int | None, event_type: str, appid: str | None = None, details: str = "", rating: int | None = None):
    if not user_id:
        return

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO interaction_events (user_id, event_type, appid, details, rating, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, event_type, appid, details.strip(), rating, utc_now()),
        )


def create_account(username: str, password: str, display_name: str, current_state: dict):
    state = current_state or guest_state()
    normalized_username = (username or "").strip().lower()
    display = (display_name or "").strip()

    if not USERNAME_PATTERN.fullmatch(normalized_username):
        return "Username must be 3-24 characters (letters, numbers, underscore).", state
    if len(password or "") < 4:
        return "Password must be at least 4 characters.", state
    if not display:
        display = normalized_username

    try:
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (username, password_hash, display_name, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (normalized_username, hash_password(password), display, utc_now()),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        return "Username already exists. Choose another username.", state

    new_state = {"user_id": user_id, "username": normalized_username, "display_name": display}
    log_event(user_id, "register", details="Profile created")
    return f"Profile created. Signed in as {display}.", new_state


def login_account(username: str, password: str, current_state: dict):
    state = current_state or guest_state()
    normalized_username = (username or "").strip().lower()

    if not normalized_username or not password:
        return "Enter both username and password.", state

    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT id, username, display_name, password_hash
            FROM users
            WHERE username = ?
            """,
            (normalized_username,),
        ).fetchone()

    if not row or row["password_hash"] != hash_password(password):
        return "Invalid username or password.", state

    new_state = {"user_id": row["id"], "username": row["username"], "display_name": row["display_name"]}
    log_event(new_state["user_id"], "login", details="Signed in")
    return f"Welcome back, {new_state['display_name']}.", new_state


def logout_account(current_state: dict):
    state = current_state or guest_state()
    user_id = get_user_id(state)
    if user_id:
        log_event(user_id, "logout", details="Signed out")
    return "You are now signed out.", guest_state()

def get_best_match(partial_name: str):
    if not partial_name:
        return None

    normalized_query = re.sub(r"\s+", " ", str(partial_name)).strip()
    if not normalized_query:
        return None

    matches = df[df["name"].str.contains(normalized_query, case=False, na=False)]
    if matches.empty:
        return None

    return matches.iloc[0]["name"]

# ------------------------------
# Image resolver
# ------------------------------
def get_image_url(idx: int, name: str) -> str:
    url = df.at[idx, "header_image_url"]
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        return url
    return f"https://placehold.co/300x168?text={quote(name)}"


def review_sentiment_score(review_text: str) -> float:
    tokens = re.findall(r"[a-zA-Z]+", (review_text or "").lower())
    if not tokens:
        return 0.0

    positive_hits = sum(1 for token in tokens if token in POSITIVE_REVIEW_WORDS)
    negative_hits = sum(1 for token in tokens if token in NEGATIVE_REVIEW_WORDS)
    return float(positive_hits - negative_hits)


def add_game_tokens(token_weights: defaultdict, appid: str, scale: float):
    if scale == 0:
        return

    index = APPID_TO_INDEX.get(str(appid))
    if index is None:
        return

    for token in GAME_TOKENS[index]:
        token_weights[token] += float(scale)


def get_user_token_weights(user_id: int) -> defaultdict:
    token_weights = defaultdict(float)

    with get_db_connection() as conn:
        saved_rows = conn.execute("SELECT appid FROM saved_games WHERE user_id = ?", (user_id,)).fetchall()
        rating_rows = conn.execute("SELECT appid, rating FROM ratings WHERE user_id = ?", (user_id,)).fetchall()
        review_rows = conn.execute("SELECT appid, review_text FROM reviews WHERE user_id = ?", (user_id,)).fetchall()

    for row in saved_rows:
        add_game_tokens(token_weights, row["appid"], 1.2)

    for row in rating_rows:
        rating_scale = float(row["rating"] - 3)
        add_game_tokens(token_weights, row["appid"], rating_scale)

    for row in review_rows:
        sentiment = review_sentiment_score(row["review_text"])
        if sentiment:
            add_game_tokens(token_weights, row["appid"], 0.4 * sentiment)

    return token_weights


def apply_personalization(ranked: list, user_state: dict) -> list:
    user_id = get_user_id(user_state)
    if not user_id:
        return ranked

    token_weights = get_user_token_weights(user_id)
    if not token_weights:
        return ranked

    adjusted = []
    for index, base_score in ranked:
        token_bonus = sum(token_weights.get(token, 0.0) for token in GAME_TOKENS[index])
        final_score = float(base_score) + (0.03 * token_bonus)
        adjusted.append((index, final_score))

    adjusted.sort(key=lambda item: (-item[1], item[0]))
    return adjusted


def recommendation_payload_from_ranked(ranked: list) -> list:
    recommendations = []
    for index, score in ranked:
        appid = INDEX_TO_APPID.get(index, f"row-{index}")
        game_name = str(df["name"].iat[index])
        image_url = get_image_url(index, game_name)
        recommendations.append({"appid": appid, "name": game_name, "image_url": image_url, "score": float(score)})
    return recommendations


def build_gallery_html(recommendations: list) -> str:
    cards = []
    for rec in recommendations:
        safe_name = html.escape(rec["name"])
        cards.append(
            f"""
            <div class="game-card">
                <img src="{rec['image_url']}" alt="{safe_name}" class="game-image">
                <div class="game-title-overlay">
                    <span>{safe_name}</span>
                </div>
            </div>
            """
        )

    return f"<div class=\"gallery-container\">{''.join(cards)}</div>"


def recommendation_choices(recommendations: list) -> list:
    return [(rec["name"], rec["appid"]) for rec in recommendations]


def resolve_selected_game(selected_appid: str):
    if not selected_appid:
        return None, None

    appid = str(selected_appid)
    game_name = APPID_TO_NAME.get(appid)
    if not game_name:
        return None, None

    return appid, game_name


def summarize_terms_from_weights(game_weights: dict, column_name: str, top_n: int = 5) -> str:
    counter = Counter()

    for appid, weight in game_weights.items():
        if weight <= 0:
            continue
        index = APPID_TO_INDEX.get(str(appid))
        if index is None:
            continue

        values = [value.strip() for value in str(df.at[index, column_name]).split(";") if value.strip()]
        for value in values:
            counter[value] += float(weight)

    if not counter:
        return "Not enough feedback yet"

    top_items = counter.most_common(top_n)
    return ", ".join([f"{html.escape(name)} ({score:.1f})" for name, score in top_items])


def get_user_game_weights(user_id: int) -> defaultdict:
    game_weights = defaultdict(float)

    with get_db_connection() as conn:
        saved_rows = conn.execute("SELECT appid FROM saved_games WHERE user_id = ?", (user_id,)).fetchall()
        rating_rows = conn.execute("SELECT appid, rating FROM ratings WHERE user_id = ?", (user_id,)).fetchall()
        review_rows = conn.execute("SELECT appid, review_text FROM reviews WHERE user_id = ?", (user_id,)).fetchall()

    for row in saved_rows:
        game_weights[row["appid"]] += 1.5

    for row in rating_rows:
        game_weights[row["appid"]] += float(row["rating"] - 3)

    for row in review_rows:
        game_weights[row["appid"]] += 0.5 * review_sentiment_score(row["review_text"])

    return game_weights


def format_event_row(row: sqlite3.Row) -> str:
    timestamp = (row["created_at"] or "").replace("T", " ").replace("+00:00", " UTC")
    event_type = row["event_type"]
    appid = row["appid"]
    game_name = APPID_TO_NAME.get(str(appid), "Unknown game") if appid else ""
    details = row["details"] or ""

    if event_type == "search":
        message = f"Searched for '{details}'"
    elif event_type == "save":
        message = f"Saved {game_name}"
    elif event_type == "rate":
        message = f"Rated {game_name}: {row['rating']}/5"
    elif event_type == "review":
        message = f"Reviewed {game_name}: {details}"
    elif event_type == "register":
        message = "Created profile"
    elif event_type == "login":
        message = "Signed in"
    elif event_type == "logout":
        message = "Signed out"
    else:
        message = details if details else event_type

    return f"{html.escape(timestamp)} - {html.escape(message)}"


def render_profile_panel(user_state: dict) -> str:
    user_id = get_user_id(user_state)
    if not user_id:
        return """
        <div style='padding:18px;border:1px solid #30363d;border-radius:10px;background:rgba(22,27,34,0.75);color:#f0f6fc;'>
            <h3 style='margin-top:0;color:#00ff88;'>Profile Summary</h3>
            <p>Sign in to see saved games, ratings, reviews, and personalization signals.</p>
        </div>
        """

    with get_db_connection() as conn:
        user_row = conn.execute(
            "SELECT username, display_name, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        saved_count = conn.execute("SELECT COUNT(*) AS count FROM saved_games WHERE user_id = ?", (user_id,)).fetchone()["count"]
        rating_stats = conn.execute(
            "SELECT COUNT(*) AS count, AVG(rating) AS avg_rating FROM ratings WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        review_count = conn.execute("SELECT COUNT(*) AS count FROM reviews WHERE user_id = ?", (user_id,)).fetchone()["count"]
        recent_rows = conn.execute(
            """
            SELECT event_type, appid, details, rating, created_at
            FROM interaction_events
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 5
            """,
            (user_id,),
        ).fetchall()

    game_weights = get_user_game_weights(user_id)
    favorite_genres = summarize_terms_from_weights(game_weights, "genres")
    favorite_developers = summarize_terms_from_weights(game_weights, "developer")

    avg_rating = rating_stats["avg_rating"] if rating_stats and rating_stats["avg_rating"] is not None else 0
    rating_count = rating_stats["count"] if rating_stats else 0
    recent_events_html = "".join([f"<li>{format_event_row(row)}</li>" for row in recent_rows])
    if not recent_events_html:
        recent_events_html = "<li>No activity yet.</li>"

    return f"""
    <div style='padding:18px;border:1px solid #30363d;border-radius:10px;background:rgba(22,27,34,0.75);color:#f0f6fc;'>
        <h3 style='margin-top:0;color:#00ff88;'>Profile Summary</h3>
        <p><strong>Name:</strong> {html.escape(user_row['display_name'])} (@{html.escape(user_row['username'])})</p>
        <p><strong>Created:</strong> {html.escape((user_row['created_at'] or '').replace('T', ' ').replace('+00:00', ' UTC'))}</p>
        <p><strong>Saved Games:</strong> {saved_count} | <strong>Ratings:</strong> {rating_count} | <strong>Reviews:</strong> {review_count}</p>
        <p><strong>Average Rating:</strong> {avg_rating:.2f}/5</p>
        <p><strong>Favorite Genres:</strong> {favorite_genres}</p>
        <p><strong>Favorite Developers:</strong> {favorite_developers}</p>
        <h4 style='color:#ff8c42;'>Recent Activity</h4>
        <ul>{recent_events_html}</ul>
    </div>
    """


def render_history_panel(user_state: dict) -> str:
    user_id = get_user_id(user_state)
    if not user_id:
        return """
        <div style='padding:18px;border:1px solid #30363d;border-radius:10px;background:rgba(22,27,34,0.75);color:#f0f6fc;'>
            <h3 style='margin-top:0;color:#00ff88;'>History Timeline</h3>
            <p>Sign in to track search history, saved games, ratings, and reviews.</p>
        </div>
        """

    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT event_type, appid, details, rating, created_at
            FROM interaction_events
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 40
            """,
            (user_id,),
        ).fetchall()

    if not rows:
        return """
        <div style='padding:18px;border:1px solid #30363d;border-radius:10px;background:rgba(22,27,34,0.75);color:#f0f6fc;'>
            <h3 style='margin-top:0;color:#00ff88;'>History Timeline</h3>
            <p>No actions yet. Start exploring games and adding feedback.</p>
        </div>
        """

    entries = "".join([f"<li>{format_event_row(row)}</li>" for row in rows])
    return f"""
    <div style='padding:18px;border:1px solid #30363d;border-radius:10px;background:rgba(22,27,34,0.75);color:#f0f6fc;'>
        <h3 style='margin-top:0;color:#00ff88;'>History Timeline</h3>
        <ul>{entries}</ul>
    </div>
    """


def recommend_game(title: str, top_n: int = 5, user_state: dict | None = None):
    state = user_state or guest_state()
    if not is_authenticated(state):
        return (
            "Please sign in to use recommendations.",
            "",
            gr.update(choices=[], value=None),
            feedback_hint_text(state),
            [],
        )

    best_match = get_best_match(title)
    if not best_match:
        return (
            f"No game found for input: '{title}'. Try another title.",
            "",
            gr.update(choices=[], value=None),
            feedback_hint_text(state),
            [],
        )

    index = df.index[df["name"] == best_match][0]
    sim_scores = cosine_sim[index].toarray().ravel() if hasattr(cosine_sim[index], "toarray") else cosine_sim[index]
    ranked = sorted(
        [(i, float(score)) for i, score in enumerate(sim_scores) if i != index],
        key=lambda item: (-item[1], item[0]),
    )
    ranked = apply_personalization(ranked, state)

    try:
        validated_top_n = int(top_n)
    except (TypeError, ValueError):
        validated_top_n = 5

    validated_top_n = max(1, min(validated_top_n, len(ranked)))
    recommendations = recommendation_payload_from_ranked(ranked[:validated_top_n])

    user_id = get_user_id(state)
    if user_id:
        log_event(user_id, "search", appid=INDEX_TO_APPID.get(index), details=str(title).strip() or best_match)

    choices = recommendation_choices(recommendations)
    default_choice = choices[0][1] if choices else None
    label_text = f"Recommendations for: {best_match}"
    return label_text, build_gallery_html(recommendations), gr.update(choices=choices, value=default_choice), feedback_hint_text(state), recommendations


def save_selected_game(selected_appid: str, user_state: dict):
    state = user_state or guest_state()
    user_id = get_user_id(state)
    if not user_id:
        return "Sign in to save games.", render_profile_panel(state), render_history_panel(state)

    appid, game_name = resolve_selected_game(selected_appid)
    if not appid:
        return "Select a recommended game first.", render_profile_panel(state), render_history_panel(state)

    with get_db_connection() as conn:
        before_changes = conn.total_changes
        conn.execute(
            """
            INSERT OR IGNORE INTO saved_games (user_id, appid, created_at)
            VALUES (?, ?, ?)
            """,
            (user_id, appid, utc_now()),
        )
        was_added = conn.total_changes > before_changes

    if was_added:
        log_event(user_id, "save", appid=appid, details=f"Saved {game_name}")
        message = f"Saved {game_name}."
    else:
        message = f"{game_name} is already in your saved list."

    return message, render_profile_panel(state), render_history_panel(state)


def rate_selected_game(selected_appid: str, rating_value: float, user_state: dict):
    state = user_state or guest_state()
    user_id = get_user_id(state)
    if not user_id:
        return "Sign in to submit ratings.", render_profile_panel(state), render_history_panel(state)

    appid, game_name = resolve_selected_game(selected_appid)
    if not appid:
        return "Select a recommended game first.", render_profile_panel(state), render_history_panel(state)

    safe_rating = max(1, min(int(rating_value), 5))
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO ratings (user_id, appid, rating, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, appid)
            DO UPDATE SET rating = excluded.rating, updated_at = excluded.updated_at
            """,
            (user_id, appid, safe_rating, utc_now()),
        )

    log_event(user_id, "rate", appid=appid, details=f"Rated {game_name}", rating=safe_rating)
    return f"Saved rating {safe_rating}/5 for {game_name}.", render_profile_panel(state), render_history_panel(state)


def review_selected_game(selected_appid: str, review_text: str, user_state: dict):
    state = user_state or guest_state()
    user_id = get_user_id(state)
    if not user_id:
        return "Sign in to submit reviews.", render_profile_panel(state), render_history_panel(state), review_text

    appid, game_name = resolve_selected_game(selected_appid)
    if not appid:
        return "Select a recommended game first.", render_profile_panel(state), render_history_panel(state), review_text

    cleaned_review = (review_text or "").strip()
    if len(cleaned_review) < 5:
        return "Write at least 5 characters for your review.", render_profile_panel(state), render_history_panel(state), review_text

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO reviews (user_id, appid, review_text, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, appid, cleaned_review, utc_now()),
        )

    review_excerpt = cleaned_review[:100] + ("..." if len(cleaned_review) > 100 else "")
    log_event(user_id, "review", appid=appid, details=review_excerpt)
    return f"Review saved for {game_name}.", render_profile_panel(state), render_history_panel(state), ""


def register_and_sync(username: str, password: str, display_name: str, user_state: dict):
    message, new_state = create_account(username, password, display_name, user_state)
    auth_visibility, app_visibility = shell_visibility_updates(new_state)
    return (
        message,
        user_badge_text(new_state),
        new_state,
        render_profile_panel(new_state),
        render_history_panel(new_state),
        feedback_hint_text(new_state),
        auth_visibility,
        app_visibility,
        "",
        "",
    )


def login_and_sync(username: str, password: str, user_state: dict):
    message, new_state = login_account(username, password, user_state)
    auth_visibility, app_visibility = shell_visibility_updates(new_state)
    return (
        message,
        user_badge_text(new_state),
        new_state,
        render_profile_panel(new_state),
        render_history_panel(new_state),
        feedback_hint_text(new_state),
        auth_visibility,
        app_visibility,
        "",
        "",
    )


def logout_and_sync(user_state: dict):
    message, new_state = logout_account(user_state)
    auth_visibility, app_visibility = shell_visibility_updates(new_state)
    return (
        message,
        user_badge_text(new_state),
        new_state,
        render_profile_panel(new_state),
        render_history_panel(new_state),
        feedback_hint_text(new_state),
        auth_visibility,
        app_visibility,
        "",
        "",
    )


# ------------------------------
# UI defaults
# ------------------------------
init_db()
initial_sample_size = min(5, len(df))
initial_indices = df.sample(initial_sample_size, replace=False).index.tolist()
initial_recommendations = recommendation_payload_from_ranked([(idx, 0.0) for idx in initial_indices])
initial_gallery_html = build_gallery_html(initial_recommendations)
initial_dropdown_choices = recommendation_choices(initial_recommendations)

# ------------------------------
# UI Function
# ------------------------------
with gr.Blocks(
    theme=gr.themes.Soft(), 
    css="""
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
        
        * {
            font-family: 'Rajdhani', 'Orbitron', monospace;
        }
        
        footer {visibility: hidden} 
        
        /* Gaming dark background with subtle matrix effect */
        .gradio-container {
            background: 
                radial-gradient(circle at 25% 25%, #0a0a0a 0%, transparent 50%),
                radial-gradient(circle at 75% 75%, #111111 0%, transparent 50%),
                linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
            min-height: 100vh;
            position: relative;
        }
        
        /* Subtle grid overlay for gaming feel */
        .gradio-container::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                linear-gradient(rgba(0, 255, 136, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 255, 136, 0.03) 1px, transparent 1px);
            background-size: 50px 50px;
            pointer-events: none;
            z-index: -1;
            animation: gridPulse 4s ease-in-out infinite;
        }
        
        @keyframes gridPulse {
            0%, 100% { opacity: 0.3; }
            50% { opacity: 0.1; }
        }
        
        /* Main gaming container */
        .main-content {
            background: rgba(13, 17, 23, 0.85);
            border: 2px solid #00ff88;
            border-radius: 12px;
            margin: 20px;
            padding: 40px;
            box-shadow: 
                0 0 30px rgba(0, 255, 136, 0.2),
                inset 0 1px 0 rgba(0, 255, 136, 0.1);
            position: relative;
        }
        
        /* Corner brackets for gaming UI feel */
        .main-content::before,
        .main-content::after {
            content: '';
            position: absolute;
            width: 30px;
            height: 30px;
            border: 3px solid #ff6b35;
        }
        
        .main-content::before {
            top: 15px;
            left: 15px;
            border-right: none;
            border-bottom: none;
        }
        
        .main-content::after {
            bottom: 15px;
            right: 15px;
            border-left: none;
            border-top: none;
        }
        
        /* Gaming title */
        #main_title {
            text-align: center; 
            font-family: 'Orbitron', monospace !important;
            font-size: 4.5rem !important;
            font-weight: 900 !important;
            margin-bottom: 10px !important;
            color: #00ff88 !important;
            position: relative;
        }
        
        #subtitle {
            text-align: center; 
            color: #ff6b35 !important;
            font-size: 1.4rem !important;
            font-weight: 600 !important;
            margin-bottom: 40px !important;
            letter-spacing: 1px;
        }
        
        /* Gaming input styling */
        .gradio-textbox input {
            background: rgba(22, 27, 34, 0.9) !important;
            border: 2px solid #30363d !important;
            border-radius: 8px !important;
            color: #00ff88 !important;
            font-size: 16px !important;
            font-family: 'Rajdhani', monospace !important;
            font-weight: 600 !important;
            padding: 16px 20px !important;
            transition: all 0.3s ease !important;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.5) !important;
        }
        
        .gradio-textbox input:focus {
            background: rgba(22, 27, 34, 1) !important;
            border-color: #00ff88 !important;
            box-shadow: 
                inset 0 2px 4px rgba(0, 0, 0, 0.5),
                0 0 0 3px rgba(0, 255, 136, 0.1),
                0 0 15px rgba(0, 255, 136, 0.3) !important;
        }
        
        .gradio-textbox input::placeholder {
            color: #7d8590 !important;
            font-style: italic;
        }
        
        .gradio-textbox label {
            color: #f0f6fc !important;
            font-weight: 700 !important;
            font-size: 16px !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px !important;
        }
        
        /* Gaming slider styling */
        .gradio-slider {
            background: rgba(22, 27, 34, 0.8) !important;
            border: 2px solid #30363d !important;
            border-radius: 8px !important;
            padding: 20px !important;
        }
        
        .gradio-slider label {
            color: #f0f6fc !important;
            font-weight: 700 !important;
            font-size: 16px !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Gaming button with hexagonal inspiration */
        .gradio-button {
            background: linear-gradient(135deg, #ff6b35 0%, #ff8c42 100%) !important;
            border: 2px solid #ff6b35 !important;
            border-radius: 0 !important;
            color: #000 !important;
            font-size: 18px !important;
            font-weight: 700 !important;
            font-family: 'Orbitron', monospace !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 18px 40px !important;
            position: relative !important;
            overflow: hidden !important;
            clip-path: polygon(10px 0%, 100% 0%, calc(100% - 10px) 100%, 0% 100%) !important;
            box-shadow: 
                0 0 20px rgba(255, 107, 53, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
            transition: all 0.3s ease !important;
        }
        
        .gradio-button:hover {
            background: linear-gradient(135deg, #ff8c42 0%, #ffad5a 100%) !important;
            box-shadow: 
                0 0 30px rgba(255, 107, 53, 0.6),
                0 5px 15px rgba(0, 0, 0, 0.3) !important;
            transform: translateY(-2px) !important;
        }
        
        .gradio-button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            transition: left 0.6s;
        }
        
        .gradio-button:hover::before {
            left: 100%;
        }
        
        /* Gaming label styling */
        .gradio-label {
            background: rgba(22, 27, 34, 0.9) !important;
            border: 2px solid #00ff88 !important;
            border-radius: 8px !important;
            padding: 20px !important;
            margin: 20px 0 !important;
            box-shadow: 
                0 0 20px rgba(0, 255, 136, 0.2),
                inset 0 1px 0 rgba(0, 255, 136, 0.1) !important;
        }
        
        .gradio-label .wrap {
            color: #00ff88 !important;
            font-size: 22px !important;
            font-weight: 700 !important;
            font-family: 'Orbitron', monospace !important;
            text-align: center !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Gaming gallery container */
        .gallery-container {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 25px;
            padding: 30px 0;
            animation: fadeInUp 0.8s ease-out;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Gaming-style game cards */
        .game-card {
            position: relative;
            width: 280px;
            height: 160px;
            background: rgba(22, 27, 34, 0.9);
            border: 2px solid #30363d;
            border-radius: 8px;
            overflow: hidden;
            transition: all 0.3s ease;
            animation: cardSlideIn 0.6s ease-out forwards;
            opacity: 0;
            transform: translateX(-50px);
            box-shadow: 
                0 4px 15px rgba(0, 0, 0, 0.5),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }
        
        @keyframes cardSlideIn {
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        .game-card:nth-child(1) { animation-delay: 0.1s; }
        .game-card:nth-child(2) { animation-delay: 0.2s; }
        .game-card:nth-child(3) { animation-delay: 0.3s; }
        .game-card:nth-child(4) { animation-delay: 0.4s; }
        .game-card:nth-child(5) { animation-delay: 0.5s; }
        
        .game-card:hover {
            border-color: #ff6b35;
            transform: translateY(-5px) scale(1.03);
            box-shadow: 
                0 8px 25px rgba(0, 0, 0, 0.7),
                0 0 20px rgba(255, 107, 53, 0.3),
                inset 0 1px 0 rgba(255, 107, 53, 0.2);
        }

        /* Cyberpunk corner brackets for cards */
        .game-card::before,
        .game-card::after {
            content: '';
            position: absolute;
            width: 15px;
            height: 15px;
            border: 2px solid #00ff88;
            opacity: 0;
            transition: opacity 0.3s ease;
            z-index: 3;
        }
        
        .game-card::before {
            top: 8px;
            left: 8px;
            border-right: none;
            border-bottom: none;
        }
        
        .game-card::after {
            bottom: 8px;
            right: 8px;
            border-left: none;
            border-top: none;
        }
        
        .game-card:hover::before,
        .game-card:hover::after {
            opacity: 1;
        }

        .game-image {
            width: 100%;
            height: 100%;
            object-fit: cover;
            filter: brightness(0.8) contrast(1.1);
            transition: all 0.3s ease;
        }
        
        .game-card:hover .game-image {
            filter: brightness(1) contrast(1.2);
            transform: scale(1.05);
        }

        .game-title-overlay {
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            background: linear-gradient(to top, rgba(13, 17, 23, 0.95), rgba(13, 17, 23, 0.7), transparent);
            color: #f0f6fc;
            padding: 15px 12px 12px;
            text-align: center;
            font-weight: 700;
            font-size: 15px;
            font-family: 'Rajdhani', monospace;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.8);
            transition: all 0.3s ease;
            border-top: 1px solid rgba(0, 255, 136, 0.2);
        }
        
        .game-card:hover .game-title-overlay {
            background: linear-gradient(to top, rgba(13, 17, 23, 0.95), rgba(255, 107, 53, 0.1), transparent);
            color: #00ff88;
            border-top-color: #ff6b35;
        }
        
        /* Gaming row styling */
        .gradio-row {
            background: rgba(22, 27, 34, 0.6) !important;
            border: 1px solid #30363d !important;
            border-radius: 8px !important;
            padding: 20px !important;
            margin: 15px 0 !important;
        }
        
        /* Responsive gaming design */
        @media (max-width: 768px) {
            #main_title {
                font-size: 3rem !important;
            }
            
            #subtitle {
                font-size: 1.2rem !important;
            }
            
            .game-card {
                width: 240px;
                height: 140px;
            }
            
            .main-content {
                margin: 10px;
                padding: 20px;
            }
        }
        
        /* Gaming loading animation */
        @keyframes scanline {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
    """
) as demo:
    user_state = gr.State(guest_state())
    latest_recommendations_state = gr.State(initial_recommendations)

    with gr.Column(elem_classes="main-content", visible=True) as auth_shell:
        gr.Markdown(
            """
            <h1 id="main_title">Respawned</h1>
            <h3 id="subtitle">Sign in or create an account to continue</h3>
            """,
            elem_id="auth_header_section",
        )
        auth_status = gr.Markdown("Welcome. Sign in or create an account to access the recommender.")

        with gr.Tabs():
            with gr.Tab("Sign In"):
                login_username = gr.Textbox(label="Username", placeholder="Enter your username")
                login_password = gr.Textbox(label="Password", type="password", placeholder="Enter password")
                login_btn = gr.Button("Sign In", variant="primary")

            with gr.Tab("Sign Up"):
                signup_username = gr.Textbox(label="Username", placeholder="Choose a username")
                signup_display_name = gr.Textbox(label="Display Name", placeholder="Shown on your profile")
                signup_password = gr.Textbox(label="Password", type="password", placeholder="Minimum 4 characters")
                register_btn = gr.Button("Create Account", variant="primary")

    with gr.Column(elem_classes="main-content", visible=False) as app_shell:
        gr.Markdown(
            """
            <h1 id="main_title">Respawned</h1>
            <h3 id="subtitle">Find. Play. Repeat. | The ultimate stop for game discovery</h3>
            """,
            elem_id="header_section",
        )

        with gr.Row():
            with gr.Column(scale=5):
                user_badge = gr.Markdown(user_badge_text(guest_state()))
            with gr.Column(scale=1, min_width=140):
                logout_btn = gr.Button("Sign Out")

        with gr.Tabs():
            with gr.Tab("Discover"):
                with gr.Row():
                    title_in = gr.Textbox(label="Enter a Game Title", placeholder="e.g. Dark Souls III", scale=3)
                    k_in = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Number of recommendations", scale=1)

                go_btn = gr.Button("Get Recommendations", variant="primary")
                output_title = gr.Markdown("### Your Next Adventure Awaits")
                gallery_html = gr.HTML(initial_gallery_html)

                selected_game = gr.Dropdown(
                    label="Select a recommended game",
                    choices=initial_dropdown_choices,
                    value=initial_dropdown_choices[0][1] if initial_dropdown_choices else None,
                )

                with gr.Row():
                    save_btn = gr.Button("Save Game")
                    rating_in = gr.Slider(minimum=1, maximum=5, value=5, step=1, label="Your rating")
                    rate_btn = gr.Button("Submit Rating")

                review_in = gr.Textbox(
                    label="Write a short review",
                    placeholder="Tell us what worked or what did not...",
                    lines=3,
                )
                review_btn = gr.Button("Submit Review")
                feedback_msg = gr.Markdown(feedback_hint_text(guest_state()))

            with gr.Tab("Profile"):
                refresh_profile_btn = gr.Button("Refresh Profile")
                profile_html = gr.HTML(render_profile_panel(guest_state()))

            with gr.Tab("History"):
                refresh_history_btn = gr.Button("Refresh History")
                history_html = gr.HTML(render_history_panel(guest_state()))

        go_btn.click(
            fn=recommend_game,
            inputs=[title_in, k_in, user_state],
            outputs=[output_title, gallery_html, selected_game, feedback_msg, latest_recommendations_state],
        )

        save_btn.click(
            fn=save_selected_game,
            inputs=[selected_game, user_state],
            outputs=[feedback_msg, profile_html, history_html],
        )

        rate_btn.click(
            fn=rate_selected_game,
            inputs=[selected_game, rating_in, user_state],
            outputs=[feedback_msg, profile_html, history_html],
        )

        review_btn.click(
            fn=review_selected_game,
            inputs=[selected_game, review_in, user_state],
            outputs=[feedback_msg, profile_html, history_html, review_in],
        )

        refresh_profile_btn.click(fn=render_profile_panel, inputs=[user_state], outputs=[profile_html])
        refresh_history_btn.click(fn=render_history_panel, inputs=[user_state], outputs=[history_html])

    login_btn.click(
        fn=login_and_sync,
        inputs=[login_username, login_password, user_state],
        outputs=[auth_status, user_badge, user_state, profile_html, history_html, feedback_msg, auth_shell, app_shell, login_password, signup_password],
    )

    register_btn.click(
        fn=register_and_sync,
        inputs=[signup_username, signup_password, signup_display_name, user_state],
        outputs=[auth_status, user_badge, user_state, profile_html, history_html, feedback_msg, auth_shell, app_shell, login_password, signup_password],
    )

    logout_btn.click(
        fn=logout_and_sync,
        inputs=[user_state],
        outputs=[auth_status, user_badge, user_state, profile_html, history_html, feedback_msg, auth_shell, app_shell, login_password, signup_password],
    )

demo.launch()