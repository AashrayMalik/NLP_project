"""
data_clean.py — Comprehensive cleaning pipeline for r/jobs SQLite DB
=====================================================================
Operations (in order):
  1. Backup the DB
  2. Remove spam posts (templated money-making spam)
  3. Remove AutoModerator / bot posts
  4. Remove megathread placeholder posts
  5. Remove duplicate-body posts (keep highest-scored)
  6. Remove URL-only posts (body is just a link, < 200 chars)
  7. Normalise post text:
       - Strip markdown bold/italic/strikethrough
       - Decode HTML entities (&amp; &lt; &gt; &nbsp;)
       - Remove zero-width spaces (&#x200B; and \u200b)
       - Strip markdown links [text](url) → text
       - Remove bare URLs
       - Strip Reddit sub/user links (/r/... /u/...)
       - Collapse excess whitespace
  8. Remove comments from deleted authors
  9. Remove very short comments (< 10 chars) — one-word replies
  10. Remove bot comments (AutoModerator, *bot* authors)
  11. Remove duplicate-body comments per post (keep highest-scored)
  12. Normalise comment text (same pipeline as posts)
  13. Remove orphan comments (post was deleted in earlier steps)
  14. Remove posts that now have 0 comments
  15. Recalculate num_comments for remaining posts
  16. VACUUM the database
  17. Print before/after summary

Usage:
    python data_clean.py                           # default DB path
    python data_clean.py --db path/to/jobs.db      # custom path
    python data_clean.py --dry-run                  # report only, no changes
"""

import argparse
import html
import os
import re
import shutil
import sqlite3
from datetime import datetime


# config

DEFAULT_DB = "data/jobs_posts.db"
MIN_COMMENT_LEN = 10          # comments shorter than this are removed

# Spam patterns (templated money-making / referral spam)
SPAM_PATTERNS = [
    "%how my life changed thanks to a random post%",
    "%makes good money online%",
    "%I highly recommend reading this Reddit post%",
    "%double it in about a week%",
    "%I make about $300 a day%",
    "%income exceeds $100 per day%",
    "%his post is still relevant%",
    "%pinned right on his profile%",
]

# bot account names
BOT_AUTHORS = {
    "AutoModerator",
    "RemindMeBot",
    "FakeLinkBot",
    "exclaim_bot",
    "Fun-Rebot",
    "Picabot3",
    "xdeebot",
    "promptolovebot",
    "goodcanadianbot97",
}

# text normalisation

_RE_MD_BOLD_ITALIC = re.compile(r"(\*{1,3}|_{1,3})(.+?)\1")   # **bold** / *italic*
_RE_MD_STRIKE       = re.compile(r"~~(.+?)~~")
_RE_MD_LINK         = re.compile(r"\[([^\]]*)\]\([^)]+\)")      # [text](url) → text
_RE_BARE_URL        = re.compile(r"https?://\S+")
_RE_REDDIT_LINK     = re.compile(r"/?[ru]/\w+")                 # /r/sub or /u/user
_RE_ZWSP            = re.compile(r"&#x200[Bb];|\u200b")
_RE_WHITESPACE      = re.compile(r"[ \t]+")
_RE_BLANKLINES      = re.compile(r"\n{3,}")
_RE_EDIT_FOOTER     = re.compile(
    r"\n*\s*(?:edit|eta)\s*[:;]\s*.*$",
    re.IGNORECASE | re.DOTALL,
)
_RE_HEADER          = re.compile(r"^#{1,6}\s+", re.MULTILINE)   # ## Header
_RE_EMOJI           = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F1E0-\U0001F1FF"   # flags
    "\U0001F900-\U0001F9FF"   # supplemental symbols
    "\U0001FA00-\U0001FA6F"   # chess symbols
    "\U0001FA70-\U0001FAFF"   # symbols extended-A
    "\U00002702-\U000027B0"   # dingbats
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U0000200D"              # zero-width joiner
    "\U000024C2-\U0001F251"   # enclosed characters
    "]+",
    flags=re.UNICODE,
)


def normalise_text(text: str) -> str:
    """Clean a Reddit post/comment body for NLP processing."""
    if not text:
        return text

    # HTML entities
    text = html.unescape(text)

    # Zero-width spaces
    text = _RE_ZWSP.sub("", text)

    # Markdown links → keep anchor text only
    text = _RE_MD_LINK.sub(r"\1", text)

    # Bare URLs → remove
    text = _RE_BARE_URL.sub("", text)

    # Reddit sub/user mentions → remove
    text = _RE_REDDIT_LINK.sub("", text)

    # Markdown formatting → keep inner text
    text = _RE_MD_STRIKE.sub(r"\1", text)
    text = _RE_MD_BOLD_ITALIC.sub(r"\2", text)

    # Markdown headers → plain text
    text = _RE_HEADER.sub("", text)

    # Emojis → remove
    text = _RE_EMOJI.sub("", text)

    # Collapse whitespace (preserve newlines for paragraph structure)
    text = _RE_WHITESPACE.sub(" ", text)
    text = _RE_BLANKLINES.sub("\n\n", text)

    return text.strip()


#helpers

def backup_db(path: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{path}.backup_{ts}"
    shutil.copy2(path, backup)
    return backup


def count(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# cleaning steps

def remove_spam_posts(conn: sqlite3.Connection) -> int:
    """Remove posts matching known spam body templates."""
    clauses = " OR ".join(f"body LIKE ?" for _ in SPAM_PATTERNS)
    cur = conn.execute(f"DELETE FROM posts WHERE {clauses}", SPAM_PATTERNS)
    conn.commit()
    return cur.rowcount


def remove_bot_posts(conn: sqlite3.Connection) -> int:
    """Remove posts from known bot accounts."""
    placeholders = ",".join("?" * len(BOT_AUTHORS))
    cur = conn.execute(
        f"DELETE FROM posts WHERE author IN ({placeholders})",
        list(BOT_AUTHORS),
    )
    conn.commit()
    return cur.rowcount


def remove_megathread_posts(conn: sqlite3.Connection) -> int:
    """Remove megathread/sticky placeholder posts."""
    cur = conn.execute("""
        DELETE FROM posts
        WHERE body LIKE '%Megathread%'
           OR title LIKE '%Megathread%'
           OR title LIKE '%megathread%'
    """)
    conn.commit()
    return cur.rowcount


def remove_duplicate_posts(conn: sqlite3.Connection) -> int:
    """Keep only the highest-scored copy of each duplicate body."""
    cur = conn.execute("""
        DELETE FROM posts
        WHERE id NOT IN (
            SELECT id FROM (
                SELECT id, ROW_NUMBER() OVER (
                    PARTITION BY body ORDER BY score DESC
                ) AS rn
                FROM posts
            )
            WHERE rn = 1
        )
    """)
    conn.commit()
    return cur.rowcount


def remove_url_only_posts(conn: sqlite3.Connection) -> int:
    """Remove posts whose body is essentially just a URL (< 200 chars, starts with http)."""
    cur = conn.execute("""
        DELETE FROM posts
        WHERE body LIKE 'http%' AND LENGTH(body) < 200
    """)
    conn.commit()
    return cur.rowcount


def normalise_posts(conn: sqlite3.Connection) -> int:
    """Apply text normalisation to all post bodies and titles."""
    rows = conn.execute("SELECT id, title, body FROM posts").fetchall()
    updates = []
    for pid, title, body in rows:
        new_title = normalise_text(title)
        new_body  = normalise_text(body)
        if new_title != title or new_body != body:
            updates.append((new_title, new_body, pid))
    if updates:
        conn.executemany(
            "UPDATE posts SET title = ?, body = ? WHERE id = ?", updates
        )
        conn.commit()
    return len(updates)


def remove_deleted_author_comments(conn: sqlite3.Connection) -> int:
    """Remove comments from deleted/null authors."""
    cur = conn.execute("""
        DELETE FROM comments WHERE author = '[deleted]' OR author IS NULL
    """)
    conn.commit()
    return cur.rowcount


def remove_short_comments(conn: sqlite3.Connection) -> int:
    """Remove very short comments (< MIN_COMMENT_LEN chars)."""
    cur = conn.execute(
        "DELETE FROM comments WHERE LENGTH(body) < ?", (MIN_COMMENT_LEN,)
    )
    conn.commit()
    return cur.rowcount


def remove_bot_comments(conn: sqlite3.Connection) -> int:
    """Remove comments from known bot accounts."""
    placeholders = ",".join("?" * len(BOT_AUTHORS))
    cur = conn.execute(
        f"DELETE FROM comments WHERE author IN ({placeholders})",
        list(BOT_AUTHORS),
    )
    conn.commit()
    return cur.rowcount


def remove_duplicate_comments(conn: sqlite3.Connection) -> int:
    """For each post, keep only the highest-scored copy of duplicate comment bodies."""
    cur = conn.execute("""
        DELETE FROM comments
        WHERE id NOT IN (
            SELECT id FROM (
                SELECT id, ROW_NUMBER() OVER (
                    PARTITION BY post_id, body ORDER BY score DESC
                ) AS rn
                FROM comments
            )
            WHERE rn = 1
        )
    """)
    conn.commit()
    return cur.rowcount


def normalise_comments(conn: sqlite3.Connection) -> int:
    """Apply text normalisation to all comment bodies."""
    rows = conn.execute("SELECT id, body FROM comments").fetchall()
    updates = []
    for cid, body in rows:
        new_body = normalise_text(body)
        if new_body != body:
            updates.append((new_body, cid))
    if updates:
        conn.executemany(
            "UPDATE comments SET body = ? WHERE id = ?", updates
        )
        conn.commit()
    return len(updates)


def remove_orphan_comments(conn: sqlite3.Connection) -> int:
    """Remove comments whose parent post no longer exists."""
    cur = conn.execute("""
        DELETE FROM comments
        WHERE post_id NOT IN (SELECT id FROM posts)
    """)
    conn.commit()
    return cur.rowcount


def remove_posts_without_comments(conn: sqlite3.Connection) -> int:
    """Remove posts that have zero comments remaining after cleaning."""
    cur = conn.execute("""
        DELETE FROM posts
        WHERE id NOT IN (SELECT DISTINCT post_id FROM comments)
    """)
    conn.commit()
    return cur.rowcount


def recalculate_num_comments(conn: sqlite3.Connection) -> int:
    """Update num_comments to reflect actual comment count in DB."""
    cur = conn.execute("""
        UPDATE posts SET num_comments = (
            SELECT COUNT(*) FROM comments WHERE comments.post_id = posts.id
        )
    """)
    conn.commit()
    return cur.rowcount


#pipeline 

def run_pipeline(db_path: str, dry_run: bool = False):
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    posts_before    = count(conn, "posts")
    comments_before = count(conn, "comments")

    print("═" * 60)
    print(f"  Data Cleaning Pipeline — r/jobs")
    print(f"  DB: {db_path}")
    print(f"  Before: {posts_before:,} posts / {comments_before:,} comments")
    print("═" * 60)

    if dry_run:
        print("\n  DRY-RUN MODE — no changes will be made\n")
        # Show what would be removed
        for label, sql, params in [
            ("Spam posts", f"SELECT COUNT(*) FROM posts WHERE {' OR '.join('body LIKE ?' for _ in SPAM_PATTERNS)}", SPAM_PATTERNS),
            ("Bot posts", f"SELECT COUNT(*) FROM posts WHERE author IN ({','.join('?' * len(BOT_AUTHORS))})", list(BOT_AUTHORS)),
            ("Megathread posts", "SELECT COUNT(*) FROM posts WHERE body LIKE '%Megathread%' OR title LIKE '%Megathread%'", []),
            ("URL-only posts", "SELECT COUNT(*) FROM posts WHERE body LIKE 'http%' AND LENGTH(body) < 200", []),
            ("Bot comments", f"SELECT COUNT(*) FROM comments WHERE author IN ({','.join('?' * len(BOT_AUTHORS))})", list(BOT_AUTHORS)),
            ("Deleted-author comments", "SELECT COUNT(*) FROM comments WHERE author = '[deleted]' OR author IS NULL", []),
            (f"Short comments (<{MIN_COMMENT_LEN} chars)", f"SELECT COUNT(*) FROM comments WHERE LENGTH(body) < {MIN_COMMENT_LEN}", []),
        ]:
            n = conn.execute(sql, params).fetchone()[0]
            print(f"  Would remove: {n:>6,}  {label}")
        conn.close()
        return

    # backup
    backup = backup_db(db_path)
    print(f"\n   Backup → {backup}\n")

    steps = [
        ("Remove spam posts",              remove_spam_posts),
        ("Remove bot posts",               remove_bot_posts),
        ("Remove megathread posts",        remove_megathread_posts),
        ("Remove duplicate posts",         remove_duplicate_posts),
        ("Remove URL-only posts",          remove_url_only_posts),
        ("Normalise post text",            normalise_posts),
        ("Remove deleted-author comments", remove_deleted_author_comments),
        ("Remove short comments",          remove_short_comments),
        ("Remove bot comments",            remove_bot_comments),
        ("Remove duplicate comments",      remove_duplicate_comments),
        ("Normalise comment text",         normalise_comments),
        ("Remove orphan comments",         remove_orphan_comments),
        ("Remove posts w/o comments",      remove_posts_without_comments),
        ("Recalculate num_comments",       recalculate_num_comments),
    ]

    for i, (label, fn) in enumerate(steps, 1):
        affected = fn(conn)
        icon = "sweep" if affected > 0 else "tick"
        print(f"  {i:2d}. {icon} {label:<36s}  affected: {affected:>6,}")

    # vacuum
    conn.execute("VACUUM")
    print(f"\n   VACUUM complete")

    posts_after    = count(conn, "posts")
    comments_after = count(conn, "comments")
    unique_authors = conn.execute(
        "SELECT COUNT(DISTINCT author) FROM posts WHERE author != '[deleted]'"
    ).fetchone()[0]
    unique_c_authors = conn.execute(
        "SELECT COUNT(DISTINCT author) FROM comments WHERE author != '[deleted]'"
    ).fetchone()[0]

    print("\n" + "═" * 60)
    print(f"  RESULTS")
    print(f"  {'':30s} {'Before':>10s}  {'After':>10s}  {'Removed':>10s}")
    print(f"  {'Posts':<30s} {posts_before:>10,}  {posts_after:>10,}  {posts_before - posts_after:>10,}")
    print(f"  {'Comments':<30s} {comments_before:>10,}  {comments_after:>10,}  {comments_before - comments_after:>10,}")
    print(f"  {'Unique post authors':<30s} {'':>10s}  {unique_authors:>10,}")
    print(f"  {'Unique comment authors':<30s} {'':>10s}  {unique_c_authors:>10,}")

    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
    backup_size_mb = os.path.getsize(backup) / (1024 * 1024)
    print(f"\n  DB size: {backup_size_mb:.1f} MB → {db_size_mb:.1f} MB")
    print("═" * 60)

    conn.close()


#cli

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean r/jobs Reddit database")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to SQLite DB")
    parser.add_argument("--dry-run", action="store_true", help="Report only, no changes")
    args = parser.parse_args()
    run_pipeline(args.db, dry_run=args.dry_run)