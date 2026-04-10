"""
r/jobs scraper — targets 25 000 qualifying posts
-------------------------------------------------
Qualifying post criteria:
  • body length > 20 chars
  • not removed / deleted

Diversity split (per month and globally):
  • 60 % HIGH-engagement : num_comments >= 10
  • 40 % LOW-engagement  : 1 <= num_comments <= 9

Strategy:
  • Base bucket  : 1 day
  • Auto-split   : halve any bucket that hits ≥ 900 raw results, floor = 1 hour
  • Monthly quota: 3 125 qualifying posts per calendar month
    (1 875 high + 1 250 low, ≈ 8 months × 3 125 = 25 000)
  • If a month comes up short after exhaustive scraping, extend the
    search window backward by one extra month (up to MAX_EXTRA_MONTHS).
  • Comments: top 5 by score fetched from Arctic Shift comments endpoint
    after each bucket, stored in a separate table.
  • Sort: "desc" (newest-first) within each time window.
  • Persistence: SQLite WAL, INSERT OR REPLACE for safe re-runs.
"""

import sqlite3
import time
from calendar import monthrange
from datetime import datetime, timezone
from typing import Iterator
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import requests

#config 

SUBREDDIT       = "jobs"
GLOBAL_TARGET   = 25_000
HIGH_RATIO      = 0.60          # >= 10 comments
LOW_RATIO       = 0.40          # 1–9 comments
POSTS_PER_MONTH = 3_125         # ceil(25000 / 8)
HIGH_PER_MONTH  = int(POSTS_PER_MONTH * HIGH_RATIO)   # 1 875
LOW_PER_MONTH   = POSTS_PER_MONTH - HIGH_PER_MONTH    # 1 250
MAX_EXTRA_MONTHS = 12
DB_PATH         = "data/jobs_posts.db"

HEADERS = {
    "User-Agent": (
        "reddit-jobs-scraper/2.0 "
        "(research project; contact: aashraymalik@gmail.com)"
    )
}

MIN_BODY_LEN    = 20
TOP_N_COMMENTS  = 5

BUCKET_DAYS     = 1
MIN_BUCKET_S    = 3_600
PAGE_DELAY_S    = 1.0
BUCKET_DELAY_S  = 1.5
COMMENT_DELAY_S = 0.5

BASE_API = "https://arctic-shift.photon-reddit.com/api"


# db

def init_db(path: str = DB_PATH) -> sqlite3.Connection:
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id            TEXT PRIMARY KEY,
            title         TEXT,
            body          TEXT,
            score         INTEGER,
            created_utc   INTEGER,
            author        TEXT,
            num_comments  INTEGER,
            flair         TEXT,
            month         TEXT,
            tier          TEXT,        -- 'high' | 'low'
            scraped_at    TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            id            TEXT PRIMARY KEY,
            post_id       TEXT NOT NULL REFERENCES posts(id),
            body          TEXT,
            score         INTEGER,
            author        TEXT,
            created_utc   INTEGER,
            scraped_at    TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_post_score   ON posts(score DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_post_flair   ON posts(flair)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_post_month   ON posts(month)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_post_tier    ON posts(tier)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_comment_post ON comments(post_id)")
    conn.commit()
    return conn


def save_posts(conn: sqlite3.Connection, posts: list[dict]) -> None:
    rows = [
        (
            p["id"], p["title"], p["body"],
            p["score"], int(p["created_utc"]),
            p["author"], p["num_comments"],
            p.get("flair"),
            datetime.utcfromtimestamp(p["created_utc"]).strftime("%Y-%m"),
            p["tier"],
        )
        for p in posts
    ]
    conn.executemany("""
        INSERT OR REPLACE INTO posts
            (id, title, body, score, created_utc, author,
             num_comments, flair, month, tier)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()


def save_comments(conn: sqlite3.Connection, comments: list[dict]) -> None:
    if not comments:
        return
    rows = [
        (c["id"], c["post_id"], c["body"],
         c["score"], c["author"], int(c["created_utc"]))
        for c in comments
    ]
    conn.executemany("""
        INSERT OR REPLACE INTO comments
            (id, post_id, body, score, author, created_utc)
        VALUES (?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()


def count_tier(conn: sqlite3.Connection, ym: str, tier: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM posts WHERE month = ? AND tier = ?", (ym, tier)
    ).fetchone()[0]


def total_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]


def existing_ids(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT id FROM posts").fetchall()}


# arctic shift posts

def _fetch_posts_page(after_ts: int, before_ts: int) -> list[dict]:
    params = {
        "subreddit": SUBREDDIT,
        "after":     after_ts,
        "before":    before_ts,
        "limit":     100,
        "sort":      "desc",
    }
    for attempt in range(4):
        try:
            resp = requests.get(
                f"{BASE_API}/posts/search",
                params=params,
                headers=HEADERS,
                timeout=30,
                verify=False,
            )
            if resp.status_code == 200:
                return resp.json().get("data", [])
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"    429 — sleeping {wait}s …")
                time.sleep(wait)
            else:
                print(f"    HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(10)
        except requests.RequestException as e:
            print(f"    Request error: {e}")
            time.sleep(10)
    return []


def _iter_bucket_pages(after_ts: int, before_ts: int) -> Iterator[dict]:
    current_before = before_ts
    seen_ids: set[str] = set()
    for _ in range(10):
        posts = _fetch_posts_page(after_ts, current_before)
        if not posts:
            return
        for p in posts:
            pid = p.get("id")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                yield p
        if len(posts) < 100:
            return
        oldest_ts = min(int(p["created_utc"]) for p in posts)
        if oldest_ts <= after_ts:
            return
        current_before = oldest_ts - 1
        time.sleep(PAGE_DELAY_S)


def scrape_bucket_raw(after_ts: int, before_ts: int) -> tuple[list[dict], list[dict]]:
    """
    Collect posts in the bucket, split into (high, low) tiers:
      high: body valid, len > MIN_BODY_LEN, num_comments >= 10
      low:  body valid, len > MIN_BODY_LEN, 1 <= num_comments <= 9

    Returns (high_posts, low_posts) — both unseen/unfiltered by quota.
    """
    high, low = [], []
    for p in _iter_bucket_pages(after_ts, before_ts):
        body = p.get("selftext", "")
        if body in ("[removed]", "[deleted]", "", None):
            continue
        if len(body) < MIN_BODY_LEN:
            continue
        nc = p.get("num_comments", 0)
        if nc < 1:
            continue
        record = {
            "id":           p["id"],
            "title":        p.get("title", ""),
            "body":         body,
            "score":        p.get("score", 0),
            "created_utc":  p["created_utc"],
            "author":       p.get("author", "[deleted]"),
            "num_comments": nc,
            "flair":        p.get("link_flair_text"),
        }
        if nc >= 10:
            record["tier"] = "high"
            high.append(record)
        else:
            record["tier"] = "low"
            low.append(record)
    return high, low


# arctic shift comments

def fetch_top_comments(post_id: str, n: int = TOP_N_COMMENTS) -> list[dict]:
    params = {
        "link_id": f"t3_{post_id}",
        "limit":   100,
        "sort":    "desc",
    }
    for attempt in range(3):
        try:
            resp = requests.get(
                f"{BASE_API}/comments/search",
                params=params,
                headers=HEADERS,
                timeout=30,
                verify=False,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                valid = [
                    c for c in data
                    if c.get("body") not in ("[removed]", "[deleted]", "", None)
                ]
                valid.sort(key=lambda c: c.get("score", 0), reverse=True)
                return [
                    {
                        "id":          c["id"],
                        "post_id":     post_id,
                        "body":        c.get("body", ""),
                        "score":       c.get("score", 0),
                        "author":      c.get("author", "[deleted]"),
                        "created_utc": c.get("created_utc", 0),
                    }
                    for c in valid[:n]
                ]
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"    [comments] 429 — sleeping {wait}s …")
                time.sleep(wait)
            else:
                print(f"    [comments] HTTP {resp.status_code} for {post_id}")
                time.sleep(5)
        except requests.RequestException as e:
            print(f"    [comments] Request error for {post_id}: {e}")
            time.sleep(5)
    return []


def fetch_comments_for_posts(
    conn: sqlite3.Connection,
    post_ids: list[str],
    verbose: bool = False,
) -> int:
    already_have = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT post_id FROM comments"
        ).fetchall()
    }
    to_fetch = [pid for pid in post_ids if pid not in already_have]
    total = 0
    for i, pid in enumerate(to_fetch):
        comments = fetch_top_comments(pid)
        if comments:
            save_comments(conn, comments)
            total += len(comments)
        if verbose and (i + 1) % 100 == 0:
            print(f"    comments: {i+1}/{len(to_fetch)} posts processed, "
                  f"{total} comments saved so far")
        time.sleep(COMMENT_DELAY_S)
    return total


#bucket splitter 

def collect_bucket(
    after_ts: int,
    before_ts: int,
    seen: set[str],
    conn: sqlite3.Connection,
    high_quota: int,
    low_quota: int,
    depth: int = 0,
) -> list[str]:
    """
    Scrape [after_ts, before_ts], respecting per-tier quotas.
    Splits if raw count (high + low) >= 900 and span > MIN_BUCKET_S.
    Returns list of new post IDs saved.
    """
    if high_quota <= 0 and low_quota <= 0:
        return []

    indent = "  " * depth
    span_s = before_ts - after_ts
    from_s = datetime.utcfromtimestamp(after_ts).strftime("%m-%d %H:%M")
    to_s   = datetime.utcfromtimestamp(before_ts).strftime("%m-%d %H:%M")

    high_raw, low_raw = scrape_bucket_raw(after_ts, before_ts)
    total_raw = len(high_raw) + len(low_raw)
    print(f"{indent}[{from_s} → {to_s}]  raw_high={len(high_raw)} raw_low={len(low_raw)}", end="")

    if total_raw >= 900 and span_s > MIN_BUCKET_S:
        print("  → splitting")
        mid = (after_ts + before_ts) // 2

        # First half
        ids1 = collect_bucket(after_ts, mid, seen, conn, high_quota, low_quota, depth + 1)
        # Recalculate remaining quotas after first half
        saved_high1 = sum(1 for pid in ids1 if _tier_of(conn, pid) == "high")
        saved_low1  = sum(1 for pid in ids1 if _tier_of(conn, pid) == "low")
        ids2 = collect_bucket(
            mid, before_ts, seen, conn,
            high_quota - saved_high1,
            low_quota  - saved_low1,
            depth + 1,
        )
        return ids1 + ids2

    # Filter to unseen
    new_high = [p for p in high_raw if p["id"] not in seen][:max(high_quota, 0)]
    new_low  = [p for p in low_raw  if p["id"] not in seen][:max(low_quota,  0)]

    new_posts = new_high + new_low
    for p in new_posts:
        seen.add(p["id"])

    if new_posts:
        save_posts(conn, new_posts)

    new_ids = [p["id"] for p in new_posts]
    print(f"  saved_high={len(new_high)} saved_low={len(new_low)}  db_total={total_count(conn)}")
    return new_ids


def _tier_of(conn: sqlite3.Connection, post_id: str) -> str:
    """Quick lookup of a post's tier from the DB."""
    row = conn.execute("SELECT tier FROM posts WHERE id = ?", (post_id,)).fetchone()
    return row[0] if row else "high"


#month windows

def month_window(year: int, month: int) -> tuple[str, int, int]:
    last_day = monthrange(year, month)[1]
    start = datetime(year, month, 1,        tzinfo=timezone.utc)
    end   = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)
    ym    = f"{year:04d}-{month:02d}"
    return ym, int(start.timestamp()), int(end.timestamp())


def prev_month(year: int, month: int) -> tuple[int, int]:
    month -= 1
    if month == 0:
        month = 12
        year -= 1
    return year, month


def build_month_list(num_months: int) -> list[tuple[str, int, int]]:
    now = datetime.now(timezone.utc)
    year, month = now.year, now.month
    items = []
    for _ in range(num_months):
        year, month = prev_month(year, month)
        items.append(month_window(year, month))
    items.reverse()
    return items


#per month scrape loop

def scrape_month(
    ym: str,
    month_start: int,
    month_end: int,
    seen: set[str],
    conn: sqlite3.Connection,
) -> list[str]:
    already_high = count_tier(conn, ym, "high")
    already_low  = count_tier(conn, ym, "low")
    rem_high = HIGH_PER_MONTH - already_high
    rem_low  = LOW_PER_MONTH  - already_low

    if rem_high <= 0 and rem_low <= 0:
        print(f"── {ym}  quota full ({already_high}H + {already_low}L) — skipping\n")
        return []

    print(f"── {ym}  need {rem_high} high + {rem_low} low qualifying posts ──────")

    all_new_ids: list[str] = []
    cursor   = month_start
    bucket_s = BUCKET_DAYS * 86_400

    while cursor < month_end:
        bucket_end = min(cursor + bucket_s, month_end)
        cur_high   = HIGH_PER_MONTH - count_tier(conn, ym, "high")
        cur_low    = LOW_PER_MONTH  - count_tier(conn, ym, "low")
        if cur_high <= 0 and cur_low <= 0:
            print(f"  Month quota reached for {ym}")
            break

        new_ids = collect_bucket(cursor, bucket_end, seen, conn, cur_high, cur_low)
        all_new_ids.extend(new_ids)
        cursor = bucket_end
        time.sleep(BUCKET_DELAY_S)

    h = count_tier(conn, ym, "high")
    l = count_tier(conn, ym, "low")
    print(f"  → {ym} done: {h} high + {l} low = {h+l} posts\n")
    return all_new_ids


# entry point

def run():
    conn = init_db(DB_PATH)
    seen = existing_ids(conn)
    print(f"Resuming — {len(seen)} qualifying posts already in DB\n")
    print(
        f"Global target: {GLOBAL_TARGET} posts  "
        f"({int(HIGH_RATIO*100)}% high ≥10 comments, "
        f"{int(LOW_RATIO*100)}% low 1–9 comments)\n"
        f"Per month: {HIGH_PER_MONTH} high + {LOW_PER_MONTH} low = {POSTS_PER_MONTH}\n"
    )

    #posts 
    initial_months = 8
    month_list = build_month_list(initial_months)
    all_new_post_ids: list[str] = []

    for ym, ms, me in month_list:
        new_ids = scrape_month(ym, ms, me, seen, conn)
        all_new_post_ids.extend(new_ids)

    # Extend backward if short
    if total_count(conn) < GLOBAL_TARGET:
        print(f"\n⚠ Only {total_count(conn)} posts — extending backward …\n")
        now = datetime.now(timezone.utc)
        year, month = now.year, now.month
        for _ in range(initial_months):
            year, month = prev_month(year, month)

        for _ in range(MAX_EXTRA_MONTHS):
            if total_count(conn) >= GLOBAL_TARGET:
                break
            ym, ms, me = month_window(year, month)
            new_ids = scrape_month(ym, ms, me, seen, conn)
            all_new_post_ids.extend(new_ids)
            year, month = prev_month(year, month)

    print(f"\n✓ Post scraping complete: {total_count(conn)} qualifying posts\n")

    #comments
    print("── Fetching comments ─────────────────────────────────────────")
    all_post_ids = [r[0] for r in conn.execute("SELECT id FROM posts").fetchall()]
    n_comments = fetch_comments_for_posts(conn, all_post_ids, verbose=True)
    print(f"✓ Comments done: {n_comments} new comments saved\n")

    print_summary(conn)
    conn.close()


# report

def print_summary(conn: sqlite3.Connection) -> None:
    total_posts    = total_count(conn)
    total_high     = conn.execute("SELECT COUNT(*) FROM posts WHERE tier='high'").fetchone()[0]
    total_low      = conn.execute("SELECT COUNT(*) FROM posts WHERE tier='low'").fetchone()[0]
    total_comments = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
    unique_authors = conn.execute(
        "SELECT COUNT(DISTINCT author) FROM posts WHERE author != '[deleted]'"
    ).fetchone()[0]

    print("═" * 52)
    print(f"  Posts    : {total_posts:>8,}  (high={total_high:,} / low={total_low:,})")
    actual_ratio = total_high / total_posts * 100 if total_posts else 0
    print(f"  Ratio    : {actual_ratio:.1f}% high / {100-actual_ratio:.1f}% low")
    print(f"  Comments : {total_comments:>8,}")
    print(f"  Authors  : {unique_authors:>8,}")

    print("\n── Monthly breakdown (H=high, L=low) ─────────────")
    rows = conn.execute("""
        SELECT month,
               SUM(CASE WHEN tier='high' THEN 1 ELSE 0 END) AS h,
               SUM(CASE WHEN tier='low'  THEN 1 ELSE 0 END) AS l
        FROM posts GROUP BY month ORDER BY month
    """).fetchall()
    for ym, h, l in rows:
        bar = "█" * ((h + l) // 50)
        print(f"  {ym}  H={h:4d} L={l:4d}  {bar}")

    print("\n── Top flairs ────────────────────────────────────")
    flairs = conn.execute("""
        SELECT flair, COUNT(*) AS cnt FROM posts
        WHERE flair IS NOT NULL
        GROUP BY flair ORDER BY cnt DESC LIMIT 15
    """).fetchall()
    for flair, cnt in flairs:
        print(f"  {cnt:5d}  {flair}")

    print("\n── Top 10 posts overall ──────────────────────────")
    rows = conn.execute("""
        SELECT id, title, score, num_comments, tier
        FROM posts ORDER BY score DESC LIMIT 10
    """).fetchall()
    for pid, title, score, nc, tier in rows:
        print(f"  [{score:>5}][{tier}] {title[:55]} ({nc} comments)")
    print("═" * 52)


if __name__ == "__main__":
    conn = init_db(DB_PATH)
    print("── Fetching missing comments ─────────────────────────────────")
    all_post_ids = [r[0] for r in conn.execute("SELECT id FROM posts").fetchall()]
    n = fetch_comments_for_posts(conn, all_post_ids, verbose=True)
    print(f"✓ Done: {n} new comments saved")
    conn.close()