#!/usr/bin/env python3
"""
Sakugabooru Anime Sakuga Scraper
================================
Downloads curated animation clips (sakuga) from sakugabooru.com
for animation research (stretch/squish, fluid motion, effects, etc.)

Uses the Moebooru JSON API. Respects rate limits with delays between requests.
Deduplicates across tag combos by post ID.

Usage:
    python scrape_sakugabooru.py                       # Full default scrape (~100 hours target)
    python scrape_sakugabooru.py --estimate             # Estimate size/duration without downloading
    python scrape_sakugabooru.py --anime jujutsu_kaisen_series --limit 500
    python scrape_sakugabooru.py --techniques smears liquid morphing --limit 2000
    python scrape_sakugabooru.py --list-tags            # Show available tags
    python scrape_sakugabooru.py --min-score 20         # Only well-rated clips
    python scrape_sakugabooru.py --resume               # Resume interrupted download
"""

import argparse
import json
import os
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE_URL = "https://www.sakugabooru.com"
API_URL = f"{BASE_URL}/post.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
}

# -- Cloudflare cookie session ------------------------------------------------
COOKIE_FILE = Path(__file__).resolve().parents[2] / "data" / "sakugabooru" / "cf_cookies.json"

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

def load_cf_cookies():
    """Load Cloudflare cookies from cf_cookies.json into the session."""
    if COOKIE_FILE.exists():
        cookies = json.loads(COOKIE_FILE.read_text())
        for name, value in cookies.items():
            SESSION.cookies.set(name, value, domain=".sakugabooru.com")
        print(f"Loaded {len(cookies)} Cloudflare cookies from {COOKIE_FILE.name}")
    else:
        print(f"WARNING: {COOKIE_FILE.name} not found - requests may be blocked by Cloudflare.")
        print(f"  To fix: visit https://www.sakugabooru.com in your browser, solve the challenge,")
        print(f"  then run: python extract_cf_cookies.py")

load_cf_cookies()

# -- Anime series tags (use _series where available for max coverage) ----------
ANIME_TAGS = {
    # Tier 1: Known for exceptional stretch/squish & fluid animation
    "my_hero_academia":              "My Hero Academia (1928 posts)",
    "jujutsu_kaisen_series":         "Jujutsu Kaisen - all seasons (1932 posts)",
    "one_piece":                     "One Piece (3569 posts)",
    "naruto_shippuuden":             "Naruto Shippuden (1713 posts)",
    "naruto":                        "Naruto original (584 posts)",
    "mob_psycho_100_series":         "Mob Psycho 100 - all seasons (535 posts)",
    "dragon_ball_series":            "Dragon Ball - all series (1408 posts)",
    "chainsaw_man_series":           "Chainsaw Man - all (502 posts)",
    "kimetsu_no_yaiba_series":       "Demon Slayer - all (660 posts)",
    "bleach_series":                 "Bleach - all (1048 posts)",

    # Tier 2: Great animation, diverse styles
    "fate_series":                   "Fate - all (2323 posts)",
    "shingeki_no_kyojin_series":     "Attack on Titan - all (952 posts)",
    "black_clover":                  "Black Clover (776 posts)",
    "fire_force_series":             "Fire Force - all (590 posts)",
    "solo_leveling":                 "Solo Leveling (410 posts)",
    "dandadan":                      "Dandadan (408 posts)",
    "undead_unluck":                 "Undead Unluck (295 posts)",
    "boruto:_naruto_next_generations": "Boruto (584 posts)",
    "kaiju_no._8":                   "Kaiju No. 8 (244 posts)",
    "haikyuu!!_series":              "Haikyuu!! - all (438 posts)",

    # Tier 3: Stylistically unique / sakuga-heavy
    "fullmetal_alchemist_brotherhood": "FMA Brotherhood (344 posts)",
    "hunter_x_hunter_2011":          "Hunter x Hunter 2011 (211 posts)",
    "jojo's_bizarre_adventure_series": "JoJo - all (571 posts)",
    "mushoku_tensei_series":         "Mushoku Tensei - all (333 posts)",
    "space_dandy":                   "Space Dandy (462 posts)",
    "flcl_series":                   "FLCL - all (390 posts)",
    "cowboy_bebop":                  "Cowboy Bebop (279 posts)",
    "kill_la_kill":                  "Kill la Kill (252 posts)",
    "ping_pong":                     "Ping Pong the Animation (127 posts)",
    "redline":                       "Redline (164 posts)",
    "akira":                         "Akira (160 posts)",
    "devilman_crybaby":              "Devilman Crybaby (87 posts)",
    "promare":                       "Promare (67 posts)",
    "vinland_saga_series":           "Vinland Saga - all (107 posts)",
    "spy_x_family_series":           "Spy x Family - all (379 posts)",
    "precure":                       "Precure (3068 posts)",
    "pokemon":                       "Pokemon (4068 posts)",
}

# -- Technique tags for stretch/squish & fluid motion -------------------------
TECHNIQUE_TAGS = {
    # Core stretch/squish
    "smears":            "Smear frames - classic squash & stretch (56k posts)",
    "morphing":          "Shape morphing / transformation (4.4k posts)",
    "impact_frames":     "Impact/hit deformation frames (13k posts)",
    "character_acting":  "Expressive character acting (81k posts)",

    # Fluid motion
    "liquid":            "Liquid/fluid animation (28k posts)",
    "smoke":             "Smoke animation (58k posts)",
    "fire":              "Fire effects (16k posts)",
    "explosions":        "Explosion effects (23k posts)",
    "wind":              "Wind effects (12k posts)",

    # Dynamic action
    "fighting":          "Fight choreography (52k posts)",
    "running":           "Run cycles (20k posts)",
    "effects":           "General effects animation (121k posts)",
    "debris":            "Debris / destruction (26k posts)",
    "sparks":            "Spark effects (14k posts)",
    "lightning":         "Lightning effects (13k posts)",

    # Secondary motion
    "hair":              "Hair animation / secondary motion (23k posts)",
    "fabric":            "Fabric / cloth simulation (21k posts)",
    "creatures":         "Creature animation (35k posts)",
    "rotation":          "Rotation / 3D turns (5.8k posts)",
    "background_animation": "Background animation (17k posts)",
}

# -- Default scrape plan: targets ~100 hours ----------------------------------
# Strategy: broad technique sweeps (no anime filter) for volume + diversity,
# plus targeted anime combos for specific shows.

DEFAULT_SCRAPE_PLAN = {
    # Phase 1: Technique-only sweeps (most diverse, highest volume)
    # These pull from ALL anime, ordered by score (best first).
    # Many clips are multi-tagged so later tags yield fewer unique clips.
    # Over-request to compensate for dedup losses (~40-60% overlap between tags).
    "technique_sweeps": [
        # (technique_tag, limit) - no anime filter
        # --- Core stretch/squish ---
        ("smears",           5000),   # THE stretch/squish tag (56k available)
        ("morphing",         3000),   # Shape deformation / transformation
        ("impact_frames",    3000),   # Hit deformation frames

        # --- Fluid motion ---
        ("liquid",           3000),   # Fluid/water animation (28k available)
        ("smoke",            2000),   # Smoke dynamics (58k available)
        ("fire",             2000),   # Fire effects (16k available)
        ("explosions",       2000),   # Explosive fluid VFX
        ("wind",             1500),   # Wind effects

        # --- Dynamic action (heavy in stretch/squish) ---
        ("fighting",         3000),   # Fight choreography (52k available)
        ("effects",          3000),   # General effects animation (121k available!)
        ("running",          2000),   # Run cycles / locomotion
        ("creatures",        2000),   # Creature animation (35k available)

        # --- Secondary motion / exaggeration ---
        ("character_acting", 2000),   # Expressive acting / exaggeration
        ("debris",           1500),   # Dynamic destruction
        ("rotation",         1000),   # 3D form turns
        ("hair",             1000),   # Hair secondary motion
        ("fabric",           1000),   # Cloth simulation
        ("sparks",           1000),   # Spark effects
        ("lightning",        1000),   # Lightning effects
        ("beams",             800),   # Energy beam effects
        ("flying",            800),   # Flying / aerial motion
        ("dancing",           800),   # Dance choreography (great for body mechanics)
        ("sports",            800),   # Sports animation (dynamic body motion)
        ("background_animation", 500), # Camera/bg motion
    ],

    # Phase 2: Targeted anime scrapes (fill gaps, ensure key shows are covered)
    # Grabs ALL posts from these anime regardless of technique tag.
    # Only fetches posts NOT already grabbed in phase 1.
    "anime_sweeps": [
        # Tier 1: Known for exceptional sakuga / stretch-squish
        ("mob_psycho_100_series",         535),   # Legendary fluid animation
        ("flcl_series",                   390),   # GAINAX/Trigger insanity
        ("space_dandy",                   462),   # Bones - wild styles
        ("redline",                       164),   # Peak animation film
        ("akira",                         160),   # Foundational sakuga
        ("ping_pong",                     127),   # Yuasa - rubber body style
        ("devilman_crybaby",               87),   # Yuasa again
        ("promare",                        67),   # Trigger peak
        ("kill_la_kill",                  252),   # Trigger stretch-squish
        ("precure",                      2000),   # Underrated - incredible action sakuga (3k avail)
        ("pokemon",                      2000),   # Surprisingly sakuga-rich (4k avail)
        ("dandadan",                      408),   # Modern sakuga showcase
        ("undead_unluck",                 295),   # Very stretchy style

        # Tier 2: Major shonen with lots of dynamic animation
        ("my_hero_academia",             1928),   # All of it
        ("jujutsu_kaisen_series",        1932),   # All of it
        ("one_piece",                    3569),   # All of it
        ("naruto_shippuuden",            1713),   # All of it
        ("naruto",                        584),   # All of it
        ("dragon_ball_series",           1408),   # All of it
        ("bleach_series",                1048),   # All of it
        ("chainsaw_man_series",           502),   # All of it
        ("kimetsu_no_yaiba_series",       660),   # All of it

        # Tier 3: Great action / unique styles
        ("fate_series",                  2323),   # Ufotable sakuga
        ("shingeki_no_kyojin_series",     952),   # AoT - dynamic action
        ("black_clover",                  776),   # Underrated action sakuga
        ("fire_force_series",             590),   # David Pro fire animation
        ("solo_leveling",                 410),   # Modern action
        ("boruto:_naruto_next_generations", 584), # Some incredible episodes
        ("kaiju_no._8",                   244),   # Creature action
        ("haikyuu!!_series",              438),   # Sports body mechanics
        ("fullmetal_alchemist_brotherhood", 344),
        ("hunter_x_hunter_2011",          211),
        ("jojo's_bizarre_adventure_series", 571),
        ("mushoku_tensei_series",         333),   # Detailed effects
        ("cowboy_bebop",                  279),   # Classic fluid action
        ("vinland_saga_series",           107),
        ("spy_x_family_series",           379),
        ("digimon",                       791),   # Good creature animation
        ("sonic_the_hedgehog",            555),   # Fast motion / stretch-squish
    ],
}

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "sakugabooru" / "clips"
STATE_FILE = OUTPUT_DIR / "_state.json"
REQUEST_DELAY = 0.8   # seconds between API page fetches
DOWNLOAD_DELAY = 0.3  # seconds between file downloads
MAX_RETRIES = 3


def load_state() -> dict:
    """Load download state for resume support."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"downloaded_ids": [], "failed_ids": [], "stats": {}}


def save_state(state: dict):
    """Persist download state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def fetch_posts(tags: list[str], page: int = 1, limit: int = 100,
                min_score: int = 0) -> list[dict]:
    """Fetch posts from Sakugabooru API."""
    tag_str = "+".join(tags) + "+order:score"
    if min_score > 0:
        tag_str += f"+score:>={min_score}"

    per_page = min(limit, 100)
    url = f"{API_URL}?tags={tag_str}&page={page}&limit={per_page}"

    for attempt in range(MAX_RETRIES):
        try:
            resp = SESSION.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                print(f"    Retry {attempt+1}/{MAX_RETRIES} in {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"    Failed after {MAX_RETRIES} attempts: {e}")
                return []


def fetch_all_posts(tags: list[str], max_posts: int, min_score: int = 0,
                    seen_ids: set = None) -> list[dict]:
    """Fetch multiple pages, deduplicating against seen_ids."""
    if seen_ids is None:
        seen_ids = set()

    all_posts = []
    page = 1
    empty_pages = 0

    while len(all_posts) < max_posts:
        posts = fetch_posts(tags, page=page, limit=100, min_score=min_score)
        if not posts:
            break

        new_posts = [p for p in posts if p["id"] not in seen_ids
                     and p.get("file_ext") in ("mp4", "webm")]

        if not new_posts:
            empty_pages += 1
            if empty_pages >= 3:  # 3 consecutive pages with no new videos
                break
        else:
            empty_pages = 0

        all_posts.extend(new_posts)
        for p in new_posts:
            seen_ids.add(p["id"])

        sys.stdout.write(f"\r  Fetched page {page}: {len(all_posts)} unique videos so far...")
        sys.stdout.flush()
        page += 1
        time.sleep(REQUEST_DELAY)

    print(f"\r  Fetched {page-1} pages: {len(all_posts)} unique videos                ")
    return all_posts[:max_posts]


def download_file(url: str, dest: Path) -> bool:
    """Download a file with progress indication."""
    if dest.exists():
        return False

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        resp = SESSION.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    mb = downloaded / 1024 / 1024
                    sys.stdout.write(f"\r    {mb:.1f}/{total/1024/1024:.1f}MB ({pct:.0f}%)")
                    sys.stdout.flush()

        tmp.rename(dest)
        print(f"\r    OK: {dest.name} ({downloaded/1024/1024:.1f}MB)                  ")
        return True
    except Exception as e:
        print(f"\n    Error: {e}")
        if tmp.exists():
            tmp.unlink()
        return False


def classify_post(post: dict) -> str:
    """Determine subdirectory based on post tags - picks the most specific anime tag."""
    post_tags = set(post.get("tags", "").split())

    # Try to match an anime tag
    for tag in ANIME_TAGS:
        if tag in post_tags:
            return tag

    # Check for _series variants
    for tag in post_tags:
        if tag.endswith("_series") or tag in ANIME_TAGS:
            return tag

    return "_other"


def scrape_phase(label: str, tag_groups: list, seen_ids: set, state: dict,
                 min_score: int, output_dir: Path, dry_run: bool,
                 global_stats: dict = None) -> dict:
    """Run a scrape phase (technique sweeps or anime sweeps)."""
    stats = {"found": 0, "downloaded": 0, "skipped": 0, "errors": 0, "deduped": 0}
    downloaded_set = set(state.get("downloaded_ids", []))
    if global_stats is None:
        global_stats = {"downloaded": len(downloaded_set), "bytes": 0,
                        "start_time": time.time()}

    for group_idx, (tags, limit) in enumerate(tag_groups):
        if isinstance(tags, str):
            tags = [tags]

        tag_label = " + ".join(tags)
        print(f"\n{'-'*60}")
        print(f"[{label} {group_idx+1}/{len(tag_groups)}] {tag_label} (limit={limit})")
        print(f"  Global progress: {global_stats['downloaded']} clips, "
              f"{global_stats['bytes']/1024/1024/1024:.1f} GB downloaded")
        print(f"{'-'*60}")

        posts = fetch_all_posts(tags, max_posts=limit, min_score=min_score,
                                seen_ids=seen_ids)
        stats["found"] += len(posts)

        if dry_run:
            if posts:
                sizes = [p.get("file_size", 0) for p in posts]
                total_mb = sum(sizes) / 1024 / 1024
                print(f"  Would download: {len(posts)} clips, ~{total_mb:.0f} MB")
            continue

        for i, post in enumerate(posts):
            pid = post["id"]

            if pid in downloaded_set:
                stats["deduped"] += 1
                continue

            file_url = post.get("file_url")
            if not file_url:
                continue

            # Organize by anime series
            anime_dir = classify_post(post)
            clip_dir = output_dir / anime_dir
            clip_dir.mkdir(parents=True, exist_ok=True)

            ext = post.get("file_ext", "mp4")
            score = post.get("score", 0)
            filename = f"{pid}_s{score}.{ext}"
            dest = clip_dir / filename

            # Save metadata
            meta_file = clip_dir / f"{pid}.json"
            if not meta_file.exists():
                with open(meta_file, "w") as f:
                    json.dump({
                        "id": pid,
                        "tags": post.get("tags", ""),
                        "score": score,
                        "source": post.get("source", ""),
                        "file_url": file_url,
                        "width": post.get("width"),
                        "height": post.get("height"),
                        "file_size": post.get("file_size"),
                    }, f, indent=2)

            if dest.exists():
                stats["skipped"] += 1
                downloaded_set.add(pid)
                file_size = post.get("file_size", 0)
                global_stats["bytes"] += file_size
                global_stats["downloaded"] += 1
                continue

            prefix = f"  [{i+1}/{len(posts)}]"
            print(f"{prefix} Post {pid} (score:{score}) -> {anime_dir}/")
            if download_file(file_url, dest):
                stats["downloaded"] += 1
                downloaded_set.add(pid)
                file_size = post.get("file_size", 0)
                global_stats["downloaded"] += 1
                global_stats["bytes"] += file_size
                # Periodic state save + progress
                if stats["downloaded"] % 50 == 0:
                    state["downloaded_ids"] = list(downloaded_set)
                    save_state(state)
                    elapsed = time.time() - global_stats["start_time"]
                    rate = global_stats["bytes"] / elapsed if elapsed > 0 else 0
                    print(f"\n  --- Progress: {global_stats['downloaded']} clips, "
                          f"{global_stats['bytes']/1024/1024/1024:.2f} GB, "
                          f"{rate/1024/1024:.1f} MB/s ---\n")
            else:
                stats["errors"] += 1
                state.setdefault("failed_ids", []).append(pid)

            time.sleep(DOWNLOAD_DELAY)

    # Final state save
    state["downloaded_ids"] = list(downloaded_set)
    save_state(state)
    return stats


def estimate(min_score: int = 0):
    """Estimate total clips and size for the default scrape plan.

    Samples 300 posts per tag with global dedup to measure overlap ratios,
    then extrapolates to the full limit.
    """
    SAMPLE = 300
    seen_ids = set()
    total_clips = 0
    total_bytes = 0

    print("Estimating default scrape plan (sampling %d posts per tag)...\n" % SAMPLE)

    # Also get total tag counts for better extrapolation
    def get_tag_count(tag):
        try:
            r = SESSION.get(
                f"{BASE_URL}/tag.json?name={tag}",
                timeout=10)
            if r.status_code == 200 and r.json():
                return r.json()[0].get("count", 0)
        except Exception:
            pass
        return 0

    def estimate_phase(label, groups):
        phase_clips = 0
        phase_bytes = 0
        print(f"{label}:")
        print(f"  {'Tag':<30} {'Avail':>6} {'Limit':>6} {'Dedup%':>7} {'EstUniq':>8} {'EstGB':>7}")
        print("  " + "-" * 70)
        for tags_item, limit in groups:
            tags = [tags_item] if isinstance(tags_item, str) else tags_item
            tag_label = "+".join(tags)

            # Get total available count
            avail = get_tag_count(tags[0]) if len(tags) == 1 else 0

            # Sample with global dedup
            sample_seen = set(seen_ids)
            posts = fetch_all_posts(tags, max_posts=SAMPLE,
                                    min_score=min_score, seen_ids=sample_seen)

            if not posts:
                print(f"  {tag_label:<30} {avail:>6} {limit:>6}     -        0    0.0")
                continue

            # Measure dedup ratio from sample
            new_ids = sample_seen - seen_ids
            dedup_ratio = len(new_ids) / max(len(posts) + (len(sample_seen) - len(seen_ids) - len(new_ids)), 1)

            avg_size = sum(p.get("file_size", 0) for p in posts) / len(posts)

            # Extrapolate: how many unique clips will we actually get from `limit` posts?
            effective_avail = min(limit, avail) if avail > 0 else limit
            est_unique = int(effective_avail * dedup_ratio)
            est_unique = max(est_unique, len(new_ids))  # at least what we sampled
            est_bytes = est_unique * avg_size
            est_gb = est_bytes / 1024 / 1024 / 1024

            # Add sampled IDs to global seen set
            seen_ids.update(new_ids)

            phase_clips += est_unique
            phase_bytes += est_bytes
            dedup_pct = (1 - dedup_ratio) * 100
            print(f"  {tag_label:<30} {avail:>6} {limit:>6} {dedup_pct:>6.0f}% {est_unique:>8} {est_gb:>6.1f}")
            time.sleep(0.3)

        phase_gb = phase_bytes / 1024 / 1024 / 1024
        print(f"  {'SUBTOTAL':<30} {'':>6} {'':>6} {'':>7} {phase_clips:>8} {phase_gb:>6.1f}")
        return phase_clips, phase_bytes

    # Phase 1
    tech_groups = [(t, l) for t, l in DEFAULT_SCRAPE_PLAN["technique_sweeps"]]
    p1_clips, p1_bytes = estimate_phase("Phase 1: Technique sweeps", tech_groups)
    total_clips += p1_clips
    total_bytes += p1_bytes

    print()

    # Phase 2
    anime_groups = [(a, l) for a, l in DEFAULT_SCRAPE_PLAN["anime_sweeps"]]
    p2_clips, p2_bytes = estimate_phase("Phase 2: Anime sweeps", anime_groups)
    total_clips += p2_clips
    total_bytes += p2_bytes

    total_gb = total_bytes / 1024 / 1024 / 1024
    est_hours = total_clips * 48 / 3600
    print(f"\n{'='*60}")
    print(f"ESTIMATE TOTAL:")
    print(f"  Unique clips:     ~{total_clips}")
    print(f"  Total size:       ~{total_gb:.0f} GB")
    print(f"  Est. duration:    ~{est_hours:.0f} hours (at ~48s avg/clip)")
    print(f"  Avg clip size:    ~{total_bytes/max(total_clips,1)/1024/1024:.1f} MB")
    print(f"\nNote: Estimates based on {SAMPLE}-post samples with dedup extrapolation.")


def list_tags():
    """Print available tags."""
    print("\nAnime Tags (series):")
    print("-" * 70)
    for tag, desc in ANIME_TAGS.items():
        print(f"  {tag:45s} {desc}")

    print("\nTechnique Tags:")
    print("-" * 70)
    for tag, desc in TECHNIQUE_TAGS.items():
        print(f"  {tag:25s} {desc}")

    plan = DEFAULT_SCRAPE_PLAN
    n_tech = len(plan["technique_sweeps"])
    n_anime = len(plan["anime_sweeps"])
    total_limit = (sum(l for _, l in plan["technique_sweeps"])
                   + sum(l for _, l in plan["anime_sweeps"]))
    print(f"\nDefault plan: {n_tech} technique sweeps + {n_anime} anime sweeps")
    print(f"Total limit: {total_limit} clips (before deduplication)")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape sakuga clips from Sakugabooru (~100 hours target)"
    )
    parser.add_argument("--anime", type=str, nargs="+",
                        help="Anime tag(s) to scrape")
    parser.add_argument("--techniques", type=str, nargs="+",
                        help="Technique tag(s) to scrape")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max clips per tag group")
    parser.add_argument("--min-score", type=int, default=0,
                        help="Minimum score threshold")
    parser.add_argument("--list-tags", action="store_true",
                        help="List available tags")
    parser.add_argument("--estimate", action="store_true",
                        help="Estimate download size without downloading")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without downloading")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted download")

    args = parser.parse_args()

    if args.list_tags:
        list_tags()
        return

    if args.estimate:
        estimate(args.min_score)
        return

    global OUTPUT_DIR, STATE_FILE
    if args.output:
        OUTPUT_DIR = Path(args.output)
        STATE_FILE = OUTPUT_DIR / "_state.json"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Always load state — safe to restart/resume at any point
    state = load_state()
    seen_ids = set(state.get("downloaded_ids", []))
    initial_count = len(seen_ids)

    if initial_count > 0:
        print(f"Resuming: {initial_count} clips already downloaded (use fresh dir to start over)")

    # Shared progress tracker across phases
    global_stats = {"downloaded": initial_count, "bytes": 0,
                    "start_time": time.time()}

    # Build scrape plan
    if args.anime or args.techniques:
        # Custom scrape
        groups = []
        if args.anime and args.techniques:
            for a in args.anime:
                for t in args.techniques:
                    groups.append(([a, t], args.limit or 500))
        elif args.anime:
            for a in args.anime:
                groups.append(([a], args.limit or 500))
        else:
            for t in args.techniques:
                groups.append(([t], args.limit or 1000))

        print(f"Sakugabooru Scraper - Custom")
        print(f"Output: {OUTPUT_DIR}")
        print(f"Groups: {len(groups)}")

        stats = scrape_phase("custom", groups, seen_ids, state,
                             args.min_score, OUTPUT_DIR, args.dry_run,
                             global_stats)
        print_final_stats({"custom": stats}, global_stats)
    else:
        # Full default plan
        plan = DEFAULT_SCRAPE_PLAN
        total_limit = (sum(l for _, l in plan["technique_sweeps"])
                       + sum(l for _, l in plan["anime_sweeps"]))
        print(f"Sakugabooru Scraper - Full Plan (~100 hours target)")
        print(f"Output: {OUTPUT_DIR}")
        print(f"Total limit: {total_limit} clips (before dedup)")
        print(f"Min score: {args.min_score}")

        if args.dry_run:
            print("\n[DRY RUN]")

        # Phase 1: technique sweeps
        tech_groups = [([t], l) for t, l in plan["technique_sweeps"]]
        stats1 = scrape_phase("technique", tech_groups, seen_ids, state,
                              args.min_score, OUTPUT_DIR, args.dry_run,
                              global_stats)

        # Phase 2: anime sweeps
        anime_groups = [([a], l) for a, l in plan["anime_sweeps"]]
        stats2 = scrape_phase("anime", anime_groups, seen_ids, state,
                              args.min_score, OUTPUT_DIR, args.dry_run,
                              global_stats)

        print_final_stats({"technique_sweeps": stats1, "anime_sweeps": stats2},
                          global_stats)


def print_final_stats(phase_stats: dict, global_stats: dict = None):
    """Print summary."""
    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"{'='*60}")
    totals = {"found": 0, "downloaded": 0, "skipped": 0, "errors": 0, "deduped": 0}
    for phase, stats in phase_stats.items():
        print(f"\n  {phase}:")
        for k, v in stats.items():
            print(f"    {k:12s}: {v}")
            totals[k] += v
    print(f"\n  TOTAL:")
    for k, v in totals.items():
        print(f"    {k:12s}: {v}")

    if global_stats:
        elapsed = time.time() - global_stats["start_time"]
        gb = global_stats["bytes"] / 1024 / 1024 / 1024
        est_hours = global_stats["downloaded"] * 48 / 3600
        print(f"\n  Total clips:      {global_stats['downloaded']}")
        print(f"  Total size:       {gb:.1f} GB")
        print(f"  Est. duration:    ~{est_hours:.0f} hours (at ~48s avg/clip)")
        print(f"  Elapsed time:     {elapsed/3600:.1f} hours")

    print(f"\n  Output: {OUTPUT_DIR}")
    print(f"\n  To resume if interrupted: python3 {sys.argv[0]} --resume")


if __name__ == "__main__":
    main()
