#!/usr/bin/env python3
"""
Extract Cloudflare cookies for Sakugabooru.

Opens a real browser window so you can solve the Cloudflare challenge manually.
Once the page loads, the cookies are saved to cf_cookies.json for the scraper.

Usage:
    python extract_cf_cookies.py              # Opens browser, you solve challenge
    python extract_cf_cookies.py --manual     # Paste cookies from browser DevTools instead
"""

import argparse
import json
import sys
from pathlib import Path

COOKIE_FILE = Path(__file__).resolve().parents[2] / "data" / "sakugabooru" / "cf_cookies.json"
TARGET = "https://www.sakugabooru.com/"


def extract_with_browser():
    """Open a real browser, let user solve CF challenge, then save cookies."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed. Run: pip install playwright && python -m playwright install firefox")
        sys.exit(1)

    print("Opening browser - please solve the Cloudflare challenge...")
    print("The script will automatically save cookies once the page loads.\n")

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=False)
        ctx = browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
            viewport={"width": 1280, "height": 900},
        )
        page = ctx.new_page()
        page.goto(TARGET, timeout=120000)

        # Wait for CF challenge to be solved (title changes from "Just a moment...")
        print("Waiting for Cloudflare challenge to be solved...")
        try:
            page.wait_for_function(
                "() => !document.title.toLowerCase().includes('just a moment')",
                timeout=120000,
            )
        except Exception:
            print("Timed out waiting for challenge. Try again or use --manual mode.")
            browser.close()
            sys.exit(1)

        print(f"Challenge solved! Page title: {page.title()}")

        # Collect all cookies for sakugabooru.com
        cookies = {}
        for c in ctx.cookies():
            if "sakugabooru" in c.get("domain", ""):
                cookies[c["name"]] = c["value"]

        browser.close()

    if not cookies:
        print("No sakugabooru cookies found. Something went wrong.")
        sys.exit(1)

    save_cookies(cookies)


def extract_manual():
    """Let user paste cookie string from browser DevTools."""
    print("How to get cookies from your browser:")
    print("  1. Open https://www.sakugabooru.com in your browser")
    print("  2. Solve the Cloudflare challenge if prompted")
    print("  3. Open DevTools (F12) -> Network tab")
    print("  4. Refresh the page")
    print("  5. Click any request to sakugabooru.com")
    print("  6. In Headers, find 'Cookie:' and copy its value")
    print()
    raw = input("Paste the Cookie header value: ").strip()

    if not raw:
        print("No input. Aborting.")
        sys.exit(1)

    cookies = {}
    for part in raw.split(";"):
        part = part.strip()
        if "=" in part:
            name, value = part.split("=", 1)
            cookies[name.strip()] = value.strip()

    if not cookies:
        print("Could not parse any cookies. Aborting.")
        sys.exit(1)

    save_cookies(cookies)


def save_cookies(cookies: dict):
    """Save cookies dict to cf_cookies.json."""
    COOKIE_FILE.write_text(json.dumps(cookies, indent=2))
    print(f"\nSaved {len(cookies)} cookies to {COOKIE_FILE}")
    print("Cookies found:")
    for name in cookies:
        print(f"  {name}")

    # Quick validation
    import requests
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
    })
    for name, value in cookies.items():
        s.cookies.set(name, value, domain=".sakugabooru.com")

    print("\nTesting API access...")
    try:
        r = s.get("https://www.sakugabooru.com/post.json?tags=smears+order:score&page=1&limit=1", timeout=10)
        if r.status_code == 200:
            data = r.json()
            print(f"SUCCESS - API returned {len(data)} post(s)")
        else:
            print(f"FAILED - Status {r.status_code}")
            if "Just a moment" in r.text:
                print("Still hitting Cloudflare. Cookies may have expired. Try again.")
    except Exception as e:
        print(f"Error testing: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Cloudflare cookies for Sakugabooru scraper")
    parser.add_argument("--manual", action="store_true",
                        help="Paste cookies from browser DevTools instead of opening a browser")
    args = parser.parse_args()

    if args.manual:
        extract_manual()
    else:
        extract_with_browser()
