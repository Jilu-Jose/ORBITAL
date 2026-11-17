#!/usr/bin/env python3
"""Download gp.json, satcat.json and cdm.json from Space-Track.org into backend/data/raw/.

    export SPACETRACK_USER="you@email.com"
    export SPACETRACK_PASS="your-password"
    python backend/spark/download_data.py                # defaults: 365 days of CDMs
    python backend/spark/download_data.py --cdm-days 730

Space-Track rate limit is roughly 30 requests/min and 300/hour; this script sleeps
between calls and retries on HTTP 429/5xx. Space-Track terms forbid redistributing
raw data, so keep data/raw out of git.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import requests

BASE = "https://www.space-track.org"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SLEEP_S = 2.5  # ~24 requests/min, under the 30/min limit


def login(session: requests.Session, user: str, password: str) -> None:
    r = session.post(f"{BASE}/ajaxauth/login", data={"identity": user, "password": password}, timeout=60)
    r.raise_for_status()
    if "Failed" in r.text:
        sys.exit("Space-Track login failed. Check SPACETRACK_USER / SPACETRACK_PASS.")


def get_json(session: requests.Session, path: str, retries: int = 5) -> list:
    url = f"{BASE}{path}"
    for attempt in range(1, retries + 1):
        time.sleep(SLEEP_S)
        r = session.get(url, timeout=300)
        if r.status_code == 200:
            try:
                data = r.json()
            except ValueError:
                sys.exit(f"Non-JSON response for {path}: {r.text[:200]}")
            if isinstance(data, dict) and data.get("error"):
                sys.exit(f"Space-Track error for {path}: {data['error']}")
            return data
        if r.status_code in (429, 500, 502, 503, 504):
            wait = 60 * attempt
            print(f"  HTTP {r.status_code}; waiting {wait}s (attempt {attempt}/{retries})")
            time.sleep(wait)
            continue
        sys.exit(f"HTTP {r.status_code} for {path}: {r.text[:200]}")
    sys.exit(f"Giving up on {path} after {retries} attempts.")


def save(name: str, data: list) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    p = RAW_DIR / name
    p.write_text(json.dumps(data))
    print(f"  wrote {p} ({len(data):,} rows, {p.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cdm-days", type=int, default=365, help="how many days of public CDMs to fetch")
    ap.add_argument("--chunk-days", type=int, default=7, help="CDM query window size")
    ap.add_argument("--skip", nargs="*", default=[], choices=["gp", "satcat", "cdm"])
    args = ap.parse_args()

    user, password = os.environ.get("SPACETRACK_USER"), os.environ.get("SPACETRACK_PASS")
    if not user or not password:
        sys.exit("Set SPACETRACK_USER and SPACETRACK_PASS environment variables first.")

    s = requests.Session()
    login(s, user, password)
    print("Logged in.")

    try:
        if "gp" not in args.skip:
            print("Fetching GP (current elements, not decayed, epoch within 30 days)...")
            gp = get_json(s, "/basicspacedata/query/class/gp/decay_date/null-val/epoch/%3Enow-30/orderby/norad_cat_id%20asc/format/json")
            save("gp.json", gp)

        if "satcat" not in args.skip:
            print("Fetching SATCAT (current, not decayed)...")
            sat = get_json(s, "/basicspacedata/query/class/satcat/current/Y/decay/null-val/orderby/norad_cat_id%20asc/format/json")
            save("satcat.json", sat)

        if "cdm" not in args.skip:
            print(f"Fetching public CDMs for the last {args.cdm_days} days in {args.chunk_days}-day chunks...")
            end = dt.date.today() + dt.timedelta(days=1)
            start = end - dt.timedelta(days=args.cdm_days)
            rows: list = []
            cur = start
            while cur < end:
                nxt = min(cur + dt.timedelta(days=args.chunk_days), end)
                path = f"/basicspacedata/query/class/cdm_public/CREATED/{cur}--{nxt}/orderby/CREATED%20asc/format/json"
                chunk = get_json(s, path)
                rows.extend(chunk)
                print(f"  {cur} -> {nxt}: {len(chunk):,} CDMs (total {len(rows):,})")
                cur = nxt
            save("cdm.json", rows)
    finally:
        try:
            s.get(f"{BASE}/ajaxauth/logout", timeout=30)
        except requests.RequestException:
            pass

    print("Done. Next: python backend/spark/train.py")


if __name__ == "__main__":
    main()
