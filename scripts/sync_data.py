"""Read-only sync of the NFL collector's data into data/raw/.

data/raw/ mirrors the server's data folder, so it can be passed straight to
`python -m model.export_joint --data-dir`:

    data/raw/fairvalues/   every fair-value CSV (the loader decides which ones count)
    data/raw/nflverse/     every games-*.csv the daily run downloaded
    data/raw/tidy.sqlite   order books, trades, rounds
    data/raw/SYNC.json     when and what was copied

Nothing is written on the server, and no SQLite lock is taken there:
- The collector opens tidy.sqlite with a 5 s busy timeout, so a long-running reader
  (e.g. `sqlite3 .dump`) could make a collector round fail. rsync only reads the file.
- The database uses a rollback journal. A copy is consistent if the file's size and
  mtime are unchanged across the copy and no -journal file existed (no write in
  progress). Otherwise the copy is retried.

Usage: python scripts/sync_data.py [--host nfl] [--skip-db]
"""

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REMOTE_DIR = "/var/lib/nfl-collector"
LOCAL_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DB = "tidy.sqlite"
ATTEMPTS = 3
RETRY_WAIT_SECONDS = 60


def ssh(host, command):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, command],
                          check=True, capture_output=True, text=True).stdout


def rsync(host, remote, local):
    # -r recursive, -t keep mtimes (used by export_joint's "--nflverse auto"); no --delete,
    # so files are never removed locally.
    subprocess.run(["rsync", "-rt", "--human-readable", "--stats", f"{host}:{remote}", str(local)],
                   check=True, stdout=subprocess.DEVNULL)


def remote_db_state(host):
    """(size and nanosecond mtime of tidy.sqlite, whether a -journal file exists)."""
    path = f"{REMOTE_DIR}/{DB}"
    out = ssh(host, f"stat -c '%s %y' {path}; test -e {path}-journal && echo journal || true")
    lines = out.strip().splitlines()
    return lines[0], "journal" in lines[1:]


def sync_db(host):
    """Copy tidy.sqlite into data/raw/.staging, verify it, then install it in data/raw/."""
    staging = LOCAL_DIR / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, ATTEMPTS + 1):
        before, journal_before = remote_db_state(host)
        if journal_before:
            print(f"  {DB}: write in progress on the server (attempt {attempt})")
        else:
            rsync(host, f"{REMOTE_DIR}/{DB}", staging)
            after, journal_after = remote_db_state(host)
            if before == after and not journal_after:
                check_db(staging / DB)
                shutil.copy2(staging / DB, LOCAL_DIR / DB)
                return before
            print(f"  {DB}: changed during the copy (attempt {attempt})")
        if attempt < ATTEMPTS:
            time.sleep(RETRY_WAIT_SECONDS)
    sys.exit(f"{DB}: no consistent copy after {ATTEMPTS} attempts; data/raw/{DB} left as it was")


def check_db(path):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        result = db.execute("PRAGMA integrity_check").fetchall()
    finally:
        db.close()
    if result != [("ok",)]:
        sys.exit(f"{path}: integrity_check failed: {result[:5]}")


def summarize_db(path):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rounds, first_slot, last_slot = db.execute(
            "SELECT count(*), min(slot), max(slot) FROM rounds").fetchone()
        books = db.execute("SELECT count(*) FROM books").fetchone()[0]
        trades, last_trade = db.execute("SELECT count(*), max(created_time) FROM trades").fetchone()
    finally:
        db.close()
    return {"rounds": rounds, "first_slot": first_slot, "last_slot": last_slot,
            "books": books, "trades": trades, "last_trade": last_trade}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="nfl", help="ssh host alias (default: nfl)")
    parser.add_argument("--skip-db", action="store_true", help="only sync the CSV folders")
    args = parser.parse_args()

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    manifest = {"host": args.host, "remote_dir": REMOTE_DIR, "started": started.isoformat(timespec="seconds")}

    for folder in ("fairvalues", "nflverse"):
        rsync(args.host, f"{REMOTE_DIR}/{folder}", LOCAL_DIR)
        files = sorted(p.name for p in (LOCAL_DIR / folder).glob("*.csv"))
        manifest[folder] = {"files": len(files), "last": files[-1] if files else None}
        print(f"{folder}: {len(files)} files, last {manifest[folder]['last']}")

    if not args.skip_db:
        print(f"{DB}: copying (rsync, no lock on the server) ...")
        manifest[DB] = {"remote_stat": sync_db(args.host), **summarize_db(LOCAL_DIR / DB)}
        s = manifest[DB]
        print(f"{DB}: integrity ok, {s['rounds']} rounds {s['first_slot']} .. {s['last_slot']}, "
              f"{s['books']} books, {s['trades']} trades (last {s['last_trade']})")

    manifest["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (LOCAL_DIR / "SYNC.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {LOCAL_DIR / 'SYNC.json'}")


if __name__ == "__main__":
    main()
