"""Load and validate one decision snapshot: fair values, prices and joint outcomes at time T.

Inputs (see scripts/sync_data.py and model/export_joint.py in nfl-playoff-algo):
    data/raw/fairvalues/<T>-calibrated-hl2-k.csv     q, bid/ask at T (one row per team)
    data/joint/<T>-x<scale>-s<sims>-seed<seed>.npz   S x 32 seed matrix (0 = missed playoffs)

Everything is checked on load. Any failure raises DataError instead of returning
something that looks fine.

Prices: an empty side of the book is kept as NaN, not filled in. With no YES ask you
can't buy YES; with no YES bid you can't buy NO (NO costs 1 - bid). Filling them with
ask = 1 or bid = 0 would offer a contract with mean PnL <= 0 that MVO might still buy
as a hedge. Use `can_buy_yes` / `can_buy_no`.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
JOINT_DIR = ROOT / "data" / "joint"

# First fair-value file that can be point-in-time: the daily run after the settings
# commits of 2026-11-06 ~04:08 UTC. Files before this came from a one-off backfill.
CLEAN_FROM = datetime(2026, 11, 6, 14, 0, tzinfo=timezone.utc)
# Only the scheduled daily run (14:00 UTC); ad-hoc runs at other times are skipped.
DAILY_FILE = re.compile(r"^(\d{8}T1400Z)-calibrated-hl2-k\.csv$")
SCALES = (1.0, 1.5, 2.0)
CENTRAL_SCALE = 1.5  # the scale prod's CSV is computed at

WILD_CARDS = 3
PLAYOFF_TEAMS = 4 + WILD_CARDS  # per conference
TO_KALSHI = {"JAX": "JAC", "LA": "LAR"}  # nflverse code -> Kalshi code, where they differ
PROB_TOL = 1e-9


class DataError(ValueError):
    """Input data failed validation."""


def playoff_ticker(team):
    return f"KXNFLPLAYOFF-27-{TO_KALSHI.get(team, team)}"


def stamp(at):
    return at.astimezone(timezone.utc).strftime("%Y%m%dT%H%MZ")


def parse_stamp(text):
    return datetime.strptime(text, "%Y%m%dT%H%MZ").replace(tzinfo=timezone.utc)


def parse_iso(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def clean_times(raw_dir=RAW_DIR):
    """Times T of the daily fair-value files that can be point-in-time, oldest first."""
    times = []
    for path in (Path(raw_dir) / "fairvalues").glob("*.csv"):
        match = DAILY_FILE.match(path.name)
        if match and parse_stamp(match.group(1)) >= CLEAN_FROM:
            times.append(parse_stamp(match.group(1)))
    return sorted(times)


def fairvalue_path(at, raw_dir=RAW_DIR):
    return Path(raw_dir) / "fairvalues" / f"{stamp(at)}-calibrated-hl2-k.csv"


def joint_path(at, scale, joint_dir=JOINT_DIR, sims=20000, seed=0):
    return Path(joint_dir) / f"{stamp(at)}-x{scale:g}-s{sims}-seed{seed}.npz"


def _require(condition, message):
    if not condition:
        raise DataError(message)


# --- fair-value CSV ---------------------------------------------------------------------

def load_fairvalues(path):
    """The fair-value CSV, validated, indexed by team."""
    path = Path(path)
    df = pd.read_csv(path)
    where = path.name
    _require(len(df) == 32 and df.team.is_unique, f"{where}: expected 32 distinct teams")
    _require((df.conference.value_counts() == 16).all() and set(df.conference) == {"AFC", "NFC"},
             f"{where}: expected 16 AFC and 16 NFC teams")
    for column in ("sims", "tau", "half_life", "slot", "sigma"):
        _require(df[column].nunique() == 1, f"{where}: {column} differs between rows")
    _require(df.p_playoffs.between(0, 1).all(), f"{where}: p_playoffs outside [0, 1]")
    # Counts from one simulation: exactly 7 teams per conference make it in every season.
    sums = df.groupby("conference").p_playoffs.sum()
    _require(np.allclose(sums, PLAYOFF_TEAMS, atol=PROB_TOL),
             f"{where}: p_playoffs sums to {sums.to_dict()}, not {PLAYOFF_TEAMS} per conference")
    _require(((df.p_playoffs_low <= df.p_playoffs) & (df.p_playoffs <= df.p_playoffs_high)).all(),
             f"{where}: p_playoffs outside its own band")
    for side in ("playoff_bid", "playoff_ask"):
        prices = df[side].dropna()
        _require(((prices > 0) & (prices < 1)).all(), f"{where}: {side} outside (0, 1)")
    both = df.playoff_bid.notna() & df.playoff_ask.notna()
    _require((df.playoff_bid[both] < df.playoff_ask[both]).all(), f"{where}: crossed or locked book")
    return df.set_index("team")


# --- joint outcomes ---------------------------------------------------------------------

def check_seed_matrix(seed, conference):
    """Every simulated season seeds each conference's teams 1..7 exactly once.

    seed: S x 32 ints, 0 = missed the playoffs. conference: 32 labels.
    """
    _require(seed.ndim == 2 and seed.shape[1] == 32, f"seed matrix has shape {seed.shape}")
    _require(seed.min() >= 0 and seed.max() <= PLAYOFF_TEAMS, "seed outside 0..7")
    for conf in ("AFC", "NFC"):
        block = seed[:, conference == conf]
        _require(block.shape[1] == 16, f"{conf}: {block.shape[1]} teams, not 16")
        for value in range(1, PLAYOFF_TEAMS + 1):
            counts = (block == value).sum(axis=1)
            bad = np.flatnonzero(counts != 1)
            if bad.size:
                raise DataError(f"{conf}: seed {value} used {counts[bad[0]]} times in sim {bad[0]}")


def load_joint(path, conference_of):
    """The joint-outcome export, validated. conference_of: {team: 'AFC' | 'NFC'}."""
    path = Path(path)
    where = path.name
    with np.load(path) as z:  # plain arrays and strings only; no pickle
        raw = {key: z[key] for key in z.files}
    settings = json.loads(str(raw["settings"]))
    teams = raw["teams"].astype(str)
    seed = raw["seed"]
    _require(len(teams) == 32 and len(set(teams)) == 32, f"{where}: expected 32 distinct teams")
    _require(set(teams) == set(conference_of), f"{where}: teams differ from the fair-value CSV")
    _require(list(raw["tickers"].astype(str)) == [playoff_ticker(t) for t in teams],
             f"{where}: tickers don't match KXNFLPLAYOFF-27-<code>")
    _require(seed.dtype == np.int8 and seed.shape == (int(raw["sims"]), 32),
             f"{where}: seed matrix {seed.dtype} {seed.shape}, expected int8 ({int(raw['sims'])}, 32)")
    _require(not settings["git_dirty"], f"{where}: exported from a checkout with uncommitted changes")
    conference = np.array([conference_of[t] for t in teams])
    check_seed_matrix(seed, conference)
    made = seed > 0
    _require(np.array_equal(made.sum(axis=0), np.rint(raw["p_playoffs"] * len(seed))),
             f"{where}: stored p_playoffs disagrees with the seed matrix")
    return {"teams": teams, "conference": conference, "seed": seed, "settings": settings,
            "at": parse_iso(str(raw["at"])), "scale": float(raw["scale"]),
            "kalshi_slot": str(raw["kalshi_slot"])}


# --- snapshot ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Snapshot:
    """One decision time T at one uncertainty scale. Arrays are in `teams` order."""
    at: datetime
    scale: float
    teams: np.ndarray  # 32 team codes (nflverse)
    conference: np.ndarray  # 'AFC' / 'NFC'
    tickers: np.ndarray  # KXNFLPLAYOFF-27-<code>
    bid: np.ndarray  # YES bid in dollars, NaN if none
    ask: np.ndarray  # YES ask in dollars, NaN if none
    seed: np.ndarray = field(repr=False)  # S x 32 int8, 0 = missed the playoffs
    settings: dict = field(repr=False)

    @property
    def sims(self):
        return self.seed.shape[0]

    @property
    def outcomes(self):
        """S x 32 floats: X = 1 if the team made the playoffs in that simulated season."""
        return (self.seed > 0).astype(float)

    @property
    def q(self):
        """Model P(make playoffs) at this scale: the mean of `outcomes`, so it matches any
        covariance estimated from the same sims."""
        return self.outcomes.mean(axis=0)

    @property
    def can_buy_yes(self):
        return ~np.isnan(self.ask)

    @property
    def can_buy_no(self):
        return ~np.isnan(self.bid)

    @property
    def mid(self):
        """(bid + ask) / 2, NaN unless both sides are quoted."""
        return (self.bid + self.ask) / 2


def load_snapshot(at, scale=CENTRAL_SCALE, raw_dir=RAW_DIR, joint_dir=JOINT_DIR, sims=20000, seed=0):
    """Fair values, prices and joint outcomes at T, cross-checked against each other."""
    _require(at >= CLEAN_FROM, f"{stamp(at)} is before {stamp(CLEAN_FROM)}: not point-in-time")
    _require(scale in SCALES, f"scale {scale} not one of {SCALES}")
    fv = load_fairvalues(fairvalue_path(at, raw_dir))
    joint = load_joint(joint_path(at, scale, joint_dir, sims, seed), fv.conference.to_dict())
    where = f"{stamp(at)} x{scale:g}"

    _require(joint["at"] == at, f"{where}: export is for {joint['at']}")
    _require(joint["scale"] == scale, f"{where}: export is for scale {joint['scale']}")
    _require(parse_iso(fv.slot.iloc[0]) == parse_iso(joint["kalshi_slot"]),
             f"{where}: CSV slot {fv.slot.iloc[0]} vs export {joint['kalshi_slot']}")
    _require(int(fv.sims.iloc[0]) == joint["seed"].shape[0], f"{where}: sims differ from the CSV")

    fv = fv.loc[joint["teams"]]
    made = joint["seed"] > 0
    q = made.mean(axis=0)
    if scale == CENTRAL_SCALE:
        # Same run as prod's CSV: counts must match exactly.
        _require(np.array_equal(made.sum(axis=0), np.rint(fv.p_playoffs.to_numpy() * len(made))),
                 f"{where}: playoff counts differ from the fair-value CSV")
    else:
        # The CSV band spans the x1..x2 runs +/- 2 SE, so every scale's q must lie inside it.
        inside = (fv.p_playoffs_low.to_numpy() - PROB_TOL <= q) & (q <= fv.p_playoffs_high.to_numpy() + PROB_TOL)
        _require(inside.all(), f"{where}: q outside the CSV band for {list(joint['teams'][~inside])}")

    return Snapshot(at=at, scale=scale, teams=joint["teams"], conference=joint["conference"],
                    tickers=np.array([playoff_ticker(t) for t in joint["teams"]]),
                    bid=fv.playoff_bid.to_numpy(float), ask=fv.playoff_ask.to_numpy(float),
                    seed=joint["seed"], settings=joint["settings"])
