"""Loader tests on a small synthetic world (hand-checkable), plus the real data if synced."""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from mvo_binaries import data
from mvo_binaries.data import DataError

AFC = ["BAL", "BUF", "CIN", "CLE", "DEN", "HOU", "IND", "JAX", "KC", "LAC", "LV", "MIA", "NE", "NYJ", "PIT", "TEN"]
NFC = ["ARI", "ATL", "CAR", "CHI", "DAL", "DET", "GB", "LA", "MIN", "NO", "NYG", "PHI", "SEA", "SF", "TB", "WAS"]
TEAMS = sorted(AFC + NFC)
CONF = {t: "AFC" if t in AFC else "NFC" for t in TEAMS}
AT = datetime(2026, 11, 8, 14, 0, tzinfo=timezone.utc)
SIMS = 4


def seed_matrix():
    """4 sims. In sim s, conference teams at positions s..s+6 (mod 16) get seeds 1..7."""
    seed = np.zeros((SIMS, 32), np.int8)
    for conf_teams in (AFC, NFC):
        for s in range(SIMS):
            for k in range(7):
                seed[s, TEAMS.index(conf_teams[(s + k) % 16])] = k + 1
    return seed


def write_world(tmp_path, seed=None, scale=1.5, edit_csv=None, edit_npz=None):
    seed = seed_matrix() if seed is None else seed
    raw, joint = tmp_path / "raw", tmp_path / "joint"
    (raw / "fairvalues").mkdir(parents=True)
    joint.mkdir()
    p = (seed > 0).mean(axis=0)
    slot = "2026-11-08T14:00:00Z"
    df = pd.DataFrame({
        "team": TEAMS, "conference": [CONF[t] for t in TEAMS], "rating": 0.0,
        "p_playoffs": p, "se": 0.0, "p_playoffs_low": np.clip(p - 0.3, 0, 1), "p_playoffs_high": np.clip(p + 0.3, 0, 1),
        "p_division": 0.0, "p_division_low": 0.0, "p_division_high": 0.0,
        "playoff_bid": 0.40, "playoff_ask": 0.45, "division_bid": np.nan, "division_ask": np.nan,
        "sigma": 12.0, "tau": "calibrated", "half_life": 2, "sims": SIMS, "slot": slot,
    })
    if edit_csv:
        edit_csv(df)
    df.to_csv(data.fairvalue_path(AT, raw), index=False)
    arrays = {
        "seed": seed, "teams": np.array(TEAMS), "tickers": np.array([data.playoff_ticker(t) for t in TEAMS]),
        "p_playoffs": p, "at": np.array(slot), "kalshi_slot": np.array(slot), "sims": np.array(SIMS),
        "scale": np.array(scale), "settings": np.array(json.dumps({"git_dirty": False})),
    }
    if edit_npz:
        edit_npz(arrays)
    np.savez(data.joint_path(AT, scale, joint, sims=SIMS), **arrays)
    return raw, joint


def load(raw, joint, scale=1.5):
    return data.load_snapshot(AT, scale, raw, joint, sims=SIMS)


def test_valid_world_loads(tmp_path):
    snap = load(*write_world(tmp_path))
    assert snap.sims == SIMS and list(snap.teams) == TEAMS
    # BAL is AFC position 0: seeded only in sim 0 (positions 0..6) -> q = 1/4.
    assert snap.q[TEAMS.index("BAL")] == 0.25
    # BUF (AFC 1) in sims 0, 1; CIN (AFC 2) in sims 0..2; CLE (AFC 3) in all 4.
    assert snap.q[[TEAMS.index(t) for t in ("BUF", "CIN", "CLE")]].tolist() == [0.5, 0.75, 1.0]
    assert np.allclose(snap.outcomes[:, snap.conference == "AFC"].sum(axis=1), 7)
    assert snap.tickers[TEAMS.index("JAX")] == "KXNFLPLAYOFF-27-JAC"
    assert snap.tickers[TEAMS.index("LA")] == "KXNFLPLAYOFF-27-LAR"
    assert np.allclose(snap.mid, 0.425)


def test_missing_prices_block_that_side(tmp_path):
    def edit(df):
        df.loc[df.team == "BUF", "playoff_ask"] = np.nan
        df.loc[df.team == "MIA", "playoff_bid"] = np.nan
    snap = load(*write_world(tmp_path, edit_csv=edit))
    buf, mia = TEAMS.index("BUF"), TEAMS.index("MIA")
    assert not snap.can_buy_yes[buf] and snap.can_buy_no[buf]
    assert snap.can_buy_yes[mia] and not snap.can_buy_no[mia]
    assert np.isnan(snap.mid[buf]) and snap.can_buy_yes.sum() == 31


def test_eight_playoff_teams_in_a_sim_fails(tmp_path):
    def edit(arrays):  # corrupt the export only; the CSV stays consistent
        arrays["seed"] = arrays["seed"].copy()
        arrays["seed"][2, TEAMS.index("BAL")] = 7  # BAL not seeded in sim 2; now seed 7 is used twice
    with pytest.raises(DataError, match="seed 7 used 2 times in sim 2"):
        load(*write_world(tmp_path, edit_npz=edit))


def test_wrong_ticker_fails(tmp_path):
    def edit(arrays):
        arrays["tickers"] = np.array([f"KXNFLPLAYOFF-27-{t}" for t in TEAMS])  # JAX, LA unmapped
    with pytest.raises(DataError, match="tickers"):
        load(*write_world(tmp_path, edit_npz=edit))


def test_central_scale_must_match_csv_exactly(tmp_path):
    def edit(df):  # move 1 sim's worth of probability between two AFC teams; sums stay 7
        df.loc[df.team == "BAL", "p_playoffs"] += 0.25
        df.loc[df.team == "BUF", "p_playoffs"] -= 0.25
    with pytest.raises(DataError, match="counts differ"):
        load(*write_world(tmp_path, edit_csv=edit))


def test_other_scale_must_lie_in_csv_band(tmp_path):
    def edit(df):
        df.loc[df.team == "BAL", "p_playoffs_high"] = 0.2  # BAL's q = 0.25 at this scale
        df.loc[df.team == "BAL", "p_playoffs"] = 0.2
        df.loc[df.team == "BUF", "p_playoffs"] += 0.05
    raw, joint = write_world(tmp_path, scale=2.0, edit_csv=edit)
    with pytest.raises(DataError, match="outside the CSV band.*BAL"):
        load(raw, joint, scale=2.0)


def test_crossed_book_fails(tmp_path):
    def edit(df):
        df.loc[df.team == "KC", "playoff_bid"] = 0.45
    with pytest.raises(DataError, match="crossed"):
        load(*write_world(tmp_path, edit_csv=edit))


def test_dirty_export_fails(tmp_path):
    def edit(arrays):
        arrays["settings"] = np.array(json.dumps({"git_dirty": True}))
    with pytest.raises(DataError, match="uncommitted"):
        load(*write_world(tmp_path, edit_npz=edit))


def test_before_clean_date_refused(tmp_path):
    with pytest.raises(DataError, match="not point-in-time"):
        data.load_snapshot(datetime(2026, 11, 5, 14, 0, tzinfo=timezone.utc), raw_dir=tmp_path)


def test_clean_times_keeps_daily_runs_from_nov_6(tmp_path):
    folder = tmp_path / "fairvalues"
    folder.mkdir()
    for name in ("20261105T1400Z", "20261106T1400Z", "20261108T0330Z", "20261108T1400Z"):
        (folder / f"{name}-calibrated-hl2-k.csv").touch()
    (folder / "20261109T1400Z-other.csv").touch()
    assert [data.stamp(t) for t in data.clean_times(tmp_path)] == ["20261106T1400Z", "20261108T1400Z"]


# --- the real synced data -----------------------------------------------------------------

REAL = data.clean_times()


@pytest.mark.skipif(not REAL, reason="no synced data in data/raw")
@pytest.mark.parametrize("scale", data.SCALES)
def test_real_snapshots_load(scale):
    for at in REAL:
        if not data.joint_path(at, scale).exists():
            continue
        snap = data.load_snapshot(at, scale)
        assert snap.sims == 20000
        # Analytic vs simulated: 7 per conference in every sim, so q sums to exactly 7.
        for conf in ("AFC", "NFC"):
            assert abs(snap.q[snap.conference == conf].sum() - 7) < 1e-9
