"""Tests for the activity window, the ensemble rule and the overlap statistics."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cells  # noqa: E402
import load  # noqa: E402

CONFIG = load.load_config()
CACHED = sorted((REPO / CONFIG["dataset"]["cache_dir"]).glob("*.npz"))
needs_cache = pytest.mark.skipif(not CACHED, reason="cache is empty; run scripts/01_describe.py")


@needs_cache
def test_conditioning_window_stops_before_the_first_shock():
    session = load.load_session("Ca-EEG3-4", "FC", CONFIG)
    window = cells.analysis_window(session, CONFIG)
    elapsed = session.timestamps - session.timestamps[0]
    assert elapsed[window].max() < session.shock_times.min()


@needs_cache
def test_other_sessions_use_the_configured_window():
    session = load.load_session("Ca-EEG3-4", "Recall1", CONFIG)
    elapsed = session.timestamps - session.timestamps[0]
    window = cells.analysis_window(session, CONFIG)
    assert elapsed[window].max() <= CONFIG["analysis"]["window_seconds"]
    assert window.sum() > 0.9 * len(window)  # the session is only just longer than the window


def test_ensemble_takes_the_requested_share_of_cells():
    rates = np.arange(200, dtype=float)
    assert cells.ensemble(rates, 25).sum() == 50
    assert cells.ensemble(rates, 10).sum() == 20
    assert cells.ensemble(rates, 25)[-1] and not cells.ensemble(rates, 25)[0]


def test_identical_ensembles_overlap_completely():
    group = np.zeros(100, dtype=bool)
    group[:25] = True
    rng = np.random.default_rng(0)
    stats = cells.overlap_stats(group, group, rng, 200)
    assert stats["observed"] == 25
    assert stats["jaccard"] == 1.0
    assert stats["ratio_to_chance"] == pytest.approx(100 / 25)


def test_independent_ensembles_sit_at_chance():
    """Two groups picked at random should give a ratio near one and a large p value."""
    rng = np.random.default_rng(CONFIG["analysis"]["seed"])
    ratios, p_values = [], []
    for _ in range(50):
        first = rng.permutation(np.arange(400) < 100)
        second = rng.permutation(np.arange(400) < 100)
        stats = cells.overlap_stats(first, second, rng, 200)
        ratios.append(stats["ratio_to_chance"])
        p_values.append(stats["p_exact"])
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.05)
    assert 0.3 < np.mean(p_values) < 0.7  # p values are uniform when the null is true


def test_exact_and_shuffled_p_values_agree():
    rng = np.random.default_rng(1)
    first = np.zeros(200, dtype=bool)
    second = np.zeros(200, dtype=bool)
    first[:50] = True
    second[25:75] = True  # half of them shared, well above chance
    stats = cells.overlap_stats(first, second, rng, 2000)
    assert stats["p_exact"] < 0.001
    assert stats["p_shuffle"] < 0.01
