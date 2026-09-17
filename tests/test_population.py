"""Tests for population vectors, binning and the similarity measures."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cells  # noqa: E402
import load  # noqa: E402
import population  # noqa: E402

CONFIG = load.load_config()
CACHED = sorted((REPO / CONFIG["dataset"]["cache_dir"]).glob("*.npz"))
needs_cache = pytest.mark.skipif(not CACHED, reason="cache is empty; run scripts/01_describe.py")


@needs_cache
def test_binning_keeps_all_the_activity():
    """Binning moves activity around in time but must not create or destroy any."""
    session = load.load_session("Ca-EEG3-4", "Recall1", CONFIG)
    rois = np.arange(session.n_cells)
    window = cells.analysis_window(session, CONFIG)
    binned = population.binned_activity(session, CONFIG, rois)
    assert binned.sum() == pytest.approx(session.deconvolved[:, window].sum(), rel=1e-5)
    assert binned.shape[0] == session.n_cells


@needs_cache
def test_bin_count_follows_the_bin_width():
    session = load.load_session("Ca-EEG3-4", "Recall1", CONFIG)
    rois = np.arange(session.n_cells)
    narrow = population.binned_activity(session, CONFIG, rois)
    wide = {**CONFIG, "analysis": {**CONFIG["analysis"], "bin_seconds": 10}}
    assert narrow.shape[1] == pytest.approx(
        10 * population.binned_activity(session, wide, rois).shape[1], rel=0.05)


def test_the_measure_separates_a_shared_pattern_from_unrelated_bins():
    """bin_similarity is the median over every pair of bins, not a bin against itself.

    Independent bins therefore sit near zero even when a session is compared with itself,
    which is exactly why the within session value is reported as the ceiling: it says how
    much repeatable pattern there is to find in the first place.
    """
    rng = np.random.default_rng(0)
    unrelated = rng.random((40, 25))
    assert population.within_session_bin_similarity(unrelated) == pytest.approx(0.0, abs=0.15)

    shared = rng.random((40, 1)) + 0.1 * rng.random((40, 25))  # every bin the same pattern
    assert population.within_session_bin_similarity(shared) > 0.8
    assert population.bin_similarity(shared, shared) > 0.8


def test_empty_bins_do_not_break_the_correlation():
    """A bin with no activity has no pattern, so it must not produce a division by zero."""
    binned = np.zeros((10, 4))
    binned[:, 0] = np.arange(10)
    assert np.isfinite(population.bin_similarity(binned, binned))


def test_shuffling_cell_identity_destroys_the_similarity():
    rng = np.random.default_rng(CONFIG["analysis"]["seed"])
    first = rng.random(200)
    second = first + 0.3 * rng.random(200)  # genuinely related
    binned_a, binned_b = rng.random((200, 20)), rng.random((200, 20))
    result = population.compare(first, second, binned_a, binned_b, rng, 200)
    assert result["pearson"] > 0.8
    assert abs(result["pearson_shuffled"]) < 0.05
    assert result["pearson_ci_low"] < result["pearson"] < result["pearson_ci_high"]
