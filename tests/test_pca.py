"""Tests for the low dimensional view."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import load  # noqa: E402
import population  # noqa: E402

CONFIG = load.load_config()


def test_stacking_standardises_cells_across_all_sessions_together():
    """Each cell should end up with mean zero over the whole stack, not per session."""
    binned = {"a": np.array([[1.0, 2, 3], [0, 0, 0]]),
              "b": np.array([[7.0, 8, 9], [1, 2, 3]])}
    stacked, labels = population.stack_sessions(binned)
    assert stacked.shape == (6, 2)
    assert list(labels) == ["a"] * 3 + ["b"] * 3
    assert stacked[:, 0].mean() == pytest.approx(0.0, abs=1e-12)
    assert stacked[labels == "a", 0].mean() < 0  # session a really is below the joint mean


def test_a_silent_cell_does_not_produce_infinities():
    binned = {"a": np.array([[1.0, 2], [5, 5]]), "b": np.array([[3.0, 4], [5, 5]])}
    stacked, _ = population.stack_sessions(binned)
    assert np.all(np.isfinite(stacked))


def test_components_are_ordered_and_explain_everything():
    rng = np.random.default_rng(0)
    stacked = rng.random((50, 8))
    _, variance = population.principal_components(stacked)
    assert np.all(np.diff(variance) <= 1e-12)          # ordered, largest first
    assert variance.sum() == pytest.approx(1.0)        # nothing is thrown away


def test_separation_is_large_even_for_meaningless_groups_when_cells_outnumber_bins():
    """The trap that the shuffled control exists to expose.

    With fewer time bins than cells, a direction fitted to the data pulls any two random
    groups apart. If this ever stops being true the shuffled control can be dropped.
    """
    rng = np.random.default_rng(CONFIG["analysis"]["seed"])
    stacked = rng.standard_normal((30, 200))           # 30 bins, 200 cells, no structure
    labels = np.array(["a"] * 15 + ["b"] * 15)
    assert population.separation(stacked, labels, "a", "b") > 1.5
    chance = population.separation_under_shuffled_labels(stacked, labels, "a", "b", rng, 50)
    assert chance > 1.5


def test_real_structure_beats_the_shuffled_control():
    rng = np.random.default_rng(1)
    stacked = rng.standard_normal((60, 20))
    labels = np.array(["a"] * 30 + ["b"] * 30)
    stacked[labels == "a"] += 2.0                      # a genuine shift between the groups
    observed = population.separation(stacked, labels, "a", "b")
    chance = population.separation_under_shuffled_labels(stacked, labels, "a", "b", rng, 50)
    assert observed > 2 * chance
