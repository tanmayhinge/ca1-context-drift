"""Tests for cross-day cell matching.

The important one builds a fake second session out of a real one by sliding the field
of view and shuffling the cell numbering. We know the right answer by construction, so
the matcher either recovers it or it does not.
"""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import load  # noqa: E402
import match  # noqa: E402

CONFIG = load.load_config()
CACHED = sorted((REPO / CONFIG["dataset"]["cache_dir"]).glob("*.npz"))
needs_cache = pytest.mark.skipif(not CACHED, reason="cache is empty; run scripts/01_describe.py")
SHIFT = (3, -4)


def a_real_session() -> load.Session:
    subject, session = CACHED[0].stem.split("_", 1)
    return load.load_session(subject, session, CONFIG)


def shifted_and_shuffled(session: load.Session, shift, order) -> load.Session:
    """The same session seen through a field of view that slid, with cells renumbered."""
    position = np.empty(len(order), dtype=int)
    position[order] = np.arange(len(order))  # where each original cell ends up
    return dataclasses.replace(
        session,
        session=session.session + "-fake",
        max_projection=np.roll(session.max_projection, shift, axis=(0, 1)),
        footprint_roi=position[session.footprint_roi],
        footprint_row=session.footprint_row + shift[0],
        footprint_col=session.footprint_col + shift[1],
        centroids=session.centroids[order] + np.array(shift),
    )


@needs_cache
def test_shift_is_estimated_exactly():
    session = a_real_session()
    fake = shifted_and_shuffled(session, SHIFT, np.arange(session.n_cells))
    assert match.estimate_shift(session.max_projection, fake.max_projection) == SHIFT


@needs_cache
def test_matcher_recovers_a_known_answer():
    session = a_real_session()
    rng = np.random.default_rng(CONFIG["analysis"]["seed"])
    order = rng.permutation(session.n_cells)
    position = np.empty(session.n_cells, dtype=int)
    position[order] = np.arange(session.n_cells)

    pairs = match.match_pair(session, shifted_and_shuffled(session, SHIFT, order), CONFIG)

    # cells whose footprint slid off the edge cannot be matched, so allow a few losses
    assert len(pairs) > 0.95 * session.n_cells
    expected = position[pairs["roi_reference"].to_numpy()]
    assert np.array_equal(pairs["roi_other"].to_numpy(), expected)


@needs_cache
def test_a_session_matches_itself_one_to_one():
    session = a_real_session()
    pairs = match.match_pair(session, session, CONFIG)
    assert len(pairs) == session.n_cells
    assert np.array_equal(pairs["roi_reference"].to_numpy(), pairs["roi_other"].to_numpy())


def test_consensus_keeps_only_cells_the_runs_agree_on():
    """Pure bookkeeping, so no data needed."""
    registration = pd.DataFrame({
        "A": [1, 1, 1, 2, 3],
        "B": [7, 7, 7, 8, match.MISSING],   # cell (1, 7) seen three times, (2, 8) once
    })
    kept = match.consensus_matches(registration, ["A", "B"], min_tables=3)
    assert kept[["A", "B"]].to_numpy().tolist() == [[1, 7]]
