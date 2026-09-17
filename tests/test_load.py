"""Tests for the streaming loader and the cache.

These run against whatever is already cached in data/cache/, so they are fast and
work offline. Build the cache first with scripts/01_describe.py.
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import load  # noqa: E402

CONFIG = load.load_config()
CACHED = sorted((REPO / CONFIG["dataset"]["cache_dir"]).glob("*.npz"))
needs_cache = pytest.mark.skipif(not CACHED, reason="cache is empty; run scripts/01_describe.py")


def cached_sessions():
    """Every session that is already on disk, loaded from the cache."""
    for path in CACHED:
        subject, session = path.stem.split("_", 1)
        yield load.load_session(subject, session, CONFIG)


@needs_cache
def test_shapes_agree_across_trace_types():
    for session in cached_sessions():
        assert session.deconvolved.shape == session.denoised.shape
        assert session.deconvolved.shape[1] == len(session.timestamps)
        assert session.centroids.shape == (session.n_cells, 2)


@needs_cache
def test_timestamps_increase_and_rate_is_plausible():
    for session in cached_sessions():
        assert np.all(np.diff(session.timestamps) > 0)
        assert 10 < session.rate_hz < 20  # Minian keeps every second frame of a 30 Hz movie


@needs_cache
def test_footprints_match_the_cells_and_the_field_of_view():
    for session in cached_sessions():
        height, width = session.max_projection.shape
        assert session.footprint_roi.max() < session.n_cells
        assert session.footprint_row.max() < height
        assert session.footprint_col.max() < width


@needs_cache
def test_every_behavioural_session_has_a_context():
    for session in cached_sessions():
        is_offline = "Offline" in session.session
        assert (session.context is None) == is_offline


def test_days_are_counted_from_the_neutral_exposure():
    """Day numbering is pure arithmetic on dates, so it needs no data."""
    def fake(session, day_of_month):
        return load.Session(
            subject="x", session=session, context=None, day=None,
            start_time=f"2022-09-{day_of_month:02d}T09:00:00-04:00", rate_hz=15.0,
            deconvolved=np.zeros((1, 1)), denoised=np.zeros((1, 1)),
            timestamps=np.zeros(1), centroids=np.zeros((1, 2)),
            footprint_roi=np.zeros(1), footprint_row=np.zeros(1),
            footprint_col=np.zeros(1), footprint_weight=np.zeros(1),
            max_projection=np.zeros((2, 2)), freezing_start=np.zeros(1),
            freezing_stop=np.zeros(1), motion=np.zeros(1), motion_rate_hz=30.0,
            shock_times=np.zeros(1))

    sessions = [fake("NeutralExposure", 17), fake("FC", 19), fake("Recall3", 22)]
    load.assign_days(sessions)
    assert [s.day for s in sessions] == [1, 3, 6]
