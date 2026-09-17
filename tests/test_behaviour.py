"""Tests for the freezing and motion measures."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import behaviour  # noqa: E402
import load  # noqa: E402

CONFIG = load.load_config()


def fake_session(freezing_start, freezing_stop, n_frames=300, rate=15.0,
                 deconvolved=None) -> load.Session:
    """A session with a known clock and known freezing bouts, and nothing else real."""
    timestamps = np.arange(n_frames) / rate
    if deconvolved is None:
        deconvolved = np.zeros((2, n_frames))
    return load.Session(
        subject="test", session="Recall1", context="shock", day=1,
        start_time="2022-09-20T09:00:00-04:00", rate_hz=rate,
        deconvolved=deconvolved, denoised=deconvolved, timestamps=timestamps,
        centroids=np.zeros((2, 2)), footprint_roi=np.zeros(1), footprint_row=np.zeros(1),
        footprint_col=np.zeros(1), footprint_weight=np.zeros(1),
        max_projection=np.zeros((2, 2)),
        freezing_start=np.array(freezing_start, dtype=float),
        freezing_stop=np.array(freezing_stop, dtype=float),
        motion=np.zeros(0), motion_rate_hz=30.0, shock_times=np.array([]))


def test_freezing_mask_marks_exactly_the_bouts():
    session = fake_session([1.0], [2.0], n_frames=60, rate=10.0)
    frozen = behaviour.freezing_mask(session)
    assert frozen[10:21].all()          # 1.0 s to 2.0 s at 10 Hz
    assert not frozen[:10].any()
    assert not frozen[21:].any()


def test_freezing_percent_is_a_share_of_the_window():
    session = fake_session([0.0], [30.0], n_frames=600, rate=10.0)  # 30 s of a 60 s session
    assert behaviour.freezing_percent(session, CONFIG) == pytest.approx(50, abs=1)


def test_a_session_with_no_bouts_reports_nothing_rather_than_zero():
    """No freezing file is different from no freezing, so it must not become 0 percent."""
    session = fake_session([], [])
    assert np.isnan(behaviour.freezing_percent(session, CONFIG))


def test_dropping_frozen_frames_removes_activity_that_only_happened_while_frozen():
    activity = np.zeros((2, 600))
    activity[0, 100:200] = 1.0   # cell 0 fires only between 10 s and 20 s
    activity[1, 400:500] = 1.0   # cell 1 fires only between 40 s and 50 s
    session = fake_session([10.0], [20.0], n_frames=600, rate=10.0, deconvolved=activity)

    rois = np.array([0, 1])
    moving = behaviour.moving_rates(session, CONFIG, rois)
    assert moving[0] == 0.0      # all of cell 0's activity was inside the bout
    assert moving[1] > 0.0
