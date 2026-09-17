"""Freezing and movement, and whether they explain what the neural measures show.

Freezing is complete immobility apart from breathing, and in a rodent it is the standard
sign of fear. It is scored from the behavioural video by ezTrack, and the dataset stores
both the freezing bouts and a continuous motion trace.

The motion trace is sampled at 30 Hz and begins within a few milliseconds of the imaging,
so its samples are placed at index divided by rate from the start of the session.
"""

from __future__ import annotations

import numpy as np

import cells
import load


def freezing_mask(session: load.Session) -> np.ndarray:
    """True for each imaging frame that falls inside a freezing bout."""
    elapsed = session.timestamps - session.timestamps[0]
    frozen = np.zeros(len(elapsed), dtype=bool)
    for start, stop in zip(session.freezing_start, session.freezing_stop):
        frozen |= (elapsed >= start) & (elapsed <= stop)
    return frozen


def freezing_percent(session: load.Session, config: dict) -> float:
    """Percent of the analysis window the animal spent frozen."""
    window = cells.analysis_window(session, config)
    if not len(session.freezing_start):
        return float("nan")
    return float(100 * (freezing_mask(session) & window).sum() / window.sum())


def mean_motion(session: load.Session, config: dict) -> float:
    """Average ezTrack motion over the analysis window.

    Motion is the number of pixels in the behavioural video whose brightness changed
    between frames, so it is a rough speed with arbitrary units, comparable within an
    animal but not between animals.
    """
    if not len(session.motion):
        return float("nan")
    window = cells.analysis_window(session, config)
    seconds = (session.timestamps[window] - session.timestamps[0])[[0, -1]]
    samples = np.arange(len(session.motion)) / session.motion_rate_hz
    inside = (samples >= seconds[0]) & (samples <= seconds[1])
    return float(session.motion[inside].mean())


def moving_rates(session: load.Session, config: dict, rois: np.ndarray) -> np.ndarray:
    """Activity rates computed only from the frames where the animal was not frozen."""
    return cells.event_rate(session, config, rois, keep=~freezing_mask(session))
