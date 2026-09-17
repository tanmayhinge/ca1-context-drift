"""Which cells were active in a session, and how much those groups overlap.

An ensemble here is the most active quarter of the cells in a session. The size of that
fraction is a choice, set in config.yaml, and scripts/03_overlap.py repeats everything at
10, 25 and 50 percent so the reader can see whether the choice drives the answer.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import hypergeom

import load


def analysis_window(session: load.Session, config: dict,
                    seconds: float | None = None) -> np.ndarray:
    """The stretch of a session that every analysis uses, as a mask over frames.

    Sessions differ in length, so we take the same amount of time from each. Fear
    conditioning is the exception: it contains three foot shocks, and activity after a
    shock reflects the shock rather than the place, so only the quiet period before the
    first shock is used.
    """
    settings = config["analysis"]
    elapsed = session.timestamps - session.timestamps[0]
    limit = settings["window_seconds"] if seconds is None else seconds
    if len(session.shock_times):
        limit = min(limit, settings["fc_baseline_seconds"], float(session.shock_times.min()))
    return elapsed < limit


def event_rate(session: load.Session, config: dict, rois: np.ndarray | None = None,
               seconds: float | None = None, keep: np.ndarray | None = None) -> np.ndarray:
    """Activity events per second for each cell, inside the analysis window.

    The deconvolved trace is zero except where the algorithm placed an event, so counting
    nonzero frames counts events. A rate is used rather than a mean amplitude because
    amplitude depends on how bright a cell happens to be, which differs between mice.
    """
    window = analysis_window(session, config, seconds)
    if keep is not None:
        window = window & keep  # for example, drop the frames where the mouse was frozen
    activity = session.deconvolved[:, window]
    if rois is not None:
        activity = activity[rois]
    seconds = float(np.sum(window)) / session.rate_hz
    return (activity > 0).sum(axis=1) / seconds


def ensemble(rates: np.ndarray, percent: float) -> np.ndarray:
    """The most active `percent` of cells, as a boolean mask."""
    cutoff = np.percentile(rates, 100 - percent)
    return rates >= cutoff


def overlap_stats(first: np.ndarray, second: np.ndarray, rng: np.random.Generator,
                  n_shuffles: int) -> dict:
    """How much two ensembles share, against what sharing chance alone would give.

    Chance here means: two groups of the same sizes drawn at random from the same cells.
    That is the hypergeometric distribution, so the expected overlap and the probability
    of seeing at least this much are exact. The shuffle is run as well, as a check that
    the exact calculation is being applied correctly.
    """
    n_cells = len(first)
    size_a, size_b = int(first.sum()), int(second.sum())
    observed = int((first & second).sum())
    expected = size_a * size_b / n_cells

    distribution = hypergeom(n_cells, size_a, size_b)
    p_exact = float(distribution.sf(observed - 1))

    draws = np.array([rng.permutation(first) & second for _ in range(n_shuffles)]).sum(axis=1)
    p_shuffle = float((draws >= observed).mean())

    # How precise is the ratio itself? Resample the cells with replacement. This is what
    # tells us whether two session pairs really differ or just look different.
    ratios = []
    for _ in range(n_shuffles):
        pick = rng.integers(0, n_cells, n_cells)
        a, b = first[pick], second[pick]
        chance = a.sum() * b.sum() / n_cells
        if chance:
            ratios.append((a & b).sum() / chance)
    low, high = np.percentile(ratios, [2.5, 97.5]) if ratios else (np.nan, np.nan)

    union = size_a + size_b - observed
    return {
        "cells": n_cells,
        "active_a": size_a,
        "active_b": size_b,
        "observed": observed,
        "expected": expected,
        "ratio_to_chance": observed / expected if expected else float("nan"),
        "jaccard": observed / union if union else float("nan"),
        "jaccard_chance": expected / (size_a + size_b - expected) if expected else float("nan"),
        "ratio_ci_low": float(low),
        "ratio_ci_high": float(high),
        "p_exact": p_exact,
        "p_shuffle": p_shuffle,
    }
