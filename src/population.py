"""Comparing the whole population pattern between sessions.

Three views of the same question, because any single one can mislead:

1. Pearson correlation between the sessions' average activity per cell. Simple, but a
   handful of very busy cells can carry it.
2. Spearman correlation of the same vectors, which uses only the ranking of cells and so
   cannot be dominated by a few outliers.
3. Correlation between short time bins, which asks whether the moment to moment pattern
   looks alike rather than only the session average.

The baseline for all three is the same measurement after shuffling which cell is which,
which is what the numbers would look like if cross-day matching told us nothing.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr

import cells
import load


def population_vector(session: load.Session, config: dict, rois: np.ndarray,
                      seconds: float | None = None) -> np.ndarray:
    """One number per matched cell: its activity events per second in this session."""
    return cells.event_rate(session, config, rois, seconds)


def binned_activity(session: load.Session, config: dict, rois: np.ndarray,
                    seconds: float | None = None) -> np.ndarray:
    """Matched cells by time bins, summing deconvolved activity inside each bin.

    Neighbouring imaging frames are nearly identical, so the raw 15 Hz frames are not
    independent samples. One second bins are coarse enough to be treated as separate
    moments and fine enough to keep the structure of the session.
    """
    window = cells.analysis_window(session, config, seconds)
    activity = session.deconvolved[rois][:, window]
    elapsed = session.timestamps[window] - session.timestamps[window][0]
    edges = np.arange(0, elapsed[-1] + config["analysis"]["bin_seconds"],
                      config["analysis"]["bin_seconds"])
    which = np.clip(np.digitize(elapsed, edges) - 1, 0, len(edges) - 2)
    binned = np.zeros((activity.shape[0], len(edges) - 1))
    np.add.at(binned.T, which, activity.T)
    return binned


def _zscore_columns(matrix: np.ndarray) -> np.ndarray:
    """Standardise each time bin across cells, so bins can be compared by dot product."""
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    spread = centred.std(axis=0, keepdims=True)
    spread[spread == 0] = np.inf  # a bin with no activity correlates with nothing
    return centred / spread


def bin_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Median correlation between every time bin of one session and every bin of the other."""
    correlations = _zscore_columns(first).T @ _zscore_columns(second) / first.shape[0]
    return float(np.median(correlations))


def within_session_bin_similarity(binned: np.ndarray) -> float:
    """The same measure inside one session, which is the ceiling for that session."""
    standardised = _zscore_columns(binned)
    correlations = standardised.T @ standardised / binned.shape[0]
    upper = np.triu_indices(correlations.shape[0], k=1)
    return float(np.median(correlations[upper]))


def compare(vector_a: np.ndarray, vector_b: np.ndarray, binned_a: np.ndarray,
            binned_b: np.ndarray, rng: np.random.Generator, n_shuffles: int) -> dict:
    """All three similarity measures, with a shuffled baseline and an interval over cells."""
    pearson = float(pearsonr(vector_a, vector_b).statistic)
    spearman = float(spearmanr(vector_a, vector_b).statistic)
    bins = bin_similarity(binned_a, binned_b)

    shuffled_pearson, shuffled_bins = [], []
    for _ in range(n_shuffles):
        order = rng.permutation(len(vector_a))
        shuffled_pearson.append(pearsonr(vector_a, vector_b[order]).statistic)
    for _ in range(min(n_shuffles, 100)):  # the bin measure is heavier, so fewer draws
        order = rng.permutation(len(vector_a))
        shuffled_bins.append(bin_similarity(binned_a, binned_b[order]))

    bootstrap = []
    n_cells = len(vector_a)
    for _ in range(n_shuffles):
        pick = rng.integers(0, n_cells, n_cells)
        if np.std(vector_a[pick]) and np.std(vector_b[pick]):
            bootstrap.append(pearsonr(vector_a[pick], vector_b[pick]).statistic)

    return {
        "cells": n_cells,
        "pearson": pearson,
        "spearman": spearman,
        "bin_similarity": bins,
        "pearson_shuffled": float(np.mean(shuffled_pearson)),
        "pearson_shuffled_p95": float(np.percentile(np.abs(shuffled_pearson), 95)),
        "bin_shuffled": float(np.mean(shuffled_bins)),
        "pearson_ci_low": float(np.percentile(bootstrap, 2.5)),
        "pearson_ci_high": float(np.percentile(bootstrap, 97.5)),
    }


# ---------------------------------------------------------------------------
# A low dimensional view
# ---------------------------------------------------------------------------

def stack_sessions(binned: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Put every session's time bins into one matrix of bins by cells, with labels.

    Each cell is standardised using its mean and spread over all the sessions together,
    not session by session. Standardising within a session would subtract exactly the
    between session differences we are looking for.
    """
    labels = np.concatenate([[name] * matrix.shape[1] for name, matrix in binned.items()])
    stacked = np.hstack(list(binned.values())).T  # bins by cells
    centred = stacked - stacked.mean(axis=0, keepdims=True)
    spread = centred.std(axis=0, keepdims=True)
    spread[spread == 0] = np.inf
    return centred / spread, labels


def principal_components(stacked: np.ndarray, n_components: int | None = None):
    """Principal components of the bins by cells matrix, with variance explained.

    By default every component is kept, so that "how many components hold half the
    variance" is a real answer rather than a number capped by what we asked for.
    """
    from sklearn.decomposition import PCA

    model = PCA(n_components=min(n_components or min(stacked.shape), *stacked.shape))
    coordinates = model.fit_transform(stacked)
    return coordinates, model.explained_variance_ratio_


def separation_under_shuffled_labels(stacked: np.ndarray, labels: np.ndarray, first: str,
                                     second: str, rng: np.random.Generator,
                                     n_shuffles: int) -> float:
    """The same measure after shuffling which bin belongs to which session.

    There are far fewer time bins than cells, so a direction fitted to the data can pull
    almost any two groups apart. This says how large that effect alone is, and therefore
    whether the real separation means anything.
    """
    subset = np.isin(labels, [first, second])
    values = stacked[subset]
    shuffled_labels = labels[subset].copy()
    scores = []
    for _ in range(n_shuffles):
        rng.shuffle(shuffled_labels)
        scores.append(separation(values, shuffled_labels, first, second))
    return float(np.mean(scores))


def separation(stacked: np.ndarray, labels: np.ndarray, first: str, second: str) -> float:
    """How far apart two sessions sit, in units of their own scatter.

    The bins of each session are projected onto the straight line joining the two session
    means, and the standardised difference along that line is returned. The line is chosen
    using the same data it is measured on, so this is an optimistic number: it is a way of
    reading the picture, not a test. The cross-validated test is the decoder in Phase 6.
    """
    a = stacked[labels == first]
    b = stacked[labels == second]
    direction = a.mean(axis=0) - b.mean(axis=0)
    length = np.linalg.norm(direction)
    if length == 0:
        return 0.0
    direction = direction / length
    projected_a, projected_b = a @ direction, b @ direction
    pooled = np.sqrt(0.5 * (projected_a.var(ddof=1) + projected_b.var(ddof=1)))
    return float((projected_a.mean() - projected_b.mean()) / pooled) if pooled else 0.0
