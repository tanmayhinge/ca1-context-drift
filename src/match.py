"""Following the same cell across days.

Two mice, two situations. Ca-EEG2-1 comes with CellReg tables produced by the original
authors, so its matching is given. Ca-EEG3-4 comes with none, so we match the cell
outlines ourselves. Because one mouse has both, the home made matcher can be scored
against CellReg before we trust it anywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

import load

MISSING = -9999  # CellReg's marker for "this cell was not found in that session"
REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# The registration tables that come with the dataset
# ---------------------------------------------------------------------------

def cellreg_column(subject: str, session: str) -> str:
    """Column name used inside the NWB tables, which spell the mouse with underscores."""
    return f"{subject.replace('-', '_', 1)}_{session}"


def load_cell_registration(subject: str, config: dict | None = None,
                           refresh: bool = False) -> pd.DataFrame:
    """Every CellReg table for one mouse, stacked into one long table.

    Columns are `table` (which registration run), `global_id` (the row within it) and
    one column per session holding that session's ROI index, or MISSING.
    """
    config = config or load.load_config()
    cache = REPO / config["dataset"]["cache_dir"] / f"{subject}_cellreg.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache)

    registration_file = config["subjects"][subject]["registration_file"]
    if registration_file is None:
        raise KeyError(f"{subject} has no registration file in the archive")

    assets = load.list_assets(config["dataset"]["dandiset"], config["dataset"]["version"])
    asset = load.find_asset(assets, subject, registration_file)
    h5file, _remote = load.open_remote_h5(asset.url)
    module = h5file[config["nwb_paths"]["cell_registration"]]

    frames = []
    for table_name in sorted(module.keys()):
        table = module[table_name]
        columns = {name: table[name][:] for name in table.keys() if name != "id"}
        frame = pd.DataFrame(columns)
        # strip the mouse prefix so columns are plain session names
        frame.columns = [name.split("_", 2)[-1] for name in frame.columns]
        frame.insert(0, "global_id", np.arange(len(frame)))
        frame.insert(0, "table", table_name.replace("vsConditioningSessions", ""))
        frames.append(frame)
    h5file.close()

    stacked = pd.concat(frames, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    stacked.to_csv(cache, index=False)
    return stacked


def consensus_matches(registration: pd.DataFrame, sessions: list[str],
                      min_tables: int) -> pd.DataFrame:
    """Cells that enough of the independent registration runs agree on.

    Each run registers one rest recording against the same behavioural sessions, so the
    runs are repeated measurements of the same thing. A cell that only one run in
    nineteen finds is not a cell we want to build a result on.
    """
    complete = registration[sessions].ne(MISSING).all(axis=1)
    found = registration.loc[complete, sessions]
    votes = found.groupby(sessions, sort=False).size().reset_index(name="tables")
    kept = votes[votes["tables"] >= min_tables].reset_index(drop=True)
    return kept.sort_values(sessions).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Matching cell outlines ourselves
# ---------------------------------------------------------------------------

def estimate_shift(reference: np.ndarray, image: np.ndarray) -> tuple[int, int]:
    """How far `image` has slid relative to `reference`, in whole pixels.

    Phase correlation: multiply the two images in the frequency domain, keep only the
    phase, and transform back. A rigid shift becomes a single bright peak. Returns the
    shift such that np.roll(image, -shift) lines up with the reference.
    """
    spectrum_a = np.fft.fft2(reference - reference.mean())
    spectrum_b = np.fft.fft2(image - image.mean())
    cross = spectrum_a * np.conj(spectrum_b)
    cross /= np.abs(cross) + 1e-12
    correlation = np.fft.ifft2(cross).real
    peak = np.unravel_index(np.argmax(correlation), correlation.shape)
    shift = tuple(int(p) if p <= n // 2 else int(p) - n for p, n in zip(peak, correlation.shape))
    return (-shift[0], -shift[1])


def roi_pixels(session: load.Session, shift: tuple[int, int] = (0, 0)) -> list[np.ndarray]:
    """Each cell as a sorted array of flat pixel indices, optionally shifted."""
    height, width = session.max_projection.shape
    rows = session.footprint_row.astype(np.int32) - shift[0]
    cols = session.footprint_col.astype(np.int32) - shift[1]
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    flat = rows[inside] * width + cols[inside]
    roi = session.footprint_roi[inside]
    order = np.argsort(roi, kind="stable")
    roi, flat = roi[order], flat[order]
    boundaries = np.searchsorted(roi, np.arange(session.n_cells + 1))
    return [np.sort(flat[boundaries[i]:boundaries[i + 1]]) for i in range(session.n_cells)]


def match_pair(reference: load.Session, other: load.Session, config: dict) -> pd.DataFrame:
    """Pair up the cells of two sessions by how much their outlines overlap.

    Candidates are limited to cells whose centres land close together once the field of
    view is aligned, then each candidate pair is scored by intersection over union, and
    the best one to one assignment is taken. One to one matters: without it, one bright
    cell can claim several partners.
    """
    settings = config["matching"]
    shift = estimate_shift(reference.max_projection, other.max_projection)

    pixels_reference = roi_pixels(reference)
    pixels_other = roi_pixels(other, shift)
    centres_reference = reference.centroids.astype(float)
    centres_other = other.centroids.astype(float) - np.array(shift, dtype=float)

    tree = cKDTree(centres_other)
    neighbours = tree.query_ball_point(centres_reference, r=settings["max_centroid_distance_px"])

    cost = np.ones((reference.n_cells, other.n_cells))  # 1 means no overlap at all
    overlap = np.zeros_like(cost)
    for i, candidates in enumerate(neighbours):
        a = pixels_reference[i]
        for j in candidates:
            b = pixels_other[j]
            shared = np.intersect1d(a, b, assume_unique=True).size
            if shared:
                union = a.size + b.size - shared
                overlap[i, j] = shared / union
                cost[i, j] = 1.0 - overlap[i, j]

    rows, columns = linear_sum_assignment(cost)
    scores = overlap[rows, columns]
    keep = scores >= settings["min_overlap"]
    return pd.DataFrame({
        "roi_reference": rows[keep],
        "roi_other": columns[keep],
        "overlap": scores[keep],
    }).sort_values("roi_reference").reset_index(drop=True)


def match_to_reference(sessions: list[load.Session], reference_name: str,
                       config: dict) -> pd.DataFrame:
    """Match every session against one reference session, usually day 1.

    Matching everything to a single reference rather than chaining day to day keeps
    errors from accumulating along the chain.
    """
    reference = next(s for s in sessions if s.session == reference_name)
    table = pd.DataFrame({reference_name: np.arange(reference.n_cells)})
    for session in sessions:
        if session.session == reference_name:
            continue
        pairs = match_pair(reference, session, config)
        mapping = np.full(reference.n_cells, MISSING)
        mapping[pairs["roi_reference"].to_numpy()] = pairs["roi_other"].to_numpy()
        table[session.session] = mapping
    return table


# ---------------------------------------------------------------------------
# Scoring the home made matcher against CellReg
# ---------------------------------------------------------------------------

def compare_to_cellreg(pairs: pd.DataFrame, registration: pd.DataFrame,
                       session_a: str, session_b: str, min_tables: int) -> dict:
    """Agreement between our pairs and the CellReg pairs for the same two sessions.

    Recall is the share of CellReg pairs we recover. Agreement is the share of our pairs
    that CellReg confirms, counted only over cells CellReg actually has an opinion about,
    since a cell CellReg never tracked is not evidence that we are wrong.
    """
    reference = consensus_matches(registration, [session_a, session_b], min_tables)
    truth = dict(zip(reference[session_a], reference[session_b]))
    ours = dict(zip(pairs["roi_reference"], pairs["roi_other"]))

    recovered = sum(1 for cell, partner in truth.items() if ours.get(cell) == partner)
    judged = {cell: partner for cell, partner in ours.items() if cell in truth}
    agreed = sum(1 for cell, partner in judged.items() if truth[cell] == partner)

    return {
        "session_a": session_a,
        "session_b": session_b,
        "cellreg_pairs": len(truth),
        "our_pairs": len(ours),
        "recovered": recovered,
        "recall": recovered / len(truth) if truth else float("nan"),
        "judged": len(judged),
        "agreement": agreed / len(judged) if judged else float("nan"),
    }


# ---------------------------------------------------------------------------
# Using the matched table in later phases
# ---------------------------------------------------------------------------

def load_matched(subject: str) -> pd.DataFrame:
    """The global cell by session table written by scripts/02_match.py."""
    path = REPO / "results" / f"02_matched_{subject}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run scripts/02_match.py first")
    return pd.read_csv(path)


def common_rois(table: pd.DataFrame, sessions: list[str]) -> dict[str, np.ndarray]:
    """ROI indices of the cells found in every one of these sessions.

    The arrays are aligned: position k in each array is the same cell. Later phases
    index each session's activity matrix with its own array and then compare rows.
    """
    keep = (table[sessions] != MISSING).all(axis=1)
    return {name: table.loc[keep, name].to_numpy() for name in sessions}
