"""Finding and reading NWB files from the DANDI archive.

The files in dandiset 000718 are 2 to 5 GB each, because every session file carries
the raw one photon movie next to the processed traces. We never download them.
Instead we read them over HTTP with byte range requests (remfile), pull out the few
megabytes we actually need, and cache that locally.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import remfile
import yaml

DANDISET = "000718"
VERSION = "0.260825.1902"
API = "https://api.dandiarchive.org/api"


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


@dataclass
class Asset:
    """One file in the archive, with everything we need to open it remotely."""

    path: str
    asset_id: str
    size: int
    subject: str
    session: str
    url: str

    @property
    def size_gb(self) -> float:
        return self.size / 1e9


def list_assets(dandiset: str = DANDISET, version: str = VERSION) -> list[Asset]:
    """Every NWB file in the dandiset, with a direct S3 url for range reads.

    We use the S3 blob url rather than the DANDI download url because the latter is a
    redirect, and redirects and range requests together are a common source of trouble.
    """
    listing = _get_json(f"{API}/dandisets/{dandiset}/versions/{version}/assets/?page_size=200")
    assets = []
    for record in listing["results"]:
        if not record["path"].endswith(".nwb"):
            continue
        info = _get_json(f"{API}/assets/{record['asset_id']}/info/")["metadata"]
        s3_urls = [u for u in info["contentUrl"] if "s3.amazonaws.com" in u]
        subject, filename = record["path"].split("/")
        session = filename.split("_ses-")[1].split("_")[0].replace(".nwb", "")
        assets.append(
            Asset(
                path=record["path"],
                asset_id=record["asset_id"],
                size=record["size"],
                subject=subject.replace("sub-", ""),
                session=session,
                url=s3_urls[0],
            )
        )
    return sorted(assets, key=lambda a: (a.subject, a.session))


def find_asset(assets: list[Asset], subject: str, session: str) -> Asset:
    """The one asset for this subject and session, or a clear error."""
    hits = [a for a in assets if a.subject == subject and a.session == session]
    if len(hits) != 1:
        available = sorted(a.session for a in assets if a.subject == subject)
        raise KeyError(f"no single asset for {subject} {session}; {subject} has {available}")
    return hits[0]


def open_remote_h5(url: str) -> tuple[h5py.File, remfile.File]:
    """Open an NWB file over the network as a plain HDF5 file.

    Returns both handles. Keep a reference to the remfile object for the lifetime of
    the h5py file, and close the h5py file when done.
    """
    remote = remfile.File(url)
    return h5py.File(remote, "r"), remote


def assets_table(assets: list[Asset]) -> pd.DataFrame:
    """The asset list as a table, for the Phase 0 report."""
    return pd.DataFrame(
        [
            {
                "subject": a.subject,
                "session": a.session,
                "size_gb": round(a.size_gb, 2),
                "path": a.path,
                "asset_id": a.asset_id,
            }
            for a in assets
        ]
    )


# ---------------------------------------------------------------------------
# Phase 1: pull the few megabytes we need out of a remote file and cache them.
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
CACHE_VERSION = 1  # bump this when the cache layout changes, so stale files are rebuilt


def load_config(path: Path | str | None = None) -> dict:
    """Every setting for the study, read from config.yaml."""
    path = Path(path) if path else REPO / "config.yaml"
    with open(path) as handle:
        return yaml.safe_load(handle)


@dataclass
class Session:
    """One recording session, with everything the later phases need.

    Activity matrices are cells by time, which is the convention used throughout the
    study, even though NWB stores them time by cells.
    """

    subject: str
    session: str
    context: str | None          # neutral, shock, novel, or None for homecage rest
    day: int | None              # day 1 is the neutral exposure for that mouse
    start_time: str
    rate_hz: float
    deconvolved: np.ndarray      # cells by time, spike-like event amplitudes
    denoised: np.ndarray         # cells by time, the smoothed fluorescence
    timestamps: np.ndarray       # seconds from the start of the session
    centroids: np.ndarray        # cells by 2, in pixels
    footprint_roi: np.ndarray    # the four footprint arrays are one sparse image stack
    footprint_row: np.ndarray
    footprint_col: np.ndarray
    footprint_weight: np.ndarray
    max_projection: np.ndarray
    freezing_start: np.ndarray
    freezing_stop: np.ndarray
    motion: np.ndarray
    motion_rate_hz: float
    shock_times: np.ndarray

    @property
    def n_cells(self) -> int:
        return self.deconvolved.shape[0]

    @property
    def n_frames(self) -> int:
        return self.deconvolved.shape[1]

    @property
    def duration_s(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0])

    def footprints_dense(self) -> np.ndarray:
        """Rebuild the full cells by height by width stack. Only for small checks."""
        height, width = self.max_projection.shape
        stack = np.zeros((self.n_cells, height, width), dtype=np.float32)
        stack[self.footprint_roi, self.footprint_row, self.footprint_col] = self.footprint_weight
        return stack


def _read_footprints(dataset, block: int = 100) -> tuple[np.ndarray, ...]:
    """Read ROI masks as a sparse list of pixels.

    The dense stack is over 2 GB in memory but only a few MB on disk, because each cell
    covers a couple of hundred pixels. We read it a slab at a time and keep the nonzero
    pixels only.
    """
    n_rois = dataset.shape[0]
    rois, rows, cols, weights = [], [], [], []
    for start in range(0, n_rois, block):
        slab = dataset[start:start + block]
        roi, row, col = np.nonzero(slab)
        rois.append(roi.astype(np.int32) + start)
        rows.append(row.astype(np.int16))
        cols.append(col.astype(np.int16))
        weights.append(slab[roi, row, col].astype(np.float32))
    return (np.concatenate(rois), np.concatenate(rows),
            np.concatenate(cols), np.concatenate(weights))


def _optional(h5file, path, dtype=np.float64) -> np.ndarray:
    """Read a dataset that some sessions do not have, for example shock times."""
    if path not in h5file:
        return np.array([], dtype=dtype)
    return np.asarray(h5file[path][:], dtype=dtype)


def _cache_path(config: dict, subject: str, session: str) -> Path:
    directory = REPO / config["dataset"]["cache_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{subject}_{session}.npz"


def fetch_session(asset: Asset, config: dict) -> dict:
    """Download just the arrays we need from one remote file."""
    paths = config["nwb_paths"]
    h5file, _remote = open_remote_h5(asset.url)

    # NWB stores traces as time by cells; the study works in cells by time.
    deconvolved = np.asarray(h5file[paths["deconvolved"]][:], dtype=np.float32).T
    denoised = np.asarray(h5file[paths["denoised"]][:], dtype=np.float32).T
    timestamps = np.asarray(h5file[paths["timestamps"]][:], dtype=np.float64)
    roi, row, col, weight = _read_footprints(h5file[paths["masks"]])

    motion_path = paths["motion"]
    motion_rate = 0.0
    if motion_path in h5file:
        motion = np.asarray(h5file[motion_path][:], dtype=np.float32)
        motion_rate = float(h5file[motion_path].parent["starting_time"].attrs["rate"])
    else:
        motion = np.array([], dtype=np.float32)

    arrays = {
        "deconvolved": deconvolved,
        "denoised": denoised,
        "timestamps": timestamps,
        "centroids": np.asarray(h5file[paths["centroids"]][:], dtype=np.int32),
        "footprint_roi": roi,
        "footprint_row": row,
        "footprint_col": col,
        "footprint_weight": weight,
        "max_projection": np.asarray(h5file[paths["max_projection"]][:], dtype=np.float32),
        "freezing_start": _optional(h5file, paths["freezing_start"]),
        "freezing_stop": _optional(h5file, paths["freezing_stop"]),
        "motion": motion,
        "shock_times": _optional(h5file, paths["shock_start"]),
    }
    start_time = h5file["session_start_time"][()]
    h5file.close()

    meta = {
        "cache_version": CACHE_VERSION,
        "subject": asset.subject,
        "session": asset.session,
        "start_time": start_time.decode() if isinstance(start_time, bytes) else str(start_time),
        "rate_hz": float(1.0 / np.median(np.diff(timestamps))),
        "motion_rate_hz": motion_rate,
        "context": config["contexts"].get(asset.session),
    }
    return {"arrays": arrays, "meta": meta}


def load_session(subject: str, session: str, config: dict | None = None,
                 assets: list[Asset] | None = None, refresh: bool = False) -> Session:
    """One session, from the local cache if possible and from the archive if not."""
    config = config or load_config()
    path = _cache_path(config, subject, session)
    meta_path = path.with_suffix(".json")

    if refresh or not path.exists() or not meta_path.exists():
        assets = assets if assets is not None else list_assets(
            config["dataset"]["dandiset"], config["dataset"]["version"])
        fetched = fetch_session(find_asset(assets, subject, session), config)
        np.savez_compressed(path, **fetched["arrays"])
        meta_path.write_text(json.dumps(fetched["meta"], indent=1))

    meta = json.loads(meta_path.read_text())
    if meta.get("cache_version") != CACHE_VERSION:
        return load_session(subject, session, config, assets, refresh=True)

    with np.load(path) as stored:
        arrays = {name: stored[name] for name in stored.files}
    return Session(
        subject=meta["subject"], session=meta["session"], context=meta["context"],
        day=None, start_time=meta["start_time"], rate_hz=meta["rate_hz"],
        motion_rate_hz=meta["motion_rate_hz"], **arrays)


def assign_days(sessions: list[Session]) -> list[Session]:
    """Number the days of one mouse, counting the neutral exposure as day 1.

    The mice were recorded a year apart, so days only make sense within an animal.
    """
    dates = {s.session: datetime.fromisoformat(s.start_time).date() for s in sessions}
    first = dates["NeutralExposure"]
    for session in sessions:
        session.day = (dates[session.session] - first).days + 1
    return sessions


def load_subject(subject: str, config: dict | None = None, include_offline: bool = True,
                 assets: list[Asset] | None = None) -> list[Session]:
    """Every session of one mouse, with day numbers filled in."""
    config = config or load_config()
    spec = config["subjects"][subject]
    names = list(spec["behavioural_sessions"])
    if include_offline:
        names += list(spec["offline_sessions"])
    sessions = [load_session(subject, name, config, assets) for name in names]
    return assign_days(sessions)
