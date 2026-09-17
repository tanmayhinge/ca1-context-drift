"""Phase 1: build the local cache and describe what we have.

Reads every session once over the network, caches it under data/cache/, and writes

    results/01_sessions.csv   one row per session
    figures/01_raster.png     activity of every cell in one session

Run:  .venv/bin/python scripts/01_describe.py
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # no display on this machine
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import figures  # noqa: E402
import load  # noqa: E402

RASTER_SUBJECT = "Ca-EEG3-4"
RASTER_SESSION = "NeutralExposure"


def describe(session: load.Session) -> dict:
    """One row of the summary table."""
    active_frames = (session.deconvolved > 0).mean(axis=1)
    freezing_s = float(np.sum(session.freezing_stop - session.freezing_start))
    return {
        "subject": session.subject,
        "session": session.session,
        "day": session.day,
        "context": session.context or "homecage",
        "date": session.start_time[:10],
        "cells": session.n_cells,
        "frames": session.n_frames,
        "rate_hz": round(session.rate_hz, 2),
        "minutes": round(session.duration_s / 60, 1),
        "median_event_rate_hz": round(float(np.median(active_frames)) * session.rate_hz, 3),
        "freezing_percent": round(100 * freezing_s / session.duration_s, 1) if freezing_s else None,
        "shocks": len(session.shock_times),
    }


def plot_raster(session: load.Session, path: Path) -> None:
    """A heatmap of every cell's activity over the session.

    Cells are ordered by when they were most active, which turns a shapeless block of
    dots into a readable diagonal and makes the ordering itself obvious as a choice.
    """
    activity = session.deconvolved
    peak_frame = np.argmax(activity, axis=1)
    order = np.argsort(peak_frame)
    minutes = (session.timestamps - session.timestamps[0]) / 60

    figure, axes = plt.subplots(2, 1, figsize=(9, 6), height_ratios=[4, 1], sharex=True)
    shown = activity[order] > 0  # deconvolved activity is 97 percent zeros, so show events
    axes[0].imshow(shown, aspect="auto", cmap="Greys", interpolation="nearest",
                   extent=[minutes[0], minutes[-1], len(order), 0])
    axes[0].set_ylabel("cell (ordered by time of peak)")
    axes[0].set_title(f"{session.subject}  {session.session}  "
                      f"({session.context}, day {session.day}, {session.n_cells} cells)")

    axes[1].plot(minutes, shown.sum(axis=0), lw=0.6, color="black")
    axes[1].set_ylabel("cells active")
    axes[1].set_xlabel("minutes from session start")
    for start, stop in zip(session.freezing_start, session.freezing_stop):
        axes[1].axvspan(start / 60, stop / 60, color="tab:blue", alpha=0.15, lw=0)
    axes[1].set_xlim(minutes[0], minutes[-1])

    figure.suptitle("shaded bands are freezing bouts", x=0.5, y=0.02, fontsize=8)
    figure.tight_layout()
    figures.save_figure(figure, path, load.load_config())
    plt.close(figure)


def main() -> None:
    config = load.load_config()
    assets = load.list_assets(config["dataset"]["dandiset"], config["dataset"]["version"])

    by_subject = {}
    for subject in config["subjects"]:
        print(f"loading {subject} ...")
        by_subject[subject] = load.load_subject(subject, config, assets=assets)

    rows = [describe(s) for sessions in by_subject.values() for s in sessions]
    table = pd.DataFrame(rows).sort_values(["subject", "day", "session"])
    (REPO / "results").mkdir(exist_ok=True)
    table.to_csv(REPO / "results" / "01_sessions.csv", index=False)
    print()
    print(table.to_string(index=False))

    (REPO / "figures").mkdir(exist_ok=True)
    raster = next(s for s in by_subject[RASTER_SUBJECT] if s.session == RASTER_SESSION)
    plot_raster(raster, REPO / "figures" / "01_raster.png")
    print(f"\nwrote figures/01_raster.png and results/01_sessions.csv")


if __name__ == "__main__":
    main()
