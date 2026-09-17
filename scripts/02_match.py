"""Phase 2: follow the same cells across days, and check that we can.

Writes
    results/02_matched_<subject>.csv   global cell by session table of ROI indices
    results/02_validation.csv          our matcher scored against CellReg
    results/02_summary.json            the numbers quoted in the report
    figures/02_matching.png            four panels, described in the report

Run:  .venv/bin/python scripts/02_match.py
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import figures  # noqa: E402
import load  # noqa: E402
import match  # noqa: E402

REFERENCE = "NeutralExposure"  # day 1 for both mice
VALIDATION_SUBJECT = "Ca-EEG2-1"  # the only mouse with CellReg tables
CROP = 110  # pixels, for the zoomed overlay panel; small enough to see single cells


def mask_image(session: load.Session, rois, shift=(0, 0)) -> np.ndarray:
    """A single binary image holding the outlines of the chosen cells."""
    height, width = session.max_projection.shape
    image = np.zeros((height, width), dtype=bool)
    wanted = np.isin(session.footprint_roi, np.asarray(rois))
    rows = session.footprint_row[wanted].astype(int) - shift[0]
    cols = session.footprint_col[wanted].astype(int) - shift[1]
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    image[rows[inside], cols[inside]] = True
    return image


def outline(axes, session, roi, rows, cols, shift=(0, 0), **style) -> None:
    """Draw one cell's outline inside a crop of the field of view.

    Each cell is drawn on its own. Contouring all of them at once merges neighbours that
    touch, which makes matched pairs look like separate cells sitting side by side.
    """
    image = mask_image(session, [roi], shift)[rows, cols]
    if image.any():
        axes.contour(image, levels=[0.5], **style)


def validate(subject: str, sessions: dict, config: dict) -> pd.DataFrame:
    """Score the footprint matcher against CellReg on every pair of sessions."""
    registration = match.load_cell_registration(subject, config)
    names = config["subjects"][subject]["behavioural_sessions"]
    days = {name: sessions[name].day for name in names}

    rows = []
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            pairs = match.match_pair(sessions[first], sessions[second], config)
            for min_tables in (1, config["matching"]["consensus_min_tables"]):
                result = match.compare_to_cellreg(pairs, registration, first, second, min_tables)
                result["min_tables"] = min_tables
                result["days_apart"] = abs(days[second] - days[first])
                rows.append(result)
    return pd.DataFrame(rows)


def plot(matched: dict, sessions: dict, validation: pd.DataFrame, config: dict,
         path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 9))

    # A: where the tracked cells are, in the day 1 field of view
    subject = "Ca-EEG3-4"
    day1 = sessions[subject][REFERENCE]
    names = config["subjects"][subject]["behavioural_sessions"]
    table = matched[subject]
    tracked = table.loc[(table[names] != match.MISSING).all(axis=1), REFERENCE].to_numpy()
    axes[0, 0].imshow(day1.max_projection, cmap="gray")
    axes[0, 0].contour(mask_image(day1, np.arange(day1.n_cells)), levels=[0.5],
                       colors="tab:blue", linewidths=0.3, alpha=0.5)
    axes[0, 0].contour(mask_image(day1, tracked), levels=[0.5],
                       colors="tab:orange", linewidths=0.7)
    axes[0, 0].set_title(f"{subject} day 1 field of view\n"
                         f"{day1.n_cells} cells, {len(tracked)} followed to day 6")
    axes[0, 0].axis("off")

    # B: the same cells on day 1 and day 6, after aligning the two fields of view
    last = sessions[subject]["Recall3"]
    shift = match.estimate_shift(day1.max_projection, last.max_projection)
    partners = table.loc[(table[names] != match.MISSING).all(axis=1), "Recall3"].to_numpy()
    centre = np.median(day1.centroids[tracked], axis=0).astype(int)
    rows = slice(centre[0] - CROP // 2, centre[0] + CROP // 2)
    cols = slice(centre[1] - CROP // 2, centre[1] + CROP // 2)
    inside = ((np.abs(day1.centroids[tracked, 0] - centre[0]) < CROP // 2 - 5)
              & (np.abs(day1.centroids[tracked, 1] - centre[1]) < CROP // 2 - 5))
    axes[0, 1].imshow(day1.max_projection[rows, cols], cmap="gray")
    for roi, partner in zip(tracked[inside], partners[inside]):
        outline(axes[0, 1], day1, roi, rows, cols,
                colors="tab:blue", linewidths=1.0)
        outline(axes[0, 1], last, partner, rows, cols, shift,
                colors="tab:orange", linewidths=1.0, linestyles="dashed")
    axes[0, 1].set_title(f"{int(inside.sum())} matched cells, day 1 (solid) and day 6 (dashed)\n"
                         f"field of view shifted by {shift} pixels")
    axes[0, 1].axis("off")

    # C: does the matcher agree with CellReg
    strict = validation[validation["min_tables"] > 1].sort_values("days_apart")
    labels = [f"{a.replace('NeutralExposure', 'Neutral')}\nvs {b}\n({d} d apart)"
              for a, b, d in zip(strict["session_a"], strict["session_b"], strict["days_apart"])]
    x = np.arange(len(strict))
    axes[1, 0].bar(x - 0.2, strict["recall"], width=0.4, label="CellReg pairs we recover")
    axes[1, 0].bar(x + 0.2, strict["agreement"], width=0.4, label="our pairs CellReg confirms")
    axes[1, 0].axhline(1.0, color="grey", lw=0.8, ls=":")
    axes[1, 0].set_xticks(x, labels, fontsize=8)
    axes[1, 0].set_ylim(0, 1.1)
    axes[1, 0].set_ylabel("fraction")
    axes[1, 0].set_title(f"{VALIDATION_SUBJECT}: matcher against CellReg")
    axes[1, 0].legend(fontsize=8, loc="lower left")

    # D: how many cells survive as the days pass
    for subject, table in matched.items():
        names = config["subjects"][subject]["behavioural_sessions"]
        days = [sessions[subject][name].day for name in names]
        counts = [(table[name] != match.MISSING).sum() for name in names]
        axes[1, 1].plot(days, counts, "o-", label=f"{subject} (pairwise)")
        cumulative = [(table[names[:i + 1]] != match.MISSING).all(axis=1).sum()
                      for i in range(len(names))]
        axes[1, 1].plot(days, cumulative, "s--", alpha=0.6,
                        label=f"{subject} (present in every session so far)")
    axes[1, 1].set_xlabel("day of the experiment")
    axes[1, 1].set_ylabel("cells followed from day 1")
    axes[1, 1].set_title("cells lost as days accumulate")
    axes[1, 1].legend(fontsize=7)

    figure.tight_layout()
    figures.save_figure(figure, path, config)
    plt.close(figure)


def main() -> None:
    config = load.load_config()
    sessions, matched = {}, {}

    for subject in config["subjects"]:
        print(f"matching {subject} ...")
        loaded = load.load_subject(subject, config)
        sessions[subject] = {s.session: s for s in loaded}
        table = match.match_to_reference(loaded, REFERENCE, config)
        table.to_csv(REPO / "results" / f"02_matched_{subject}.csv", index=False)
        matched[subject] = table

    print(f"validating against CellReg on {VALIDATION_SUBJECT} ...")
    validation = validate(VALIDATION_SUBJECT, sessions[VALIDATION_SUBJECT], config)
    validation.to_csv(REPO / "results" / "02_validation.csv", index=False)

    summary = {"reference_session": REFERENCE, "subjects": {}}
    for subject, table in matched.items():
        names = config["subjects"][subject]["behavioural_sessions"]
        offline = config["subjects"][subject]["offline_sessions"]
        summary["subjects"][subject] = {
            "cells_day1": int(sessions[subject][REFERENCE].n_cells),
            "matched_per_session": {name: int((table[name] != match.MISSING).sum())
                                    for name in names},
            "matched_all_behavioural": int((table[names] != match.MISSING).all(axis=1).sum()),
            "matched_all_including_offline": int(
                (table[names + offline] != match.MISSING).all(axis=1).sum()),
        }
    registration = match.load_cell_registration(VALIDATION_SUBJECT, config)
    names = config["subjects"][VALIDATION_SUBJECT]["behavioural_sessions"]
    consensus = match.consensus_matches(registration, names,
                                        config["matching"]["consensus_min_tables"])
    summary["cellreg_consensus_cells"] = int(len(consensus))
    summary["validation"] = validation.to_dict(orient="records")

    (REPO / "results" / "02_summary.json").write_text(json.dumps(summary, indent=1))
    plot(matched, sessions, validation, config, REPO / "figures" / "02_matching.png")

    print()
    print(validation[["session_a", "session_b", "days_apart", "min_tables",
                      "cellreg_pairs", "our_pairs", "recall", "agreement"]].to_string(index=False))
    print()
    print(json.dumps(summary["subjects"], indent=1))
    print("\nwrote figures/02_matching.png and results/02_*.csv")


if __name__ == "__main__":
    main()
