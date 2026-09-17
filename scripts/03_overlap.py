"""Phase 3: which cells are active in each session, and how much the groups overlap.

Writes
    results/03_overlap.csv    every session pair, per animal, per ensemble size
    results/03_summary.json   the numbers quoted in the report
    figures/03_overlap.png    overlap matrices and the robustness check

Run:  .venv/bin/python scripts/03_overlap.py
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cells  # noqa: E402
import figures  # noqa: E402
import load  # noqa: E402
import match  # noqa: E402

SHORT = {"NeutralExposure": "Neutral d1", "FC": "Shock d3", "Recall1": "Shock d4",
         "Recall2": "Novel d5", "Recall3": "Neutral d6"}


def label(session: str) -> str:
    return SHORT.get(session, session)


def ensembles_for(subject: str, config: dict,
                  seconds: float | None = None) -> tuple[dict, dict, list[str]]:
    """Event rates for the cells this mouse keeps across all of its sessions."""
    names = config["subjects"][subject]["behavioural_sessions"]
    rois = match.common_rois(match.load_matched(subject), names)
    sessions = {s.session: s for s in load.load_subject(subject, config, include_offline=False)}
    rates = {name: cells.event_rate(sessions[name], config, rois[name], seconds)
             for name in names}
    return rates, sessions, names


def why_not_a_fixed_threshold(config: dict, path: Path) -> None:
    """Record why the ensemble is a percentile, with the numbers that forced the choice.

    The study plan asked for a fixed activity threshold. On this dataset that rule calls
    almost every cell active, because the segmentation only keeps cells it could detect.
    This writes the evidence rather than asserting it.
    """
    rows = []
    for subject in config["subjects"]:
        for session in load.load_subject(subject, config, include_offline=False):
            rates = cells.event_rate(session, config)
            rows.append({
                "session": f"{subject} {session.session}",
                "cells": session.n_cells,
                "at least 0.05 events per second": f"{np.mean(rates >= 0.05):.0%}",
                "at least 0.1": f"{np.mean(rates >= 0.1):.0%}",
            })
    percent = config["analysis"]["ensemble_percent"]
    checked = config["analysis"]["ensemble_percent_checked"]
    path.write_text(
        "# Why the ensemble is a percentile and not a fixed threshold\n\n"
        "The study plan said to call a cell active when its activity passes a threshold.\n"
        "That rule does not work on this dataset, and the numbers show why. Minian only\n"
        "keeps cells it could detect in the first place, so within a few minutes almost\n"
        "every kept cell fires at least a few times:\n\n"
        + pd.DataFrame(rows).to_markdown(index=False)
        + "\n\nWritten by scripts/03_overlap.py, using the session windows in config.yaml.\n"
        "If a rule calls this many cells active, every pair of sessions overlaps almost\n"
        "completely and the measurement carries no information.\n\n"
        f"So an ensemble here is the most active {percent} percent of the cells in that\n"
        f"session. Chance overlap is then well defined, and the analysis is repeated at\n"
        f"{checked} percent so the reader can see whether the cut-off drives the result.\n")


def main() -> None:
    config = load.load_config()
    settings = config["analysis"]
    why_not_a_fixed_threshold(config, REPO / "results" / "03_overlap.md")
    rows = []

    # "standard" takes 300 s of each session, except conditioning which only has 120 s
    # of pre-shock time. "equal" takes 120 s of every session, so no session is ranked
    # from more data than another.
    windows = {"standard": None, "equal": settings["fc_baseline_seconds"]}

    for subject in config["subjects"]:
      print(f"overlap for {subject} ...")
      for window_name, seconds in windows.items():
        rates, sessions, names = ensembles_for(subject, config, seconds)
        for percent in settings["ensemble_percent_checked"]:
            groups = {name: cells.ensemble(rates[name], percent) for name in names}
            for first, second in combinations(names, 2):
                rng = np.random.default_rng(settings["seed"])
                stats = cells.overlap_stats(groups[first], groups[second], rng,
                                            settings["n_shuffles"])
                stats.update(subject=subject, session_a=first, session_b=second,
                             percent=percent, window=window_name,
                             days_apart=abs(sessions[second].day - sessions[first].day),
                             same_context=sessions[first].context == sessions[second].context)
                rows.append(stats)

    table = pd.DataFrame(rows)
    table.to_csv(REPO / "results" / "03_overlap.csv", index=False)

    main_percent = settings["ensemble_percent"]
    headline = table[(table["percent"] == main_percent) & (table["window"] == "standard")]
    equal = table[(table["percent"] == main_percent) & (table["window"] == "equal")]
    plot(headline, table, config, REPO / "figures" / "03_overlap.png")

    summary = {
        "ensemble_percent": main_percent,
        "cells_used": {s: int(headline[headline.subject == s]["cells"].iloc[0])
                       for s in config["subjects"]},
        "pairs_equal_window": equal[["subject", "session_a", "session_b",
                                     "ratio_to_chance"]].to_dict(orient="records"),
        "pairs": headline[["subject", "session_a", "session_b", "days_apart", "same_context",
                           "active_a", "active_b", "observed", "expected", "ratio_to_chance",
                           "ratio_ci_low", "ratio_ci_high", "jaccard", "p_exact", "p_shuffle"]].to_dict(orient="records"),
    }
    (REPO / "results" / "03_summary.json").write_text(json.dumps(summary, indent=1))

    shown = headline[["subject", "session_a", "session_b", "days_apart", "same_context",
                      "observed", "expected", "ratio_to_chance", "ratio_ci_low",
                      "ratio_ci_high", "p_exact"]]
    print()
    print(shown.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\ncells used: {summary['cells_used']}")
    print("wrote figures/03_overlap.png and results/03_*.{csv,json}")


def plot(headline: pd.DataFrame, everything: pd.DataFrame, config: dict, path: Path) -> None:
    subjects = list(config["subjects"])
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6),
                                width_ratios=[1, 1.25, 1.35])
    # every pair sits above chance, so the scale runs from chance to the largest value
    # seen rather than saturating and making all the squares look alike
    top = float(headline["ratio_to_chance"].max())

    for axis, subject in zip(axes[:2], subjects):
        names = config["subjects"][subject]["behavioural_sessions"]
        grid = np.full((len(names), len(names)), np.nan)
        rows = headline[headline["subject"] == subject]
        for _, row in rows.iterrows():
            i, j = names.index(row["session_a"]), names.index(row["session_b"])
            grid[i, j] = grid[j, i] = row["ratio_to_chance"]
        np.fill_diagonal(grid, np.nan)

        image = axis.imshow(grid, cmap="Reds", vmin=1.0, vmax=top)
        for i in range(len(names)):
            for j in range(len(names)):
                if not np.isnan(grid[i, j]):
                    axis.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=8)
        axis.set_xticks(range(len(names)), [label(n) for n in names], rotation=45,
                        ha="right", fontsize=8)
        axis.set_yticks(range(len(names)), [label(n) for n in names], fontsize=8)
        cell_count = int(rows["cells"].iloc[0])
        axis.set_title(f"{subject}\n{cell_count} matched cells, "
                       f"top {config['analysis']['ensemble_percent']} percent active")
        figure.colorbar(image, ax=axis, fraction=0.046, label="overlap / chance")

    # robustness: does the ensemble size drive the answer
    axis = axes[2]
    subject = "Ca-EEG3-4"
    reference = "NeutralExposure"
    rows = everything[(everything["subject"] == subject)
                      & (everything["session_a"] == reference)
                      & (everything["window"] == "standard")]
    for session, group in rows.groupby("session_b", sort=False):
        group = group.sort_values("percent")
        axis.errorbar(group["percent"], group["ratio_to_chance"],
                      yerr=[group["ratio_to_chance"] - group["ratio_ci_low"],
                            group["ratio_ci_high"] - group["ratio_to_chance"]],
                      fmt="o-", capsize=3, label=label(session))
    axis.axhline(1.0, color="grey", ls=":", lw=1)
    axis.set_xlabel("ensemble size (percent of cells called active)")
    axis.set_ylabel("overlap with Neutral d1, relative to chance")
    axis.set_title(f"{subject}: does the cut-off drive the answer?\n"
                   "bars are 95 percent intervals over cells")
    axis.legend(fontsize=8)

    figure.tight_layout()
    figures.save_figure(figure, path, config)
    plt.close(figure)


if __name__ == "__main__":
    main()
