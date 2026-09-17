"""Phase 5: sessions and contexts as points on a plot.

Writes
    results/05_pca.csv          variance explained, and separation between session pairs
    results/05_summary.json     the numbers quoted in the report
    figures/05_pca.png          the first two components, and a scree plot

Run:  .venv/bin/python scripts/05_pca.py
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

import figures  # noqa: E402
import load  # noqa: E402
import match  # noqa: E402
import population  # noqa: E402

SHORT = {"NeutralExposure": "Neutral d1", "FC": "Shock d3", "Recall1": "Shock d4",
         "Recall2": "Novel d5", "Recall3": "Neutral d6"}
COLOURS = {"neutral": "tab:green", "shock": "tab:red", "novel": "tab:purple"}


def embed(subject: str, config: dict):
    """Time bins of every session, standardised, projected onto principal components."""
    names = config["subjects"][subject]["behavioural_sessions"]
    rois = match.common_rois(match.load_matched(subject), names)
    sessions = {s.session: s for s in load.load_subject(subject, config, include_offline=False)}
    coarse = {**config, "analysis": {**config["analysis"],
                                     "bin_seconds": config["analysis"]["similarity_bin_seconds"]}}
    binned = {n: population.binned_activity(sessions[n], coarse, rois[n]) for n in names}
    stacked, labels = population.stack_sessions(binned)
    coordinates, variance = population.principal_components(stacked)
    return sessions, names, stacked, labels, coordinates, variance


def main() -> None:
    config = load.load_config()
    rows, summary = [], {"bin_seconds": config["analysis"]["similarity_bin_seconds"],
                         "subjects": {}}
    embeddings = {}

    for subject in config["subjects"]:
        print(f"pca for {subject} ...")
        sessions, names, stacked, labels, coordinates, variance = embed(subject, config)
        embeddings[subject] = (sessions, names, labels, coordinates, variance)

        summary["subjects"][subject] = {
            "cells": int(stacked.shape[1]),
            "time_bins": int(stacked.shape[0]),
            "variance_pc1": float(variance[0]),
            "variance_pc2": float(variance[1]),
            "variance_first_two": float(variance[:2].sum()),
            "components_kept": int(len(variance)),
            "components_for_half_the_variance": int(np.searchsorted(np.cumsum(variance), 0.5) + 1),
        }
        for first, second in combinations(names, 2):
            rng = np.random.default_rng(config["analysis"]["seed"])
            chance = population.separation_under_shuffled_labels(
                stacked, labels, first, second, rng, 200)
            observed = population.separation(stacked, labels, first, second)
            rows.append({
                "subject": subject, "session_a": first, "session_b": second,
                "days_apart": abs(sessions[second].day - sessions[first].day),
                "same_context": sessions[first].context == sessions[second].context,
                "separation": observed,
                "separation_shuffled_labels": chance,
                "separation_above_chance": observed - chance,
            })

    table = pd.DataFrame(rows)
    table.to_csv(REPO / "results" / "05_pca.csv", index=False)
    summary["separation"] = table.to_dict(orient="records")
    (REPO / "results" / "05_summary.json").write_text(json.dumps(summary, indent=1))

    plot(embeddings, config, REPO / "figures" / "05_pca.png")
    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print()
    print(json.dumps(summary["subjects"], indent=1))
    print("\nwrote figures/05_pca.png and results/05_*.{csv,json}")


def plot(embeddings: dict, config: dict, path: Path) -> None:
    subjects = list(config["subjects"])
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6), width_ratios=[1.2, 1.2, 1])

    for axis, subject in zip(axes[:2], subjects):
        sessions, names, labels, coordinates, variance = embeddings[subject]
        days = [sessions[n].day for n in names]
        for name in names:
            mask = labels == name
            context = sessions[name].context
            # later days are drawn paler, so time reads as fading without another colour
            shade = 1.0 - 0.55 * (sessions[name].day - min(days)) / max(max(days) - min(days), 1)
            axis.scatter(coordinates[mask, 0], coordinates[mask, 1], s=22,
                         color=COLOURS[context], alpha=0.35 * shade + 0.2, edgecolor="none")
            centre = coordinates[mask, :2].mean(axis=0)
            # the big marker is the session average, which is what the eye should compare
            axis.scatter(*centre, s=190, marker="X", color=COLOURS[context], alpha=shade,
                         edgecolor="black", linewidth=0.8, zorder=5,
                         label=SHORT.get(name, name))
        axis.set_xlabel(f"PC1 ({variance[0]:.0%} of variance)")
        axis.set_ylabel(f"PC2 ({variance[1]:.0%})")
        axis.set_title(f"{subject}\ncrosses are session averages", fontsize=11)
        axis.legend(fontsize=7, loc="best", framealpha=0.9)

    axis = axes[2]
    for subject in subjects:
        variance = embeddings[subject][4]
        shown = min(len(variance), 30)
        axis.plot(np.arange(1, shown + 1), np.cumsum(variance)[:shown], "o-", ms=3,
                  label=subject)
    axis.axhline(0.5, color="grey", ls=":", lw=1)
    axis.set_xlabel("number of components")
    axis.set_ylabel("share of variance explained")
    axis.set_title("how much a 2D picture leaves out")
    axis.legend(fontsize=8)

    figure.suptitle(f"each small dot is {config['analysis']['similarity_bin_seconds']} s "
                    f"of population activity", y=0.02, fontsize=9)
    figure.tight_layout()
    figures.save_figure(figure, path, config)
    plt.close(figure)


if __name__ == "__main__":
    main()
