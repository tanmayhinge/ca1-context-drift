"""Phase 7: is the neural result just a behavioural result in disguise.

Writes
    results/07_behaviour.csv     freezing and motion per session
    results/07_freezing_check.csv  population similarity with and without frozen frames
    results/07_summary.json      the numbers quoted in the report
    figures/07_behaviour.png     freezing per session, and the two checks

Run:  .venv/bin/python scripts/07_behaviour.py
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import behaviour  # noqa: E402
import figures  # noqa: E402
import load  # noqa: E402
import match  # noqa: E402
import population  # noqa: E402

SHORT = {"NeutralExposure": "Neutral d1", "FC": "Shock d3", "Recall1": "Shock d4",
         "Recall2": "Novel d5", "Recall3": "Neutral d6"}
COLOURS = {"neutral": "tab:green", "shock": "tab:red", "novel": "tab:purple"}


def main() -> None:
    config = load.load_config()
    behaviour_rows, check_rows = [], []

    for subject in config["subjects"]:
        print(f"behaviour for {subject} ...")
        names = config["subjects"][subject]["behavioural_sessions"]
        rois = match.common_rois(match.load_matched(subject), names)
        sessions = {s.session: s
                    for s in load.load_subject(subject, config, include_offline=False)}

        for name in names:
            session = sessions[name]
            behaviour_rows.append({
                "subject": subject, "session": name, "day": session.day,
                "context": session.context,
                "freezing_percent": behaviour.freezing_percent(session, config),
                "mean_motion": behaviour.mean_motion(session, config),
                "freezing_bouts": int(len(session.freezing_start)),
            })

        everything = {n: population.population_vector(sessions[n], config, rois[n])
                      for n in names}
        moving = {n: behaviour.moving_rates(sessions[n], config, rois[n]) for n in names}
        freezing = {row["session"]: row["freezing_percent"]
                    for row in behaviour_rows if row["subject"] == subject}

        for first, second in combinations(names, 2):
            check_rows.append({
                "subject": subject, "session_a": first, "session_b": second,
                "days_apart": abs(sessions[second].day - sessions[first].day),
                "same_context": sessions[first].context == sessions[second].context,
                "pearson_all_frames": float(pearsonr(everything[first],
                                                     everything[second]).statistic),
                "pearson_moving_only": float(pearsonr(moving[first], moving[second]).statistic),
                "freezing_difference": abs(freezing[first] - freezing[second]),
            })

    behaviour_table = pd.DataFrame(behaviour_rows)
    check_table = pd.DataFrame(check_rows)
    check_table["change"] = check_table["pearson_moving_only"] - check_table["pearson_all_frames"]
    behaviour_table.to_csv(REPO / "results" / "07_behaviour.csv", index=False)
    check_table.to_csv(REPO / "results" / "07_freezing_check.csv", index=False)

    summary = {"behaviour": behaviour_table.to_dict(orient="records"), "links": {}}
    for subject, group in check_table.groupby("subject"):
        if len(group) >= 3:
            link = spearmanr(group["freezing_difference"], group["pearson_all_frames"])
            summary["links"][subject] = {
                "session_pairs": int(len(group)),
                "spearman_freezing_difference_vs_similarity": float(link.statistic),
                "p_value": float(link.pvalue),
                "largest_change_from_dropping_frozen_frames":
                    float(group["change"].abs().max()),
                "mean_change_from_dropping_frozen_frames": float(group["change"].mean()),
            }
    summary["freezing_check"] = check_table.to_dict(orient="records")
    (REPO / "results" / "07_summary.json").write_text(json.dumps(summary, indent=1))

    plot(behaviour_table, check_table, config, REPO / "figures" / "07_behaviour.png")
    print()
    print(behaviour_table.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print()
    print(check_table[["subject", "session_a", "session_b", "freezing_difference",
                       "pearson_all_frames", "pearson_moving_only", "change"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(json.dumps(summary["links"], indent=1))
    print("\nwrote figures/07_behaviour.png and results/07_*.{csv,json}")


def plot(behaviour_table: pd.DataFrame, check: pd.DataFrame, config: dict, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4), width_ratios=[1.2, 1, 1])

    # A: freezing per session, per animal
    axis = axes[0]
    offset = 0
    for subject, group in behaviour_table.groupby("subject"):
        positions = np.arange(len(group)) + offset
        axis.bar(positions, group["freezing_percent"],
                 color=[COLOURS[c] for c in group["context"]],
                 edgecolor="black" if subject == "Ca-EEG2-1" else "none", linewidth=1.2)
        for position, row in zip(positions, group.itertuples()):
            axis.text(position, row.freezing_percent + 1.5, SHORT.get(row.session, row.session),
                      rotation=90, ha="center", va="bottom", fontsize=7)
        offset += len(group) + 1
    axis.set_ylabel("percent of the window spent frozen")
    axis.set_ylim(0, 80)
    axis.set_xticks([1, 6], list(behaviour_table["subject"].unique()), fontsize=9)
    axis.set_title("freezing per session\ncolour is context, outline marks the "
                   "strongly shocked mouse", fontsize=10)

    # B: does a change in behaviour go with a change in the code
    axis = axes[1]
    for subject, group in check.groupby("subject"):
        axis.scatter(group["freezing_difference"], group["pearson_all_frames"],
                     label=f"{subject} ({len(group)} pairs)", s=40)
    axis.set_xlabel("difference in freezing between the two sessions (points)")
    axis.set_ylabel("population similarity")
    axis.set_title("do behavioural shifts explain\nthe neural ones?", fontsize=10)
    axis.legend(fontsize=8)

    # C: does the result survive dropping the frozen frames
    axis = axes[2]
    for subject, group in check.groupby("subject"):
        axis.scatter(group["pearson_all_frames"], group["pearson_moving_only"],
                     s=40, label=subject)
    limits = [min(check["pearson_all_frames"].min(), check["pearson_moving_only"].min()) - 0.05,
              max(check["pearson_all_frames"].max(), check["pearson_moving_only"].max()) + 0.05]
    axis.plot(limits, limits, ls=":", color="grey", lw=1)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.legend(fontsize=8, loc="upper left")
    axis.set_xlabel("similarity, all frames")
    axis.set_ylabel("similarity, moving frames only")
    axis.set_title("dropping frozen frames\nchanges almost nothing", fontsize=10)

    figure.tight_layout()
    figures.save_figure(figure, path, config)
    plt.close(figure)


if __name__ == "__main__":
    main()
