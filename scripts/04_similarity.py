"""Phase 4: how similar is the whole population pattern between sessions.

Writes
    results/04_similarity.csv   every session pair, three measures, baselines
    results/04_summary.json     the numbers quoted in the report
    figures/04_similarity.png   similarity matrices and the drift curve

Run:  .venv/bin/python scripts/04_similarity.py
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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
REFERENCE = "NeutralExposure"


def similarity_against_days(pairs: pd.DataFrame, config: dict) -> dict:
    """Does similarity fall as the gap between two sessions grows.

    Spearman is used because only the ordering of the gaps is meaningful. The p value comes
    from shuffling which similarity belongs to which gap, since the ten pairs are built from
    five sessions and therefore share sessions with each other, which makes the usual
    formula for a p value optimistic.
    """
    gaps = pairs["days_apart"].to_numpy(float)
    values = pairs["pearson"].to_numpy(float)
    observed = float(spearmanr(gaps, values).statistic)

    rng = np.random.default_rng(config["analysis"]["seed"])
    draws = np.array([spearmanr(gaps, rng.permutation(values)).statistic
                      for _ in range(config["analysis"]["n_shuffles"] * 10)])
    return {
        "pairs": int(len(pairs)),
        "spearman": observed,
        "p_shuffled": float((np.abs(draws) >= abs(observed)).mean()),
        "shuffles": int(len(draws)),
    }


def bin_size_sweep(subject: str, config: dict, path: Path) -> pd.DataFrame:
    """Why the bin by bin measure uses 10 s bins and not 1 s.

    The honest check is the within session value: if two halves of the same session do not
    correlate above the shuffled level, the measure cannot detect anything between sessions
    either. That is exactly what happens at 1 s.
    """
    names = config["subjects"][subject]["behavioural_sessions"]
    rois = match.common_rois(match.load_matched(subject), names)
    sessions = {s.session: s for s in load.load_subject(subject, config, include_offline=False)}
    first, second = names[0], names[2]
    rng = np.random.default_rng(config["analysis"]["seed"])

    rows = []
    for seconds in [1, 2, 5, 10, 20, 30]:
        settings = {**config, "analysis": {**config["analysis"], "bin_seconds": seconds}}
        binned = {n: population.binned_activity(sessions[n], settings, rois[n])
                  for n in (first, second)}
        order = rng.permutation(len(rois[first]))
        rows.append({
            "bin_seconds": seconds,
            "bins_per_session": binned[first].shape[1],
            f"within {first}": population.within_session_bin_similarity(binned[first]),
            f"within {second}": population.within_session_bin_similarity(binned[second]),
            f"{first} vs {second}": population.bin_similarity(binned[first], binned[second]),
            "shuffled cell identities": population.bin_similarity(binned[first],
                                                                  binned[second][order]),
        })
    sweep = pd.DataFrame(rows)
    sweep.to_csv(path, index=False)
    return sweep


def prepare(subject: str, config: dict, seconds: float | None = None):
    """Population vectors and binned activity for the cells matched across all sessions.

    `seconds` shortens every session to the same length. Without it the conditioning
    session contributes 120 s of pre-shock time against 300 s elsewhere, and a shorter
    recording gives a noisier estimate of each cell's rate, which drags its correlations
    down for reasons that have nothing to do with the brain.
    """
    names = config["subjects"][subject]["behavioural_sessions"]
    rois = match.common_rois(match.load_matched(subject), names)
    sessions = {s.session: s for s in load.load_subject(subject, config, include_offline=False)}
    vectors = {n: population.population_vector(sessions[n], config, rois[n], seconds)
               for n in names}
    # the bin by bin measure needs coarser bins than the decoder does, see bin_size_sweep
    coarse = {**config, "analysis": {**config["analysis"],
                                     "bin_seconds": config["analysis"]["similarity_bin_seconds"]}}
    binned = {n: population.binned_activity(sessions[n], coarse, rois[n], seconds)
              for n in names}
    return sessions, names, vectors, binned


def main() -> None:
    config = load.load_config()
    settings = config["analysis"]
    sweep = bin_size_sweep("Ca-EEG3-4", config, REPO / "results" / "04_bin_size.csv")
    rows, ceilings = [], []

    windows = {"standard": None, "equal": settings["fc_baseline_seconds"]}
    for subject in config["subjects"]:
      print(f"similarity for {subject} ...")
      for window_name, seconds in windows.items():
        sessions, names, vectors, binned = prepare(subject, config, seconds)
        for name in names:
            ceilings.append({
                "subject": subject, "session": name, "window": window_name,
                "within_session_bin_similarity":
                    population.within_session_bin_similarity(binned[name]),
                "bins": binned[name].shape[1],
            })
        for first, second in combinations(names, 2):
            rng = np.random.default_rng(settings["seed"])
            result = population.compare(vectors[first], vectors[second],
                                        binned[first], binned[second],
                                        rng, settings["n_shuffles"])
            ceiling = np.mean([population.within_session_bin_similarity(binned[first]),
                               population.within_session_bin_similarity(binned[second])])
            result.update(bin_ceiling=float(ceiling),
                          bin_share_of_ceiling=float(result["bin_similarity"] / ceiling),
                          subject=subject, session_a=first, session_b=second,
                          window=window_name,
                          days_apart=abs(sessions[second].day - sessions[first].day),
                          same_context=sessions[first].context == sessions[second].context)
            rows.append(result)

    table = pd.DataFrame(rows)
    table.to_csv(REPO / "results" / "04_similarity.csv", index=False)
    ceiling_table = pd.DataFrame(ceilings)
    standard = table[table["window"] == "standard"].reset_index(drop=True)

    plot(table, ceiling_table, config, REPO / "figures" / "04_similarity.png")

    drift = {}
    for subject in config["subjects"]:
        rows_for = standard[standard["subject"] == subject]
        if len(rows_for) >= 6:  # only worth asking where there are enough session pairs
            drift[subject] = similarity_against_days(rows_for, config)

    summary = {
        "bin_size_sweep": sweep.to_dict(orient="records"),
        "similarity_against_days_apart": drift,
        "measures": ["pearson", "spearman", "bin_similarity"],
        "within_session_ceiling": ceiling_table.to_dict(orient="records"),
        "pairs": table.to_dict(orient="records"),
    }
    (REPO / "results" / "04_summary.json").write_text(json.dumps(summary, indent=1))

    shown = table[["subject", "session_a", "session_b", "window", "days_apart", "same_context",
                   "pearson", "pearson_ci_low", "pearson_ci_high", "pearson_shuffled",
                   "spearman", "bin_similarity", "bin_ceiling", "bin_share_of_ceiling"]]
    print()
    print(shown.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(ceiling_table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(sweep.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print("similarity against days apart:", json.dumps(drift, indent=1))
    print("\nwrote figures/04_similarity.png and results/04_*.{csv,json}")


def plot(table: pd.DataFrame, ceilings: pd.DataFrame, config: dict, path: Path) -> None:
    subjects = list(config["subjects"])
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6), width_ratios=[1, 1.25, 1.5])

    standard = table[table["window"] == "standard"]
    equal = table[table["window"] == "equal"]
    values = standard["pearson"]
    for axis, subject in zip(axes[:2], subjects):
        names = config["subjects"][subject]["behavioural_sessions"]
        grid = np.full((len(names), len(names)), np.nan)
        rows = standard[standard["subject"] == subject]
        for _, row in rows.iterrows():
            i, j = names.index(row["session_a"]), names.index(row["session_b"])
            grid[i, j] = grid[j, i] = row["pearson"]
        image = axis.imshow(grid, cmap="viridis", vmin=values.min(), vmax=values.max())
        for i in range(len(names)):
            for j in range(len(names)):
                if not np.isnan(grid[i, j]):
                    axis.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                              fontsize=8, color="white")
        axis.set_xticks(range(len(names)), [SHORT.get(n, n) for n in names],
                        rotation=45, ha="right", fontsize=8)
        axis.set_yticks(range(len(names)), [SHORT.get(n, n) for n in names], fontsize=8)
        axis.set_title(f"{subject}\n{int(rows['cells'].iloc[0])} matched cells")
        figure.colorbar(image, ax=axis, fraction=0.046, label="correlation of activity rates")

    # the drift curve: similarity to day 1 as the days pass
    axis = axes[2]
    markers = {"neutral": "o", "shock": "s", "novel": "^"}
    for subject in subjects:
        contexts = config["contexts"]
        fair = equal[(equal["subject"] == subject) & (equal["session_a"] == REFERENCE)]
        axis.plot(fair["days_apart"], fair["pearson"], ":", color="grey", lw=1.0, zorder=1,
                  label="equal 120 s windows")
        rows = standard[(standard["subject"] == subject)
                        & (standard["session_a"] == REFERENCE)]
        axis.plot(rows["days_apart"], rows["pearson"], "-", color="grey", lw=0.8, zorder=1)
        for _, row in rows.iterrows():
            context = contexts[row["session_b"]]
            axis.errorbar(row["days_apart"], row["pearson"],
                          yerr=[[row["pearson"] - row["pearson_ci_low"]],
                                [row["pearson_ci_high"] - row["pearson"]]],
                          fmt=markers[context], capsize=3,
                          color="tab:blue" if subject == subjects[0] else "tab:red",
                          label=f"{subject}, {context}")
    band = standard["pearson_shuffled_p95"].max()
    axis.axhspan(-band, band, color="grey", alpha=0.2, lw=0,
                 label="shuffled cell identities (95 percent)")
    axis.set_xlabel("days between the session and day 1")
    axis.set_ylabel("correlation with the day 1 neutral session")
    axis.set_title("drift away from day 1\nmarker shape is the context")
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), fontsize=7, loc="lower right",
                framealpha=0.95)

    figure.tight_layout()
    figures.save_figure(figure, path, config)
    plt.close(figure)


if __name__ == "__main__":
    main()
