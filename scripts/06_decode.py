"""Phase 6: can a context learned on one day be recognised on later days.

Writes
    results/06_decoding.csv    every test session, both feature variants
    results/06_summary.json    the numbers quoted in the report
    figures/06_decoding.png    accuracy against days, and the within session control

Run:  .venv/bin/python scripts/06_decode.py
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import decode  # noqa: E402
import figures  # noqa: E402
import load  # noqa: E402
import match  # noqa: E402
import population  # noqa: E402

SHORT = {"NeutralExposure": "Neutral d1", "FC": "Shock d3", "Recall1": "Shock d4",
         "Recall2": "Novel d5", "Recall3": "Neutral d6"}


def features_per_session(subject: str, config: dict, variant: str):
    """One matrix of time bins by matched cells for each session."""
    names = config["subjects"][subject]["behavioural_sessions"]
    rois = match.common_rois(match.load_matched(subject), names)
    sessions = {s.session: s for s in load.load_subject(subject, config, include_offline=False)}
    binned = {n: population.binned_activity(sessions[n], config, rois[n]) for n in names}
    prepare = decode.standardise_within_session if variant == "per session" else decode.raw_features
    return sessions, names, {n: prepare(binned[n]) for n in names}


def split_in_half(features: np.ndarray) -> dict[str, np.ndarray]:
    """The control: the first and second half of one recording, as if two sessions."""
    middle = features.shape[0] // 2
    return {"first half": features[:middle], "second half": features[middle:]}


def main() -> None:
    config = load.load_config()
    train_names = config["decoding"]["train_sessions"]
    rows, controls, pooled = [], [], []

    for subject in config["subjects"]:
        for variant in ("per session", "raw"):
            print(f"decoding {subject} ({variant} features) ...")
            sessions, names, features = features_per_session(subject, config, variant)
            labels = {name: sessions[name].context for name in train_names}
            training = decode.equalise({name: features[name] for name in train_names})

            within = decode.block_cross_validation(training, labels, config)
            rng = np.random.default_rng(config["analysis"]["seed"])
            shuffled = decode.shuffled_cross_validation(training, labels, config, rng, 50)

            fitted = decode.fit(np.vstack([training[n] for n in train_names]),
                                np.concatenate([[labels[n]] * training[n].shape[0]
                                                for n in train_names]), config)

            for name in names:
                predictions = decode.predict(fitted, features[name])
                truth = sessions[name].context
                called_shock = float((predictions == "shock").mean())
                rows.append({
                    "subject": subject, "variant": variant, "session": name,
                    "day": sessions[name].day, "context": truth,
                    "in_training": name in train_names,
                    "days_from_training": min(abs(sessions[name].day - sessions[n].day)
                                              for n in train_names),
                    "bins": int(features[name].shape[0]),
                    "called_shock": called_shock,
                    "accuracy": called_shock if truth == "shock"
                                else (1 - called_shock) if truth == "neutral" else np.nan,
                    "within_training_accuracy": within,
                    "shuffled_accuracy": float(np.mean(shuffled)),
                    "shuffled_p95": float(np.percentile(shuffled, 95)),
                })

            # One balanced accuracy over the two held out sessions that have a correct
            # answer, pooled. Day 4 is the shock context and day 6 the neutral one, so a
            # decoder that answers the same thing on both scores one half here, whatever
            # its accuracy looks like session by session.
            pooled_truth, pooled_guess = [], []
            for name in names:
                if name in train_names or sessions[name].context == "novel":
                    continue
                guess = decode.predict(fitted, features[name])
                pooled_guess.append(guess)
                pooled_truth.append(np.repeat(sessions[name].context, len(guess)))
            if pooled_truth:
                pooled.append({
                    "subject": subject, "variant": variant,
                    "sessions": [n for n in names if n not in train_names
                                 and sessions[n].context != "novel"],
                    "balanced_accuracy": float(balanced_accuracy_score(
                        np.concatenate(pooled_truth), np.concatenate(pooled_guess))),
                })

            # control: split single sessions in half and decode one half from the other
            for name in names:
                halves = split_in_half(features[name])
                score = decode.block_cross_validation(
                    halves, {"first half": "first half", "second half": "second half"}, config)
                controls.append({"subject": subject, "variant": variant, "session": name,
                                 "halves_accuracy": score})

    table = pd.DataFrame(rows)
    control_table = pd.DataFrame(controls)
    pooled_table = pd.DataFrame(pooled)
    pooled_table.to_csv(REPO / "results" / "06_pooled_held_out.csv", index=False)
    table.to_csv(REPO / "results" / "06_decoding.csv", index=False)
    control_table.to_csv(REPO / "results" / "06_halves_control.csv", index=False)

    plot(table, control_table, pooled_table, config, REPO / "figures" / "06_decoding.png")

    headline = table[table["variant"] == "per session"]
    summary = {
        "train_sessions": train_names,
        "chance": 0.5,
        "within_training_accuracy": {s: float(headline[headline.subject == s]
                                              ["within_training_accuracy"].iloc[0])
                                     for s in config["subjects"]},
        "shuffled_accuracy": {s: float(headline[headline.subject == s]
                                       ["shuffled_accuracy"].iloc[0])
                              for s in config["subjects"]},
        "pooled_held_out": pooled_table[pooled_table["variant"] == "per session"]
            .to_dict(orient="records"),
        "tests": headline.to_dict(orient="records"),
        "halves_control": control_table.to_dict(orient="records"),
    }
    (REPO / "results" / "06_summary.json").write_text(json.dumps(summary, indent=1))

    print()
    print(table[["subject", "variant", "session", "day", "context", "in_training",
                 "called_shock", "accuracy", "within_training_accuracy",
                 "shuffled_accuracy"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(pooled_table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(control_table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nwrote figures/06_decoding.png and results/06_*.{csv,json}")


def plot(table: pd.DataFrame, controls: pd.DataFrame, pooled: pd.DataFrame,
         config: dict, path: Path) -> None:
    headline = table[table["variant"] == "per session"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6), width_ratios=[1.3, 1.2, 1])
    markers = {"neutral": "o", "shock": "s", "novel": "^"}
    colours = {"Ca-EEG2-1": "tab:blue", "Ca-EEG3-4": "tab:red"}

    # A: what the decoder calls each session
    axis = axes[0]
    for subject, group in headline.groupby("subject"):
        held_out = group[~group["in_training"]]
        axis.plot(group["day"], group["called_shock"], "-", color=colours[subject],
                  alpha=0.4, lw=1)
        for _, row in group.iterrows():
            axis.plot(row["day"], row["called_shock"], markers[row["context"]],
                      color=colours[subject], ms=10,
                      markerfacecolor=colours[subject] if row["in_training"] else "white",
                      markeredgewidth=1.6, label=subject)
        del held_out
    axis.axhline(0.5, color="grey", ls=":", lw=1)
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlabel("day of the experiment")
    axis.set_ylabel("share of time bins called shock")
    axis.set_title("what the decoder says\nfilled markers are training sessions\n"
                   "circle neutral, square shock, triangle novel", fontsize=10)
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(),
                fontsize=8, loc="center left")

    # B: the share of each held out day called shock, which is the quantity that shows
    # the collapse onto one answer. A decoder carrying context would put the shock day
    # high and the neutral day low; the horizontal line is what the training days give.
    axis = axes[1]
    width = 0.35
    subjects = list(headline["subject"].unique())
    for index, subject in enumerate(subjects):
        group = headline[(headline["subject"] == subject) & (~headline["in_training"])]
        positions = np.arange(len(group)) + index * width
        axis.bar(positions, group["called_shock"], width=width, color=colours[subject],
                 label=subject)
        for position, row in zip(positions, group.itertuples()):
            axis.text(position, row.called_shock + 0.02,
                      f"{row.called_shock:.2f}", ha="center", fontsize=7)
    longest = headline[~headline["in_training"]].groupby("subject").size().max()
    labels = [SHORT.get(n, n) for n in
              headline[(headline["subject"] == subjects[-1]) & (~headline["in_training"])]["session"]]
    axis.set_xticks(np.arange(longest) + width / 2, labels, fontsize=8)
    axis.axhline(0.5, color="grey", ls=":", lw=1)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("share of time bins called shock")
    axis.set_title("held out days, what share is called shock\n"
                   "a context code would separate these bars", fontsize=10)
    axis.legend(fontsize=8, loc="upper right")

    # C: the control, decoding one half of a session from the other
    axis = axes[2]
    control = controls[controls["variant"] == "per session"]
    positions = np.arange(len(control))
    axis.barh(positions, control["halves_accuracy"],
              color=[colours[s] for s in control["subject"]])
    axis.axvline(0.5, color="grey", ls=":", lw=1)
    axis.set_yticks(positions, [f"{r.subject} {SHORT.get(r.session, r.session)}"
                                for r in control.itertuples()], fontsize=7)
    axis.set_xlabel("accuracy")
    axis.set_title("control: first half against second half\nof one single session",
                   fontsize=10)

    figure.tight_layout()
    figures.save_figure(figure, path, config)
    plt.close(figure)


if __name__ == "__main__":
    main()
