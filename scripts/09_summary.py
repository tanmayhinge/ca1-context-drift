"""Phase 9: rerun every phase, freeze the numbers, and prepare the report.

Writes
    results/summary.json   every number the report quotes, in one place
    report/numbers.tex     the same numbers as LaTeX macros, so none is typed twice

Run:  .venv/bin/python scripts/09_summary.py            (aggregate what is on disk)
      .venv/bin/python scripts/09_summary.py --rerun    (rerun phases 1 to 7 first)
"""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PHASES = ["01_describe.py", "02_match.py", "03_overlap.py", "04_similarity.py",
          "05_pca.py", "06_decode.py", "07_behaviour.py"]


def rerun_every_phase() -> None:
    for script in PHASES:
        print(f"running {script} ...", flush=True)
        subprocess.run([sys.executable, str(REPO / "scripts" / script)],
                       check=True, capture_output=True)


def read(name: str) -> dict:
    return json.loads((REPO / "results" / name).read_text())


def find(records: list[dict], **conditions) -> dict:
    """The one record matching every condition, or a clear error."""
    hits = [r for r in records
            if all(r.get(key) == value for key, value in conditions.items())]
    if len(hits) != 1:
        raise LookupError(f"{len(hits)} records match {conditions}")
    return hits[0]


def collect() -> dict:
    """Pull the headline numbers out of each phase's own output."""
    matching = read("02_summary.json")
    overlap = read("03_summary.json")
    similarity = read("04_summary.json")
    pca = read("05_summary.json")
    decoding = read("06_summary.json")
    behaviour = read("07_summary.json")

    standard = [p for p in similarity["pairs"] if p["window"] == "standard"]
    weak = "Ca-EEG3-4"   # the mouse with five sessions and three contexts
    strong = "Ca-EEG2-1"  # the mouse with the CellReg tables

    validation = [v for v in matching["validation"] if v["min_tables"] > 1]
    amplitudes = read("00_shock_amplitudes.json")  # read from the NWB, not from prose
    summary = {
        "dataset": {
            "dandiset": "000718",
            "mice": 2,
            "strong_shock_mouse": strong,
            "weak_shock_mouse": weak,
            "shock_mA_strong": amplitudes[strong],
            "shock_mA_weak": amplitudes[weak],
        },
        "matching": {
            "cells_tracked_strong": matching["subjects"][strong]["matched_all_behavioural"],
            "cells_tracked_weak": matching["subjects"][weak]["matched_all_behavioural"],
            "cellreg_consensus_cells": matching["cellreg_consensus_cells"],
            "recall_min": min(v["recall"] for v in validation),
            "recall_max": max(v["recall"] for v in validation),
            "agreement_min": min(v["agreement"] for v in validation),
            "agreement_max": max(v["agreement"] for v in validation),
            # percentages read better in prose than "0.985 to 1"
            "recall_min_percent": round(100 * min(v["recall"] for v in validation), 1),
            "recall_max_percent": round(100 * max(v["recall"] for v in validation), 1),
            "agreement_min_percent": round(100 * min(v["agreement"] for v in validation), 1),
            "agreement_max_percent": round(100 * max(v["agreement"] for v in validation), 1),
        },
        "overlap": {
            "ensemble_percent": overlap["ensemble_percent"],
            "ratio_min": min(p["ratio_to_chance"] for p in overlap["pairs"]),
            "ratio_max": max(p["ratio_to_chance"] for p in overlap["pairs"]),
            "same_context_5_days": find(overlap["pairs"], subject=weak,
                                        session_a="NeutralExposure", session_b="Recall3"),
            "novel_context_4_days": find(overlap["pairs"], subject=weak,
                                         session_a="NeutralExposure", session_b="Recall2"),
        },
        "similarity": {
            "drift_spearman": similarity["similarity_against_days_apart"][weak]["spearman"],
            "drift_p_shuffled": similarity["similarity_against_days_apart"][weak]["p_shuffled"],
            "drift_pairs": similarity["similarity_against_days_apart"][weak]["pairs"],
            "drift_shuffles": similarity["similarity_against_days_apart"][weak]["shuffles"],
            "shuffled_band": max(p["pearson_shuffled_p95"] for p in standard),
            "pearson_min": min(p["pearson"] for p in standard),
            "pearson_max": max(p["pearson"] for p in standard),
            "same_context_5_days": find(standard, subject=weak,
                                        session_a="NeutralExposure", session_b="Recall3"),
            "novel_context_4_days": find(standard, subject=weak,
                                         session_a="NeutralExposure", session_b="Recall2"),
            "shock_context_3_days": find(standard, subject=weak,
                                         session_a="NeutralExposure", session_b="Recall1"),
        },
        "pca": {
            "variance_first_two_weak": pca["subjects"][weak]["variance_first_two"],
            "variance_first_two_strong": pca["subjects"][strong]["variance_first_two"],
            # percentages as well, because the report quotes them that way
            "variance_first_two_weak_percent":
                round(100 * pca["subjects"][weak]["variance_first_two"], 1),
            "variance_first_two_strong_percent":
                round(100 * pca["subjects"][strong]["variance_first_two"], 1),
            "components_for_half_weak": pca["subjects"][weak]["components_for_half_the_variance"],
            "separation_shuffled_min": min(p["separation_shuffled_labels"]
                                           for p in pca["separation"]),
            "separation_shuffled_max": max(p["separation_shuffled_labels"]
                                           for p in pca["separation"]),
            "same_context_excess": find(pca["separation"], subject=weak,
                                        session_a="NeutralExposure",
                                        session_b="Recall3")["separation_above_chance"],
        },
        "decoding": {
            "within_training_weak": decoding["within_training_accuracy"][weak],
            "within_training_strong": decoding["within_training_accuracy"][strong],
            "shuffled_weak": decoding["shuffled_accuracy"][weak],
            "shuffled_strong": decoding["shuffled_accuracy"][strong],
            "called_shock_recall1_weak": find(decoding["tests"], subject=weak,
                                              session="Recall1")["called_shock"],
            "called_shock_recall3_weak": find(decoding["tests"], subject=weak,
                                              session="Recall3")["called_shock"],
            "accuracy_recall1_weak": find(decoding["tests"], subject=weak,
                                          session="Recall1")["accuracy"],
            "accuracy_recall3_weak": find(decoding["tests"], subject=weak,
                                          session="Recall3")["accuracy"],
            "accuracy_recall1_strong": find(decoding["tests"], subject=strong,
                                            session="Recall1")["accuracy"],
            "called_shock_recall1_weak_percent":
                round(100 * find(decoding["tests"], subject=weak,
                                 session="Recall1")["called_shock"], 1),
            "called_shock_recall3_weak_percent":
                round(100 * find(decoding["tests"], subject=weak,
                                 session="Recall3")["called_shock"], 1),
            "called_shock_recall2_weak_percent":
                round(100 * find(decoding["tests"], subject=weak,
                                 session="Recall2")["called_shock"], 1),
            "called_shock_recall1_strong_percent":
                round(100 * find(decoding["tests"], subject=strong,
                                 session="Recall1")["called_shock"], 1),
            "pooled_held_out_weak": find(decoding["pooled_held_out"],
                                         subject=weak)["balanced_accuracy"],
            "halves_min": min(h["halves_accuracy"] for h in decoding["halves_control"]
                              if h["variant"] == "per session"),
            "halves_max": max(h["halves_accuracy"] for h in decoding["halves_control"]
                              if h["variant"] == "per session"),
        },
        "behaviour": {
            # freezing is reported to one decimal place, as a percentage of the window
            "freezing_strong_neutral": round(find(behaviour["behaviour"], subject=strong,
                                            session="NeutralExposure")["freezing_percent"], 1),
            "freezing_strong_recall1": round(find(behaviour["behaviour"], subject=strong,
                                            session="Recall1")["freezing_percent"], 1),
            "freezing_weak_recall1": round(find(behaviour["behaviour"], subject=weak,
                                          session="Recall1")["freezing_percent"], 1),
            "freezing_weak_recall3": round(find(behaviour["behaviour"], subject=weak,
                                          session="Recall3")["freezing_percent"], 1),
            "largest_change_dropping_frozen":
                behaviour["links"][weak]["largest_change_from_dropping_frozen_frames"],
            "link_spearman_weak":
                behaviour["links"][weak]["spearman_freezing_difference_vs_similarity"],
            "link_p_weak": behaviour["links"][weak]["p_value"],
        },
    }
    return summary


def latex_macros(summary: dict) -> str:
    """Every number as a LaTeX command, so the report never repeats a literal.

    Names are built from the path through the summary, for example
    summary["decoding"]["within_training_weak"] becomes \\numDecodingWithinTrainingWeak.
    """
    lines = ["% Generated by scripts/09_summary.py. Do not edit by hand.",
             "% Every number in report.tex comes from here, and therefore from results/."]

    # LaTeX command names may only contain letters, so digits become words
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
              "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}

    def camel(text: str) -> str:
        word = "".join(part.capitalize() for part in str(text).split("_"))
        return "".join(digits.get(character, character) for character in word)

    def emit(prefix: str, value) -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                emit(prefix + camel(key), inner)
        elif isinstance(value, bool):
            lines.append(f"\\newcommand{{\\num{prefix}}}{{{'yes' if value else 'no'}}}")
        elif isinstance(value, float):
            text = f"{value:.3f}".rstrip("0").rstrip(".")
            lines.append(f"\\newcommand{{\\num{prefix}}}{{{text}}}")
        elif isinstance(value, (int, str)):
            lines.append(f"\\newcommand{{\\num{prefix}}}{{{value}}}")

    emit("", summary)
    return "\n".join(lines) + "\n"


def main() -> None:
    if "--rerun" in sys.argv:
        rerun_every_phase()

    summary = collect()
    (REPO / "results" / "summary.json").write_text(json.dumps(summary, indent=1))
    (REPO / "report").mkdir(exist_ok=True)
    (REPO / "report" / "numbers.tex").write_text(latex_macros(summary))

    print(json.dumps(summary, indent=1))
    print(f"\nwrote results/summary.json and report/numbers.tex")


if __name__ == "__main__":
    main()
