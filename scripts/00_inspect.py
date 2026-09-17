"""Phase 0: look inside DANDI 000718 before writing any analysis.

Nothing is downloaded. Every file is read over HTTP with byte range requests, so this
script touches a few megabytes of a 40 GB archive. It writes:

    results/00_assets.csv            every NWB file with its size
    results/00_session_inventory.csv what each session contains
    results/00_nwb_tree.txt          the internal structure of three representative files
    results/00_inspect.md            the human readable report for the decision gate

Run:  .venv/bin/python scripts/00_inspect.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import load  # noqa: E402

RESULTS = REPO / "results"
MISSING = -9999  # CellReg writes this when a cell has no counterpart in a session
BEHAVIOURAL_COLUMNS = ["Ca_EEG2-1_NeutralExposure", "Ca_EEG2-1_FC", "Ca_EEG2-1_Recall1"]


def dump_tree(h5file: h5py.File, title: str, out) -> None:
    """Write every group and dataset in the file, with shapes and dtypes."""
    out.write(f"\n{'=' * 78}\n{title}\n{'=' * 78}\n")

    def visit(name, obj):
        if name.startswith("specifications"):
            return  # the NWB schema itself, not data
        if isinstance(obj, h5py.Dataset):
            out.write(f"{name}  shape={obj.shape} dtype={obj.dtype}\n")
        else:
            out.write(f"{name}/\n")

    h5file.visititems(visit)


def scalar(h5file: h5py.File, path: str):
    """Read a scalar dataset, decoding bytes to text."""
    if path not in h5file:
        return None
    value = h5file[path][()]
    return value.decode() if isinstance(value, bytes) else value


def inventory_one_session(asset: load.Asset) -> dict:
    """What a single session file actually contains."""
    h5file, _remote = load.open_remote_h5(asset.url)
    row = {"subject": asset.subject, "session": asset.session, "size_gb": round(asset.size_gb, 2)}
    row["start_time"] = scalar(h5file, "session_start_time")

    traces = "processing/ophys/Fluorescence/Deconvolved/data"
    if traces in h5file:
        frames, cells = h5file[traces].shape
        stamps = h5file["processing/ophys/Fluorescence/Deconvolved/timestamps"][:]
        row["cells"] = cells
        row["frames"] = frames
        row["rate_hz"] = round(1.0 / float(np.median(np.diff(stamps))), 2)
        row["minutes"] = round((stamps[-1] - stamps[0]) / 60.0, 1)
    else:
        row["cells"] = row["frames"] = row["rate_hz"] = row["minutes"] = None

    row["denoised"] = "processing/ophys/Fluorescence/Denoised/data" in h5file
    row["masks"] = "processing/ophys/ImageSegmentation/PlaneSegmentation/image_mask" in h5file
    row["centroids"] = "processing/ophys/ImageSegmentation/PlaneSegmentation/ROICentroids" in h5file
    row["max_projection"] = "processing/ophys/SegmentationImages/maximum_projection" in h5file

    freezing = "processing/behavior/FreezingIntervals/start_time"
    row["freezing_bouts"] = int(h5file[freezing].shape[0]) if freezing in h5file else None
    row["motion"] = "processing/behavior/MotionSeries/data" in h5file

    shocks = "stimulus/presentation/ShockStimuli/start_time"
    row["shock_times_s"] = [round(float(t), 1) for t in h5file[shocks][:]] if shocks in h5file else None
    # the shock amplitude decides which group of the original study a mouse belongs to,
    # so it is read from the file rather than from the session description text
    amplitude = "stimulus/presentation/ShockStimuli/shock_amplitude"
    row["shock_amplitude_mA"] = (float(np.unique(h5file[amplitude][:])[0])
                                 if amplitude in h5file else None)

    # the sleep module exists in the offline files but is empty for these short recordings
    row["sleep_states"] = "processing/sleep" in h5file and len(h5file["processing/sleep"]) > 0

    h5file.close()
    return row


def summarise_cell_registration(asset: load.Asset) -> dict:
    """How many cells the provided CellReg tables follow across sessions.

    The dataset holds 19 independent registrations, one per offline recording, each of
    which also registers the three behavioural sessions. Comparing them tells us how
    reliable a matched cell is, which no single table can tell us on its own.
    """
    h5file, _remote = load.open_remote_h5(asset.url)
    module = h5file["processing/cell_registration"]
    table_names = sorted(module.keys())

    per_table = {}
    triplet_votes: Counter = Counter()
    for name in table_names:
        table = module[name]
        columns = {column: table[column][:] for column in BEHAVIOURAL_COLUMNS}
        complete = np.all(np.vstack([values != MISSING for values in columns.values()]), axis=0)
        per_table[name] = int(complete.sum())
        rows = np.vstack([columns[column][complete] for column in BEHAVIOURAL_COLUMNS]).T
        triplet_votes.update(tuple(int(value) for value in row) for row in rows)

    example = "OfflineDay2Session1vsConditioningSessions"
    table = module[example]
    columns = [name for name in table.keys() if name != "id"]
    ids = {name: table[name][:] for name in columns}
    present = np.vstack([values != MISSING for values in ids.values()])

    result = {
        "n_tables": len(table_names),
        "example": example,
        "example_columns": columns,
        "example_global_cells": len(next(iter(ids.values()))),
        "example_per_session": {name: int((v != MISSING).sum()) for name, v in ids.items()},
        "example_all_columns": int(present.all(axis=0).sum()),
        "per_table_behavioural": per_table,
        "unique_triplets": len(triplet_votes),
        "triplets_unanimous": sum(1 for v in triplet_votes.values() if v == len(table_names)),
        "triplets_majority": sum(1 for v in triplet_votes.values() if v >= 10),
    }
    h5file.close()
    return result


def probe_fallback_dandiset() -> list[str]:
    """A quick look at 001710, the fallback named in the study plan."""
    try:
        assets = load.list_assets(dandiset="001710", version="draft")
    except Exception as error:  # the probe must never block the gate report
        return [f"could not list 001710: {error}"]

    subjects = sorted({a.subject for a in assets})
    sizes = [a.size_gb for a in assets]
    notes = [
        f"- {len(assets)} NWB files, {len(subjects)} subjects, {sum(sizes):.1f} GB total, "
        f"{min(sizes):.2f} to {max(sizes):.2f} GB per file",
        f"- sessions for {subjects[0]}: {sorted(a.session for a in assets if a.subject == subjects[0])}",
    ]

    h5file, _remote = load.open_remote_h5(assets[0].url)
    paths = []
    h5file.visititems(lambda name, obj: paths.append(name))
    paths = [p for p in paths if not p.startswith("specifications")]
    behaviour = sorted({p for p in paths if "2P-aligned behavior/" in p and p.count("/") == 3})
    ophys = sorted({p for p in paths if p.startswith("processing/ophys/") and p.count("/") == 3})
    identity = [p for p in paths if any(k in p.lower() for k in ("global", "match", "registration", "crossday"))]
    notes += [
        f"- behaviour aligned to imaging: {[p.split('/')[-1] for p in behaviour]}",
        f"- ophys contents: {ophys}",
        f"- fields naming cross day cell identity: {identity if identity else 'none'}",
    ]
    h5file.close()
    return notes


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    print("listing assets ...")
    assets = load.list_assets()
    table = load.assets_table(assets)
    table.to_csv(RESULTS / "00_assets.csv", index=False)

    print("dumping NWB trees ...")
    with open(RESULTS / "00_nwb_tree.txt", "w") as out:
        out.write("Internal structure of three representative files from DANDI 000718.\n")
        for subject, session in [("Ca-EEG3-4", "Recall1"),
                                 ("Ca-EEG3-4", "OfflineDay2Session1"),
                                 ("Ca-EEG2-1", "Week")]:
            asset = load.find_asset(assets, subject, session)
            h5file, _remote = load.open_remote_h5(asset.url)
            dump_tree(h5file, f"{subject}  {session}  ({asset.size_gb:.2f} GB)", out)
            h5file.close()

    print("inventory of every session ...")
    rows = []
    for asset in assets:
        if asset.session == "Week":
            continue
        print(f"    {asset.subject} {asset.session}")
        rows.append(inventory_one_session(asset))
    inventory = pd.DataFrame(rows)
    inventory.to_csv(RESULTS / "00_session_inventory.csv", index=False)

    # the two mice come from opposite groups of the original study, so the amplitudes are
    # written out on their own for the report to quote
    shocked = inventory.dropna(subset=["shock_amplitude_mA"])
    amplitudes = dict(zip(shocked["subject"], shocked["shock_amplitude_mA"]))
    (RESULTS / "00_shock_amplitudes.json").write_text(json.dumps(amplitudes, indent=1))

    print("reading cell registration tables ...")
    cellreg = summarise_cell_registration(load.find_asset(assets, "Ca-EEG2-1", "Week"))

    print("probing fallback dandiset 001710 ...")
    fallback = probe_fallback_dandiset()

    counts = sorted(cellreg["per_table_behavioural"].values())
    report = [
        "# Phase 0: what is inside DANDI 000718",
        "",
        "Written by `scripts/00_inspect.py`. Nothing was downloaded; every read was an HTTP",
        "byte range request against the archive.",
        "",
        "## Files",
        "",
        table.to_markdown(index=False),
        "",
        "## What each session contains",
        "",
        inventory.to_markdown(index=False),
        "",
        "## Cross day cell registration",
        "",
        "The `cell_registration` module exists only in `sub-Ca-EEG2-1_ses-Week.nwb`, so the",
        "archive provides matched cells for one of the two mice.",
        "",
        f"- {cellreg['n_tables']} tables, one per offline recording, for example `{cellreg['example']}`",
        f"- columns of that table: {cellreg['example_columns']}",
        f"- global cells listed: {cellreg['example_global_cells']}",
        f"- cells found per session: {cellreg['example_per_session']}",
        f"- cells present in every column of that table: {cellreg['example_all_columns']}",
        "",
        "Each table registers the same three behavioural sessions independently, so the tables",
        "can be compared against each other:",
        "",
        f"- cells matched across all three behavioural sessions, per table: "
        f"{counts[0]} to {counts[-1]} (median {int(np.median(counts))})",
        f"- distinct matched cells pooled over all {cellreg['n_tables']} tables: {cellreg['unique_triplets']}",
        f"- of those, found by at least 10 of {cellreg['n_tables']} tables: {cellreg['triplets_majority']}",
        f"- of those, found by all {cellreg['n_tables']} tables: {cellreg['triplets_unanimous']}",
        "",
        "## Fallback dandiset 001710 (Plitt)",
        "",
    ]
    report += fallback
    (RESULTS / "00_inspect.md").write_text("\n".join(report) + "\n")
    print(f"\nwrote {RESULTS / '00_inspect.md'}")


if __name__ == "__main__":
    main()
