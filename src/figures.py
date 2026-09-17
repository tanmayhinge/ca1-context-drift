"""One place to save a figure, so every phase produces the same two files.

The PNG is for reading in the repository and on GitHub. The PDF is what the report
includes, because vector text stays sharp when printed.
"""

from __future__ import annotations

from pathlib import Path


def save_figure(figure, path: Path, config: dict) -> None:
    """Write both a PNG at the configured resolution and a PDF beside it."""
    path = Path(path)
    figure.savefig(path.with_suffix(".png"), dpi=config["report"]["figure_dpi"])
    figure.savefig(path.with_suffix(".pdf"))
