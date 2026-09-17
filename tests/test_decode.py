"""Tests for the decoder, its cross validation and its baselines."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import decode  # noqa: E402
import load  # noqa: E402

CONFIG = load.load_config()


def test_blocks_are_contiguous_and_cover_everything():
    """Folds must be stretches of consecutive time, not scattered frames."""
    blocks = decode.contiguous_blocks(97, 5)
    assert len(blocks) == 5
    assert sorted(np.concatenate(blocks).tolist()) == list(range(97))
    for block in blocks:
        assert np.array_equal(block, np.arange(block[0], block[-1] + 1))


def test_equalise_gives_every_session_the_same_number_of_bins():
    features = {"a": np.zeros((300, 5)), "b": np.zeros((120, 5))}
    equalised = decode.equalise(features)
    assert {name: value.shape[0] for name, value in equalised.items()} == {"a": 120, "b": 120}


def test_within_session_standardising_removes_the_session_average():
    binned = np.array([[1.0, 2, 3], [10, 10, 16]])
    features = decode.standardise_within_session(binned)
    assert features.shape == (3, 2)
    assert features.mean(axis=0) == pytest.approx([0, 0], abs=1e-12)


def test_a_silent_cell_survives_standardising():
    binned = np.array([[1.0, 2, 3], [5, 5, 5]])
    assert np.all(np.isfinite(decode.standardise_within_session(binned)))


def test_separable_data_is_decoded_and_unrelated_data_is_not():
    rng = np.random.default_rng(CONFIG["analysis"]["seed"])
    n_bins, n_cells = 120, 30
    shared = {"one": rng.standard_normal((n_bins, n_cells)),
              "two": rng.standard_normal((n_bins, n_cells))}
    labels = {"one": "neutral", "two": "shock"}

    # with no difference between the sessions, the score must sit at chance
    noise_score = decode.block_cross_validation(shared, labels, CONFIG)
    assert 0.35 < noise_score < 0.65

    # give one session a genuine offset and it should become easy
    separable = {"one": shared["one"], "two": shared["two"] + 1.5}
    assert decode.block_cross_validation(separable, labels, CONFIG) > 0.9


def test_shuffled_labels_score_at_chance():
    """The decoder must not find structure once the labels are meaningless."""
    rng = np.random.default_rng(CONFIG["analysis"]["seed"])
    features = {"one": rng.standard_normal((100, 20)),
                "two": rng.standard_normal((100, 20)) + 1.5}
    labels = {"one": "neutral", "two": "shock"}
    scores = decode.shuffled_cross_validation(features, labels, CONFIG, rng, 20)
    assert np.mean(scores) == pytest.approx(0.5, abs=0.08)
