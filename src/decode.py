"""Training a decoder on two sessions and asking what it says about later days.

The design of the test matters more than the classifier. Training on the neutral session
of day 1 against the conditioning session of day 3 lets the decoder use anything that
separates those two recordings, context included but also the simple passage of time. The
later sessions are what pull those apart:

    Recall1, day 4, shock context  - both context and elapsed time predict "shock",
                                     so it cannot distinguish the two accounts
    Recall3, day 6, neutral context - context predicts "neutral", while elapsed time
                                     predicts "shock", because day 6 is nearer day 3
                                     than day 1. This one is diagnostic.
    Recall2, day 5, novel context  - no correct answer, so we report which way it leans

Frames are binned to one second and each cell is standardised within its own session, so
that a day when the whole population is simply brighter cannot be read as a context.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


def standardise_within_session(binned: np.ndarray) -> np.ndarray:
    """Cells by bins to bins by cells, with each cell centred inside this session.

    Removing the session mean is what makes the test about the pattern across cells
    rather than about how active the animal's hippocampus happened to be that day.
    """
    centred = binned - binned.mean(axis=1, keepdims=True)
    spread = binned.std(axis=1, keepdims=True)
    spread[spread == 0] = np.inf
    return (centred / spread).T


def raw_features(binned: np.ndarray) -> np.ndarray:
    """The same data without the within session step, for the variant analysis."""
    return binned.T


def contiguous_blocks(n_samples: int, folds: int) -> list[np.ndarray]:
    """Split a recording into equal blocks of consecutive bins.

    Neighbouring bins come from the same few seconds of behaviour, so a random split
    would put nearly identical samples in training and testing and inflate accuracy.
    """
    edges = np.linspace(0, n_samples, folds + 1).astype(int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(folds)]


def equalise(session_features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Give every training session the same number of time bins.

    The neutral session lasts ten minutes and the pre-shock part of conditioning only two,
    so without this the decoder sees two and a half times as many neutral bins. It would
    then be rewarded for simply answering "neutral", and the accuracy of a decoder that
    has learned nothing would sit well above one half.
    """
    shortest = min(features.shape[0] for features in session_features.values())
    return {name: features[:shortest] for name, features in session_features.items()}


def fit(features: np.ndarray, labels: np.ndarray, config: dict):
    """A standardiser and a logistic regression, fitted on the training bins only."""
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(C=config["decoding"]["regularisation_C"], max_iter=5000,
                               class_weight="balanced")
    model.fit(scaler.transform(features), labels)
    return scaler, model


def predict(fitted, features: np.ndarray) -> np.ndarray:
    scaler, model = fitted
    return model.predict(scaler.transform(features))


def block_cross_validation(session_features: dict[str, np.ndarray],
                           session_labels: dict[str, str], config: dict) -> float:
    """Accuracy within the training days, holding out one time block at a time."""
    folds = config["decoding"]["block_cv_folds"]
    blocks = {name: contiguous_blocks(features.shape[0], folds)
              for name, features in session_features.items()}

    predictions, expected = [], []
    for fold in range(folds):
        train_x, train_y, test_x, test_y = [], [], [], []
        for name, features in session_features.items():
            held = blocks[name][fold]
            keep = np.setdiff1d(np.arange(features.shape[0]), held)
            train_x.append(features[keep])
            train_y += [session_labels[name]] * len(keep)
            test_x.append(features[held])
            test_y += [session_labels[name]] * len(held)
        fitted = fit(np.vstack(train_x), np.array(train_y), config)
        predictions.append(predict(fitted, np.vstack(test_x)))
        expected.append(np.array(test_y))
    # balanced accuracy averages the accuracy of each class, so an uneven number of
    # test bins between the two sessions cannot flatter the score
    return float(balanced_accuracy_score(np.concatenate(expected),
                                         np.concatenate(predictions)))


def shuffled_cross_validation(session_features: dict[str, np.ndarray],
                              session_labels: dict[str, str], config: dict,
                              rng: np.random.Generator, n_shuffles: int) -> list[float]:
    """The same measurement with the labels scrambled, which is the empirical chance level.

    Labels are shuffled across all the training bins together, so any real difference
    between the two sessions is destroyed while the data keep their structure.
    """
    pooled = np.vstack(list(session_features.values()))
    labels = np.concatenate([[session_labels[name]] * features.shape[0]
                             for name, features in session_features.items()])
    sizes = [features.shape[0] for features in session_features.values()]

    scores = []
    for _ in range(n_shuffles):
        scrambled = rng.permutation(labels)
        pieces, start = {}, 0
        fake_labels = {}
        for (name, _), size in zip(session_features.items(), sizes):
            pieces[name] = pooled[start:start + size]
            # every bin of this block keeps its scrambled label, so we rebuild per bin
            fake_labels[name] = scrambled[start:start + size]
            start += size
        scores.append(_cv_with_per_bin_labels(pieces, fake_labels, config))
    return scores


def _cv_with_per_bin_labels(session_features: dict[str, np.ndarray],
                            per_bin_labels: dict[str, np.ndarray], config: dict) -> float:
    """Block cross validation where every bin carries its own label."""
    folds = config["decoding"]["block_cv_folds"]
    blocks = {name: contiguous_blocks(features.shape[0], folds)
              for name, features in session_features.items()}

    guessed, expected = [], []
    for fold in range(folds):
        train_x, train_y, test_x, test_y = [], [], [], []
        for name, features in session_features.items():
            held = blocks[name][fold]
            keep = np.setdiff1d(np.arange(features.shape[0]), held)
            train_x.append(features[keep])
            train_y.append(per_bin_labels[name][keep])
            test_x.append(features[held])
            test_y.append(per_bin_labels[name][held])
        train_y = np.concatenate(train_y)
        if len(np.unique(train_y)) < 2:
            continue
        fitted = fit(np.vstack(train_x), train_y, config)
        guessed.append(predict(fitted, np.vstack(test_x)))
        expected.append(np.concatenate(test_y))
    if not expected:
        return float("nan")
    return float(balanced_accuracy_score(np.concatenate(expected), np.concatenate(guessed)))
