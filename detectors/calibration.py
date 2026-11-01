"""
calibration.py - turn the raw score of each detector head into one chance.

    P(fake) = sigmoid(w1 * logit(score1) + w2 * logit(score2) + ... + bias)

This is logistic regression on the heads' scores. The weights are fitted on
validation clips by eval/calibrate.py, separately for wideband and phone
audio, and saved in detectors/calibration.json. With a single head it is
ordinary Platt calibration.

Why more than one head: the original head and a fine-tuned head make
different mistakes, so together they do better than either alone.
"""
import numpy as np


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def combine(scores, params):
    """scores: one value per head, or a table with one column per head (for
    example one row per clip, or per window). Returns the chance(s) of fake."""
    return 1 / (1 + np.exp(-(logit(scores) @ np.array(params["weights"]) + params["bias"])))
