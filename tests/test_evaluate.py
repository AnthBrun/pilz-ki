import numpy as np

from pilzki.evaluate import average_precision, prefill, threshold_for_precision

GROUPS = [
    {"name": "gruppe", "exclusive": True, "requires": None, "tags": ["hutpilz", "kugel"]},
    {"name": "farbe", "exclusive": True, "requires": None, "tags": ["rot", "braun"]},
    {"name": "stiel", "exclusive": False, "requires": "hutpilz", "tags": ["ring", "netz"]},
]
TAGS = ["hutpilz", "kugel", "rot", "braun", "ring", "netz"]
IDX = {t: i for i, t in enumerate(TAGS)}


def test_prefill_exclusive_requires_and_disabled():
    thr = {t: 0.5 for t in TAGS} | {"braun": None}
    p = np.array([0.9, 0.1, 0.6, 0.7, 0.8, 0.55])
    # braun is the group argmax but disabled -> no colour at all
    assert prefill(p, GROUPS, IDX, thr, 0.5) == ["hutpilz", "ring", "netz"]
    p2 = np.array([0.3, 0.6, 0.9, 0.1, 0.9, 0.9])
    # not a hutpilz -> stem group skipped
    assert prefill(p2, GROUPS, IDX, thr, 0.5) == ["kugel", "rot"]


def test_threshold_for_precision():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    y = np.array([1, 1, 0, 1, 0])
    assert threshold_for_precision(scores, y, 1.0, 1) == 0.8
    assert threshold_for_precision(scores, y, 0.75, 1) == 0.6
    assert threshold_for_precision(scores, y, 1.0, 3) is None


def test_average_precision():
    assert average_precision(np.array([0.9, 0.1]), np.array([1, 0])) == 1.0
    assert average_precision(np.array([0.9, 0.1]), np.array([1, 1])) is None
