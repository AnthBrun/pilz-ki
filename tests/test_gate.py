import numpy as np
import pytest

pytest.importorskip("torch")

from pilzki.gate import fit_logistic, gate_scores, held_out  # noqa: E402
from pilzki.predict import Predictor  # noqa: E402


def test_fit_logistic_separates_and_folds_standardisation():
    rng = np.random.default_rng(0)
    pos = rng.normal(5.0, 2.0, (300, 8)) + np.array([3, 0, 0, 0, 0, 0, 0, 0])
    neg = rng.normal(5.0, 2.0, (500, 8)) - np.array([3, 0, 0, 0, 0, 0, 0, 0])
    w, b = fit_logistic(pos, neg)
    # weights act on raw (unstandardised) features, as in the exported model
    assert gate_scores(pos, w, b).mean() > 0.85
    assert gate_scores(neg, w, b).mean() < 0.15


def test_held_out_is_stable_and_about_a_fifth():
    from pathlib import Path

    names = [Path(f"img{i}.jpg") for i in range(2000)]
    share = np.mean([held_out(p) for p in names])
    assert 0.15 < share < 0.25
    assert held_out(Path("img7.jpg")) == held_out(Path("/elsewhere/img7.jpg"))


def make_predictor(threshold):
    pred = Predictor.__new__(Predictor)
    pred.meta = {
        "species": ["A", "B"],
        "tags": [{"tag": "hutpilz", "threshold": 0.5}],
        "groups": [{"name": "gruppe", "exclusive": True, "requires": None, "tags": ["hutpilz"]}],
        "require_p": 0.5,
        "gate": {"output": "fungus_prob", "threshold": threshold},
    }
    return pred


def test_describe_respects_gate():
    pred = make_predictor(0.3)
    tags, species = np.array([0.9]), np.array([0.8, 0.2])
    assert pred.describe(tags, species, 0.9)["prefill"] == ["hutpilz"]
    rejected = pred.describe(tags, species, 0.1)
    assert rejected["fungus"] is False and rejected["prefill"] == [] and rejected["species"] == []


def test_find_averages_only_photos_that_pass():
    pred = make_predictor(0.3)
    tags = np.array([[0.9], [0.0]])          # the rejected photo would pull hutpilz below 0.5
    species = np.array([[0.8, 0.2], [0.1, 0.9]])
    d = pred.describe_find(tags, species, np.array([0.9, 0.1]))
    assert d["photos_used"] == 1 and d["prefill"] == ["hutpilz"] and d["species"][0][0] == "A"
    assert pred.describe_find(tags, species, np.array([0.1, 0.2]))["fungus"] is False
