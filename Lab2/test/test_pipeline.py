import json
import os

import pytest
from src import promote_model
from src.data import load_split
from src.evaluate_model import evaluate, passes_gate
from src.train_model import MODEL_TYPES, build_model, train


def test_split_is_fixed():
    # Same test rows every time, so runs are comparable
    _, X_test_a, _, y_test_a = load_split()
    _, X_test_b, _, y_test_b = load_split()
    assert (X_test_a == X_test_b).all()
    assert (y_test_a == y_test_b).all()


def test_split_has_no_overlap():
    X_train, X_test, _, _ = load_split()
    assert len(X_train) + len(X_test) == 569
    assert len(X_test) == 114


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_each_model_trains_and_scores(model_type):
    model = train(model_type, n_estimators=10)
    metrics = evaluate(model)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0
    # Every supported model should clear a low bar on this dataset
    assert metrics["f1"] > 0.85


def test_unknown_model_type_raises():
    with pytest.raises(ValueError):
        build_model("svm")


def test_quality_gate():
    assert passes_gate({"f1": 0.95}, 0.90)
    assert passes_gate({"f1": 0.90}, 0.90)
    assert not passes_gate({"f1": 0.89}, 0.90)


def _registry_with_champion(f1):
    return {"champion": {"version": "old", "metrics": {"f1": f1}}, "history": []}


@pytest.mark.parametrize(
    "challenger_f1, registry, force, expected",
    [
        (0.90, {"champion": None, "history": []}, False, True),  # first model
        (0.97, _registry_with_champion(0.95), False, True),      # better
        (0.95, _registry_with_champion(0.95), False, False),     # tie keeps champion
        (0.93, _registry_with_champion(0.95), False, False),     # worse
        (0.93, _registry_with_champion(0.95), True, True),       # forced
    ],
)
def test_should_promote(challenger_f1, registry, force, expected):
    decision, _ = promote_model.should_promote(challenger_f1, registry, force)
    assert decision is expected


def test_promote_updates_registry(tmp_path, monkeypatch):
    build = tmp_path / "build"
    build.mkdir()
    (build / "model_v1.joblib").write_bytes(b"fake model")
    (build / "metrics_v1.json").write_text(json.dumps({"f1": 0.96}))
    (build / "params_v1.json").write_text(json.dumps({"version": "v1", "model_type": "random_forest"}))
    monkeypatch.chdir(tmp_path)

    promote_model.promote("v1", "build")

    registry = promote_model.load_registry()
    assert registry["champion"]["version"] == "v1"
    assert len(registry["history"]) == 1
    assert os.path.exists("models/model_v1.joblib")
    assert os.path.exists("metrics/metrics_v1.json")
