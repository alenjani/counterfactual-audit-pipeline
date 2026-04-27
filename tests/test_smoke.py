"""Smoke tests — verify imports and core wiring without GPU or network."""
from __future__ import annotations


def test_imports():
    import cap
    from cap import analysis, auditors, generator, validator, viz, utils  # noqa: F401

    assert cap.__version__


def test_config_load(tmp_path):
    import yaml

    from cap.utils import load_config

    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text(yaml.safe_dump({"a": {"b": {"c": 42}}, "x": 1}))
    cfg = load_config(cfg_path)
    assert cfg["a.b.c"] == 42
    assert cfg["x"] == 1
    assert cfg.get("missing", default="d") == "d"


def test_config_overrides(tmp_path):
    import yaml

    from cap.utils import load_config

    cfg_path = tmp_path / "t.yaml"
    cfg_path.write_text(yaml.safe_dump({"a": {"b": 1}}))
    cfg = load_config(cfg_path, overrides={"a.b": 99, "c.d": "new"})
    assert cfg["a.b"] == 99
    assert cfg["c.d"] == "new"


def test_seeding():
    import random

    from cap.utils import set_global_seed

    set_global_seed(123)
    a = random.random()
    set_global_seed(123)
    b = random.random()
    assert a == b


def test_prompt_construction():
    from cap.generator.prompts import build_demographic_prompt

    prompt = build_demographic_prompt(
        base_attributes={"pose": "frontal", "expression": "neutral expression"},
        demographic_attributes={"skin_tone": 4, "gender": "female", "age": 30},
    )
    assert "frontal" in prompt
    assert "type IV" in prompt
    assert "female" in prompt
    assert "30s" in prompt


def test_auditor_registry():
    from cap.auditors.registry import build_auditors

    # Validates registry has all expected entries — does not instantiate (avoids API/SDK init)
    from cap.auditors.registry import _REGISTRY

    expected = {
        "aws_rekognition", "azure_face", "google_vision", "face_plus_plus",
        "deepface", "insightface", "arcface",
    }
    assert expected.issubset(_REGISTRY.keys())


def test_run_manifest(tmp_path):
    from cap.utils import RunManifest

    m = RunManifest.create(run_id="test", config_path="x.yaml", config_resolved={"a": 1})
    m.finish()
    out = tmp_path / "manifest.json"
    m.write(out)
    assert out.exists()
    import json

    data = json.loads(out.read_text())
    assert data["run_id"] == "test"
    assert data["finished_at"] is not None


def test_fairness_metrics():
    import pandas as pd

    from cap.analysis.fairness_metrics import counterfactual_flip_rate, intersectional_error_table

    df = pd.DataFrame({
        "seed_identity_id": ["a", "a", "b", "b", "c", "c"],
        "prediction": ["male", "female", "male", "male", "female", "female"],
        "ground_truth": ["male"] * 6,
        "skin_tone": [1, 6, 1, 6, 1, 6],
        "gender": ["male", "male", "male", "male", "male", "male"],
    })
    fr = counterfactual_flip_rate(df)
    assert fr.iloc[0]["flip_rate"] == 1 / 3  # only "a" flipped

    tbl = intersectional_error_table(df)
    assert "error_rate" in tbl.columns


def test_viz_theme_applies():
    import matplotlib.pyplot as plt

    from cap.viz.theme import apply_paper_style

    apply_paper_style()
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    assert plt.rcParams["pdf.fonttype"] == 42
    plt.close(fig)


def test_intersectional_heatmap():
    import pandas as pd

    from cap.viz.intersectional_heatmap import intersectional_error_heatmap

    df = pd.DataFrame({
        "skin_tone": [1, 1, 2, 2, 3, 3],
        "gender": ["m", "f"] * 3,
        "error_rate": [0.01, 0.05, 0.02, 0.10, 0.03, 0.20],
    })
    fig = intersectional_error_heatmap(df)
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)
