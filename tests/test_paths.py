from pathlib import Path

from mimir_decide import paths


def test_data_root_follows_the_environment_variable(monkeypatch, tmp_path):
    monkeypatch.delenv("MIMIR_DECIDE_DATA", raising=False)
    assert paths.data_root() == Path("~/mimir-decide-data").expanduser()
    monkeypatch.setenv("MIMIR_DECIDE_DATA", str(tmp_path / "persistent"))
    assert paths.data_root() == tmp_path / "persistent"


def test_resolve_replaces_the_placeholder_everywhere(monkeypatch, tmp_path):
    monkeypatch.setenv("MIMIR_DECIDE_DATA", str(tmp_path))
    out = paths.resolve({"a": "{DATA_ROOT}/x", "b": ["{DATA_ROOT}/y", 3], "c": {"d": "no placeholder"}, "e": None})
    assert out == {"a": f"{tmp_path}/x", "b": [f"{tmp_path}/y", 3], "c": {"d": "no placeholder"}, "e": None}


def test_repository_configs_use_the_data_root(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("MIMIR_DECIDE_DATA", str(tmp_path / "data"))
    assert paths.load_config(root / "configs" / "data.yaml")["output_dir"] == str(tmp_path / "data" / "mixture-v0")
    assert paths.load_config(root / "configs" / "data.yaml")["cache_dir"] == str(tmp_path / "data" / "cache")
    for name in ("train_slot.yaml", "train_baseline.yaml"):
        assert paths.load_config(root / "configs" / name)["data_dir"] == str(tmp_path / "data" / "mixture-v0")
    monkeypatch.delenv("MIMIR_DECIDE_DATA")
    assert paths.load_config(root / "configs" / "data.yaml")["output_dir"] == str(Path("~/mimir-decide-data/mixture-v0").expanduser())
