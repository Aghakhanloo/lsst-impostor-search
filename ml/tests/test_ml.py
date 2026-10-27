"""Fast checks of the ML prototype (skipped if PyTorch is not installed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulate import simulate_object, CLASSES, mag_to_njy, luptitude   # noqa: E402


def test_simulated_objects_are_valid():
    rng = np.random.default_rng(3)
    for label in range(len(CLASSES)):
        o = simulate_object(rng, label)
        assert o["label"] == label and len(o["t"]) > 20
        assert o["t"].min() == 0 and np.all(np.diff(o["t"]) >= 0)          # times start at 0, sorted
        assert np.isfinite(o["diff_flux"]).all() and np.all(o["sigma"] > 0)


def test_luptitude_matches_magnitude_for_bright_sources():
    f = mag_to_njy(18.0)
    assert abs(luptitude(np.array([f]), np.array([f / 1000]))[0] - 18.0) < 0.01


def test_oracle_architecture_and_conversion():
    torch = pytest.importorskip("torch")
    from oracle_transfer import OracleLiteGRU, ImpostorClassifier, make_oracle_dataset, load_oracle
    d = make_oracle_dataset({0: 3, 1: 3, 2: 3}, seed=2)
    assert d["ts"].shape[2] == 5 and np.all(d["lengths"] > 20)
    L = d["lengths"][0]
    assert set(np.unique(d["ts"][0, :L, 4])) <= {0.0, 1.0}            # detection flag
    weights = Path(__file__).resolve().parents[1] / "pretrained" / "oracle1_elasticc_lite.pth"
    enc = load_oracle(weights) if weights.exists() else OracleLiteGRU()
    model = ImpostorClassifier(enc)
    out = model(torch.as_tensor(d["ts"]), torch.as_tensor(d["lengths"]), torch.as_tensor(d["ctx"]))
    assert out.shape == (9, 3) and torch.isfinite(out).all()
