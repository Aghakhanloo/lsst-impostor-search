"""Fast checks of the ML prototype (skipped if PyTorch is not installed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulate import make_dataset, CLASSES, mag_to_njy, luptitude   # noqa: E402


def test_dataset_shapes_and_labels():
    d = make_dataset(20, seed=3)
    n = 20 * len(CLASSES)
    assert d["x"].shape == (n, 256, 2) and d["mask"].shape == (n, 256)
    assert np.bincount(d["y"]).tolist() == [20, 20, 20]
    assert np.all(d["mask"].sum(1) > 20)
    assert np.isfinite(d["x"][d["mask"]]).all() and np.isfinite(d["ctx"]).all()


def test_luptitude_matches_magnitude_for_bright_sources():
    f = mag_to_njy(18.0)
    assert abs(luptitude(np.array([f]), np.array([f / 1000]))[0] - 18.0) < 0.01


def test_model_forward():
    torch = pytest.importorskip("torch")
    from model import LightCurveTransformer
    d = make_dataset(4, seed=1)
    m = LightCurveTransformer()
    out = m(torch.as_tensor(d["t"]), torch.as_tensor(d["band"]), torch.as_tensor(d["x"]),
            torch.as_tensor(d["mask"]), torch.as_tensor(d["ctx"]))
    assert out.shape == (12, 3) and torch.isfinite(out).all()


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
