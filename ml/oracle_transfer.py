"""Transfer learning from ORACLE-1 (Shah et al. 2025) to SN impostor classification.

ORACLE-1 "lite" is a 2-layer GRU trained on ~0.5M simulated LSST alert light
curves (ELAsTiCC) to classify 19 transient and variable classes. It has no
impostor or LBV class. Here we reuse its light-curve encoder and fine-tune it to
separate impostors from supernovae and AGN-like variability.

The network below reproduces ORACLE-1's GRU layer by layer (same layer names and
shapes), so the public weights load directly. Weights: models/ELAsTiCC-lite/
revived-star-159/best_model_f1.pth in github.com/dev-ved30/Oracle (modified MIT
license); fetch them with ml/get_oracle_weights.sh.

Input per observation, as in ORACLE's ELAsTiCC loader (no rescaling):
    [days since first observation, FLUXCAL, FLUXCALERR, band mean wavelength (micron), detection flag]
FLUXCAL is difference-image flux with AB zero point 27.5.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

ORACLE_NODES = [
    "Alert", "Transient", "Variable", "SN", "Fast", "Long", "Periodic", "AGN",
    "SNIa", "SNIb/c", "SNIax", "SNI91bg", "SNII",
    "KN", "Dwarf Novae", "uLens", "M-dwarf Flare",
    "SLSN", "TDE", "ILOT", "CART", "PISN",
    "Cepheid", "RR Lyrae", "Delta Scuti", "EB",
]
# sibling groups of the taxonomy, in level order (for ORACLE's conditional softmax)
SIBLINGS = [[1, 2], [3, 4, 5], [6, 7], [8, 9, 10, 11, 12], [13, 14, 15, 16],
            [17, 18, 19, 20, 21], [22, 23, 24, 25]]
LEVEL2 = [3, 4, 5, 6, 7]                       # SN, Fast, Long, Periodic, AGN
BAND_WAVELENGTH = {"g": (400 + 552) / 2000, "r": (552 + 691) / 2000, "i": (691 + 818) / 2000}
NJY_TO_FLUXCAL = 10 ** (-0.4 * (31.4 - 27.5))  # nJy (ZP 31.4) -> SNANA FLUXCAL (ZP 27.5)
WEIGHTS_SHA256 = "46f915c48b03bc6324435873bad9e249fdc9f242f8974472d9cd55d9e560b4e2"


class OracleLiteGRU(nn.Module):
    """ORACLE-1 lite: GRU(5 -> 100, 2 layers) -> 64 -> 32 -> 16 -> 26 taxonomy nodes."""

    def __init__(self, n_out=26):
        super().__init__()
        self.gru = nn.GRU(input_size=5, hidden_size=100, num_layers=2, batch_first=True)
        self.dense1 = nn.Linear(100, 64)
        self.dense2 = nn.Linear(64, 32)
        self.dense3 = nn.Linear(32, 16)
        self.fc_out = nn.Linear(16, n_out)
        self.tanh, self.relu = nn.Tanh(), nn.ReLU()

    def embed(self, ts, lengths):
        packed = pack_padded_sequence(ts, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        x = self.tanh(self.dense1(hidden[-1]))
        x = self.tanh(self.dense2(x))
        return self.dense3(x)                     # 16-d light-curve embedding

    def forward(self, ts, lengths):
        return self.fc_out(self.relu(self.embed(ts, lengths)))


def load_oracle(path, verify=True):
    if verify:
        import hashlib
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if h != WEIGHTS_SHA256:
            raise ValueError(f"unexpected ORACLE weights checksum {h}")
    model = OracleLiteGRU()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model


def conditional_probabilities(logits):
    """ORACLE's hierarchical output: softmax within each sibling group."""
    p = torch.zeros_like(logits)
    for g in SIBLINGS:
        p[:, g] = torch.softmax(logits[:, g], dim=1)
    return p


class ImpostorClassifier(nn.Module):
    """ORACLE encoder (pretrained or random) + a new head with host context."""

    def __init__(self, encoder, n_ctx=2, n_classes=3, hidden=32):
        super().__init__()
        self.encoder = encoder
        self.ctx = nn.Sequential(nn.Linear(n_ctx, 16), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(16 + 16, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, ts, lengths, ctx):
        z = torch.relu(self.encoder.embed(ts, lengths))
        return self.head(torch.cat([z, self.ctx(ctx)], dim=1))


# ---------------------------------------------------------------- data
def make_oracle_dataset(counts, seed=0, max_len=256):
    """Simulate objects and convert them to ORACLE's input format.

    counts: number of objects per class, e.g. {0: 100, 1: 4000, 2: 4000}.
    """
    from simulate import simulate_object, BANDS
    rng = np.random.default_rng(seed)
    labels = np.concatenate([np.full(n, k) for k, n in sorted(counts.items())])
    rng.shuffle(labels)
    n = len(labels)
    ts = np.full((n, max_len, 5), -9.0, np.float32)           # ORACLE's padding value
    lengths = np.zeros(n, np.int64)
    ctx = np.zeros((n, 2), np.float32)
    meta = {k: np.full(n, np.nan) for k in ("mu", "amplitude", "period")}
    subtype = np.empty(n, dtype=object)
    wl = np.array([BAND_WAVELENGTH[b] for b in BANDS])
    for i, lab in enumerate(labels):
        o = simulate_object(rng, int(lab))
        L = min(len(o["t"]), max_len)
        flux = o["diff_flux"][:L] * NJY_TO_FLUXCAL
        err = o["sigma"][:L] * NJY_TO_FLUXCAL
        ts[i, :L] = np.stack([o["t"][:L], flux, err, wl[o["band"][:L]],
                              (np.abs(flux) / err >= 5).astype(float)], axis=1)
        lengths[i] = L
        med = np.median(o["lup"][o["band"] == 1])
        ctx[i] = [(med - o["mu"] + 12.0) / 4.0, (o["mu"] - 30.75) / 1.3]
        for k in meta:
            meta[k][i] = o[k]
        subtype[i] = o["subtype"]
    return dict(ts=ts, lengths=lengths, ctx=ctx, y=labels.astype(np.int64), subtype=subtype, **meta)
