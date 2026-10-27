"""Transfer learning from ORACLE-1 (Shah et al. 2025,
https://doi.org/10.3847/1538-4357/ae1130) to SN impostor classification.

ORACLE-1 "lite" is a 2-layer GRU trained on ~450,000 simulated LSST alert
light curves (ELAsTiCC) to classify 19 transient and variable classes. Its
closest class, ILOT, represents a single generic outburst; it has no class for
recurring eruptions of massive stars. Here we reuse its light-curve encoder and fine-tune it to
separate SN impostors from supernovae and AGN-like variability.

The network below reproduces ORACLE-1's GRU layer by layer (same layer names and
shapes), so the public weights load directly. Weights: models/ELAsTiCC-lite/
revived-star-159/best_model_f1.pth in github.com/dev-ved30/Oracle (modified MIT
license); fetch them with ml/get_oracle_weights.sh.

Input per observation, as in ORACLE's ELAsTiCC loader (no rescaling):
    [days since first observation, FLUXCAL, FLUXCALERR, band mean wavelength (micron), detection flag]
FLUXCAL is difference-image flux with AB zero point 27.5.

How to read this file: a "class" defines a kind of object (here, a network);
each "def" is a function. See ml/figures/training_modes.png for a diagram.
"""
import numpy as np                       # arrays of numbers
import torch                             # PyTorch: the neural-network library
import torch.nn as nn                    # PyTorch's building blocks (layers)
from torch.nn.utils.rnn import pack_padded_sequence   # lets the GRU skip empty padding slots

# ---- ORACLE's class tree: 26 boxes ("nodes"), from broad to specific ----
ORACLE_NODES = [
    "Alert", "Transient", "Variable", "SN", "Fast", "Long", "Periodic", "AGN",
    "SNIa", "SNIb/c", "SNIax", "SNI91bg", "SNII",
    "KN", "Dwarf Novae", "uLens", "M-dwarf Flare",
    "SLSN", "TDE", "ILOT", "CART", "PISN",
    "Cepheid", "RR Lyrae", "Delta Scuti", "EB",
]
# sibling groups of the taxonomy, in level order (for ORACLE's conditional softmax):
# boxes in the same group compete with each other (e.g. Transient vs Variable)
SIBLINGS = [[1, 2], [3, 4, 5], [6, 7], [8, 9, 10, 11, 12], [13, 14, 15, 16],
            [17, 18, 19, 20, 21], [22, 23, 24, 25]]
LEVEL2 = [3, 4, 5, 6, 7]                       # SN, Fast, Long, Periodic, AGN (used for the zero-shot test)
# central wavelength of each band in microns: midpoint of the filter edges (nm) / 1000
BAND_WAVELENGTH = {"g": (400 + 552) / 2000, "r": (552 + 691) / 2000, "i": (691 + 818) / 2000}
NJY_TO_FLUXCAL = 10 ** (-0.4 * (31.4 - 27.5))  # nJy (ZP 31.4) -> SNANA FLUXCAL (ZP 27.5), ORACLE's flux unit
WEIGHTS_SHA256 = "46f915c48b03bc6324435873bad9e249fdc9f242f8974472d9cd55d9e560b4e2"   # expected file fingerprint


class OracleLiteGRU(nn.Module):
    """ORACLE-1 lite: GRU(5 -> 100, 2 layers) -> 64 -> 32 -> 16 -> 26 taxonomy nodes."""

    def __init__(self, n_out=26):
        super().__init__()
        # GRU: reads the light curve one observation at a time (5 numbers each) and keeps a
        # running "memory" of 100 numbers; two GRU layers are stacked
        self.gru = nn.GRU(input_size=5, hidden_size=100, num_layers=2, batch_first=True)
        # dense layers squeeze the memory: 100 -> 64 -> 32 -> 16 numbers
        self.dense1 = nn.Linear(100, 64)
        self.dense2 = nn.Linear(64, 32)
        self.dense3 = nn.Linear(32, 16)
        self.fc_out = nn.Linear(16, n_out)        # 16 numbers -> scores for ORACLE's 26 class boxes
        self.tanh, self.relu = nn.Tanh(), nn.ReLU()   # squashing functions between layers

    def embed(self, ts, lengths):
        """The ENCODER: light curve -> 16-number summary of its shape."""
        # tell the GRU where each light curve really ends, so it ignores the padding
        packed = pack_padded_sequence(ts, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        x = self.tanh(self.dense1(hidden[-1]))    # hidden[-1]: memory after the last observation
        x = self.tanh(self.dense2(x))             # tanh squashes values into -1..+1
        return self.dense3(x)                     # 16-d light-curve embedding

    def forward(self, ts, lengths):
        """The original ORACLE: summary -> 26 class scores (used only for the zero-shot test)."""
        return self.fc_out(self.relu(self.embed(ts, lengths)))


def load_oracle(path, verify=True):
    """Build the network and fill it with ORACLE's published weights."""
    if verify:                                    # check the file's fingerprint first
        import hashlib
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if h != WEIGHTS_SHA256:
            raise ValueError(f"unexpected ORACLE weights checksum {h}")
    model = OracleLiteGRU()                       # empty network with ORACLE's shape
    state = torch.load(path, map_location="cpu", weights_only=True)   # read the weights file
    # strict=True: every layer name and size must match exactly, proving the rebuild is identical
    model.load_state_dict(state, strict=True)
    return model


def conditional_probabilities(logits):
    """ORACLE's hierarchical output: softmax within each sibling group.

    softmax turns scores into probabilities that add up to 1 within each group,
    e.g. P(SN) + P(Fast) + P(Long) + P(Periodic) + P(AGN) = 1.
    """
    p = torch.zeros_like(logits)
    for g in SIBLINGS:
        p[:, g] = torch.softmax(logits[:, g], dim=1)
    return p


class ImpostorClassifier(nn.Module):
    """ORACLE encoder (pretrained or random) + a new head with host context."""

    def __init__(self, encoder, n_ctx=2, n_classes=3, hidden=32):
        super().__init__()
        self.encoder = encoder                    # ORACLE's encoder (pretrained or random weights)
        # turns the 2 host numbers (median abs. mag, distance) into 16 numbers;
        # ReLU keeps positive values and sets negative ones to 0
        self.ctx = nn.Sequential(nn.Linear(n_ctx, 16), nn.ReLU())
        # the HEAD: 16 light-curve + 16 host numbers -> 32 -> 3 scores (SN impostor, supernova, AGN)
        self.head = nn.Sequential(nn.Linear(16 + 16, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, ts, lengths, ctx):
        z = torch.relu(self.encoder.embed(ts, lengths))   # light curve -> 16-number summary
        return self.head(torch.cat([z, self.ctx(ctx)], dim=1))   # join with host info -> decision


# ---------------------------------------------------------------- data
def make_oracle_dataset(counts, seed=0, max_len=256):
    """Simulate objects and convert them to ORACLE's input format.

    counts: number of objects per class, e.g. {0: 100, 1: 4000, 2: 4000}.
    """
    from simulate import simulate_object, BANDS
    rng = np.random.default_rng(seed)                          # same seed gives the same data
    labels = np.concatenate([np.full(n, k) for k, n in sorted(counts.items())])   # list of labels
    rng.shuffle(labels)                                        # in random order
    n = len(labels)
    ts = np.full((n, max_len, 5), -9.0, np.float32)           # ORACLE's padding value
    lengths = np.zeros(n, np.int64)                            # real number of points per object
    ctx = np.zeros((n, 2), np.float32)                         # 2 host numbers per object
    meta = {k: np.full(n, np.nan) for k in ("mu", "amplitude", "period")}   # true parameters
    subtype = np.empty(n, dtype=object)
    wl = np.array([BAND_WAVELENGTH[b] for b in BANDS])         # wavelength for g, r, i
    for i, lab in enumerate(labels):                           # make each object
        o = simulate_object(rng, int(lab))
        L = min(len(o["t"]), max_len)                          # keep at most max_len points
        flux = o["diff_flux"][:L] * NJY_TO_FLUXCAL             # difference-image flux, in ORACLE's units
        err = o["sigma"][:L] * NJY_TO_FLUXCAL                  # its error
        # the 5 numbers per observation: time, flux, error, wavelength, detected (|flux| >= 5 sigma)
        ts[i, :L] = np.stack([o["t"][:L], flux, err, wl[o["band"][:L]],
                              (np.abs(flux) / err >= 5).astype(float)], axis=1)
        lengths[i] = L
        med = np.median(o["lup"][o["band"] == 1])              # median r-band magnitude
        # host numbers: median absolute magnitude and distance modulus, rescaled to about -1..+1
        ctx[i] = [(med - o["mu"] + 12.0) / 4.0, (o["mu"] - 30.75) / 1.3]
        for k in meta:
            meta[k][i] = o[k]
        subtype[i] = o["subtype"]
    # y holds the true answers (labels)
    return dict(ts=ts, lengths=lengths, ctx=ctx, y=labels.astype(np.int64), subtype=subtype, **meta)
