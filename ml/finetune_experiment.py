"""Label-efficiency experiment: does a pretrained LSST classifier help when SN impostors are rare?

For each number of labeled SN impostors N (e.g. 25 ... 4000), with plenty of
supernovae and AGN, train the same network three ways and evaluate on one fixed,
balanced test set:

  scratch     ORACLE-1 architecture, random initial weights
  pretrained  ORACLE-1 weights (trained on ELAsTiCC), all layers fine-tuned
  frozen      ORACLE-1 weights kept fixed; only the new head is trained

Each configuration is repeated with several seeds (different SN impostor subsets
and initializations). The simulator plays the role of the augmentation step:
a handful of real SN impostors would be expanded into many simulated ones.

Example (laptop, small):
    python ml/finetune_experiment.py --counts 25 100 1000 --seeds 2 --steps 600 --out ml/results_transfer_small
Example (one GPU, see ml/rivanna_job.slurm):
    python ml/finetune_experiment.py --out ml/results_transfer

What main() does, in order:
  1. simulate a training pool and a separate test set (never used for training)
  2. zero-shot test: what does the original ORACLE call the test objects?
  3. for each seed, each number of SN impostors N, and each mode: train a model
     (train_one), then score it on the test set (predict + metrics)
  4. save every result to results.csv as it goes
  5. average over seeds into summary.json and draw label_efficiency.png
Default: 3 seeds x 6 values of N x 3 modes = 54 training runs.
"""
import argparse                      # reads settings from the command line
import csv                           # writes results.csv
import json                          # writes summary.json
import sys
import time                          # measures run times
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F      # loss functions (cross_entropy)

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so Python finds oracle_transfer.py
from oracle_transfer import (OracleLiteGRU, ImpostorClassifier, load_oracle,      # noqa: E402
                             conditional_probabilities, make_oracle_dataset,
                             ORACLE_NODES)

# plot colors and legend labels for the three training modes
MODE_COLORS = {"scratch": "#eb6834", "pretrained": "#2a78d6", "frozen": "#1baf7a"}
MODE_LABELS = {"scratch": "from scratch", "pretrained": "pretrained ORACLE, fine-tuned",
               "frozen": "pretrained ORACLE, frozen"}


def parse_args():
    """Settings that can be changed from the command line, each with a default."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--counts", type=int, nargs="+", default=[25, 50, 100, 250, 1000, 4000],
                   help="numbers of labeled SN impostors to train with")
    p.add_argument("--n-other", type=int, default=4000, help="supernovae and AGN per class in training")
    p.add_argument("--n-test", type=int, default=1500, help="test objects per class")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=3000, help="optimizer steps per run")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--modes", nargs="+", default=["scratch", "pretrained", "frozen"])
    p.add_argument("--weights", default=str(Path(__file__).resolve().parent / "pretrained" / "oracle1_elasticc_lite.pth"))
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="ml/results_transfer")
    return p.parse_args()


def pick_device(name):
    """Use the GPU if there is one, otherwise the CPU."""
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def subset(d, idx):
    """Keep only the objects at positions idx."""
    return {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def to_dev(d, device):
    """Convert arrays to PyTorch tensors on the GPU (or CPU)."""
    return (torch.as_tensor(d["ts"], device=device), torch.as_tensor(d["lengths"]),
            torch.as_tensor(d["ctx"], device=device), torch.as_tensor(d["y"], device=device))


def build_model(mode, weights):
    """Make one model for the given mode (see ml/figures/training_modes.png)."""
    # pretrained/frozen: load ORACLE's weights; scratch: same network with random weights
    enc = load_oracle(weights) if mode in ("pretrained", "frozen") else OracleLiteGRU()
    model = ImpostorClassifier(enc)
    if mode == "frozen":
        for p in model.encoder.parameters():
            p.requires_grad = False          # the lock: these weights are never changed
    return model


def train_one(mode, train, args, device, seed):
    """Train one model; returns the model and the training time in seconds."""
    torch.manual_seed(seed)                       # same seed -> same random start
    model = build_model(mode, args.weights).to(device)
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = list(model.head.parameters()) + list(model.ctx.parameters())
    # learning rate ("lr") = size of each adjustment step. The new head uses 0.001;
    # the pretrained encoder uses a 5x smaller step, so ORACLE's knowledge is adjusted gently.
    groups = [{"params": head_params, "lr": 1e-3}]
    if enc_params:
        groups.append({"params": enc_params, "lr": 2e-4 if mode == "pretrained" else 1e-3})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)   # the algorithm that adjusts the weights
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)   # shrinks the step size toward the end

    ts, lengths, ctx, y = to_dev(train, device)
    # class balancing: with few SN impostors, the model could ignore them and still look accurate,
    # so mistakes on rarer classes count more
    counts = torch.bincount(y, minlength=3).float()
    weight = (counts.sum() / (3 * counts.clamp(min=1))).to(device)    # balance rare SN impostors
    gen = torch.Generator().manual_seed(seed)
    n = len(y)
    model.train()                                 # switch to training mode
    t0 = time.time()
    for step in range(args.steps):                # repeat many times:
        idx = torch.randint(0, n, (min(args.batch_size, n),), generator=gen)   # 1. pick a random batch
        logits = model(ts[idx.to(device)], lengths[idx], ctx[idx.to(device)])   # 2. the model guesses
        loss = F.cross_entropy(logits, y[idx.to(device)], weight=weight)   # 3. how wrong are the guesses?
        opt.zero_grad(set_to_none=True)
        loss.backward()                           # 4. which way to nudge every weight
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # prevent one huge nudge
        opt.step()                                # 5. nudge the weights
        sched.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return model, time.time() - t0


@torch.no_grad()
def predict(model, data, device, batch=2048):
    """Run a trained model on data; returns probabilities for SN impostor, supernova, AGN."""
    model.eval()                                  # switch to evaluation mode (no training)
    ts, lengths, ctx, _ = to_dev(data, device)
    out = []
    for k in range(0, len(lengths), batch):
        out.append(torch.softmax(model(ts[k:k + batch], lengths[k:k + batch], ctx[k:k + batch]), 1).cpu())
    return torch.cat(out).numpy()


def _rank_average(x):
    """1-based ranks with ties averaged (same as scipy.stats.rankdata)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return (sums / counts)[inv]


def auroc(score, positive):
    """Area under the ROC curve via the rank statistic (ties averaged)."""
    r = _rank_average(np.asarray(score, float))
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((r[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics(prob, y):
    """Compare the model's best guesses with the true labels (0 = SN impostor).

    recall: fraction of true SN impostors found; precision: fraction of objects called
    SN impostors that really are; F1 combines both; macro_f1 averages F1 over the 3 classes.
    """
    pred = prob.argmax(1)
    pos, call = (y == 0), (pred == 0)
    tp = np.sum(pos & call)
    rec = tp / max(pos.sum(), 1)
    prec = tp / max(call.sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    f1s = []
    for k in range(3):
        tpk = np.sum((y == k) & (pred == k))
        pk, rk = tpk / max(np.sum(pred == k), 1), tpk / max(np.sum(y == k), 1)
        f1s.append(2 * pk * rk / max(pk + rk, 1e-12))
    return dict(impostor_recall=float(rec), impostor_precision=float(prec), impostor_f1=float(f1),
                impostor_auroc=auroc(prob[:, 0], pos), macro_f1=float(np.mean(f1s)),
                accuracy=float(np.mean(pred == y)))


@torch.no_grad()
def zero_shot(weights, data):
    """What does the original ORACLE call our simulated objects? (level-2 node of the predicted branch)"""
    model = load_oracle(weights).eval()
    p = conditional_probabilities(model(torch.as_tensor(data["ts"]), torch.as_tensor(data["lengths"])))
    table = {}
    for k, name in enumerate(("impostor", "sn", "agn")):
        calls = []
        for row in p[data["y"] == k]:
            branch = [3, 4, 5] if row[1] > row[2] else [6, 7]
            calls.append(ORACLE_NODES[branch[int(row[branch].argmax())]])
        v, c = np.unique(calls, return_counts=True)
        table[name] = {str(a): round(float(b) / len(calls), 3) for a, b in zip(v, c)}
    return table


def plot(rows, out):
    """Two panels (F1 and AUROC vs N); points are means over seeds, bars the spread."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for ax, key, lab in ((axes[0], "impostor_f1", "SN impostor F1 score"),
                         (axes[1], "impostor_auroc", "SN impostor AUROC")):
        for mode in MODE_COLORS:
            r = [x for x in rows if x["mode"] == mode]
            if not r:
                continue
            ns = sorted({x["n_impostors"] for x in r})
            mean = [np.mean([x[key] for x in r if x["n_impostors"] == n]) for n in ns]
            std = [np.std([x[key] for x in r if x["n_impostors"] == n]) for n in ns]
            ax.errorbar(ns, mean, yerr=std, color=MODE_COLORS[mode], marker="o", ms=5, lw=2,
                        capsize=0, label=MODE_LABELS[mode])
        ax.set_xscale("log")
        ax.set_xlabel("labeled SN impostors in training set")
        ax.set_ylabel(lab)
        ax.grid(axis="y", color="#e4e3dc", lw=0.8)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Label efficiency on simulated LSST light curves (test set, mean and spread over seeds)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "label_efficiency.png", dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    gpu = torch.cuda.get_device_name() if device.type == "cuda" else None
    print(f"device: {device}" + (f" ({gpu})" if gpu else ""))

    t0 = time.time()
    n_imp_max = max(args.counts)
    # 1. training pool (SN impostors, supernovae, AGN) and a separate test set
    pool = make_oracle_dataset({0: n_imp_max, 1: args.n_other, 2: args.n_other}, seed=11)
    test = make_oracle_dataset({0: args.n_test, 1: args.n_test, 2: args.n_test}, seed=99)
    print(f"simulated {len(pool['y'])} training-pool and {len(test['y'])} test light curves "
          f"in {time.time() - t0:.0f} s")

    zs = zero_shot(args.weights, test)            # 2. zero-shot test of the original ORACLE
    print("zero-shot ORACLE calls on the test set:", json.dumps(zs))

    imp_idx = np.flatnonzero(pool["y"] == 0)
    other_idx = np.flatnonzero(pool["y"] != 0)
    rows = []
    for seed in range(args.seeds):                # 3. repeat with different random choices
        perm = np.random.default_rng(1000 + seed).permutation(imp_idx)   # random order of SN impostors
        for n_imp in sorted(args.counts):         # for each N (25 ... 4000)
            train = subset(pool, np.concatenate([perm[:n_imp], other_idx]))   # N SN impostors + all others
            for mode in args.modes:               # scratch, pretrained, frozen
                model, secs = train_one(mode, train, args, device, seed=10 * seed + 1)
                m = metrics(predict(model, test, device), test["y"])
                row = dict(mode=mode, n_impostors=n_imp, seed=seed, train_seconds=round(secs, 1), **m)
                rows.append(row)
                print(f"seed {seed} N={n_imp:5d} {mode:10s} F1 {m['impostor_f1']:.3f} "
                      f"recall {m['impostor_recall']:.3f} precision {m['impostor_precision']:.3f} "
                      f"AUROC {m['impostor_auroc']:.3f} ({secs:.0f} s)")
                # 4. save after every run, so nothing is lost if the job stops
                with open(out / "results.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)

    # 5. average over seeds, save summary.json, draw the figure
    summary = {"device": str(device), "gpu": gpu, "args": vars(args), "zero_shot_oracle": zs,
               "total_seconds": round(time.time() - t0, 1), "by_mode_and_n": {}}
    for mode in args.modes:
        for n in sorted(args.counts):
            r = [x for x in rows if x["mode"] == mode and x["n_impostors"] == n]
            summary["by_mode_and_n"][f"{mode}_{n}"] = {
                k: [round(float(np.mean([x[k] for x in r])), 4), round(float(np.std([x[k] for x in r])), 4)]
                for k in ("impostor_f1", "impostor_recall", "impostor_precision", "impostor_auroc", "macro_f1")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    plot(rows, out)
    print(f"done in {time.time() - t0:.0f} s; results in {out}")


if __name__ == "__main__":
    main()
