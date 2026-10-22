"""Train and evaluate the light-curve transformer on simulated LSST-like data.

Example (laptop, ~10-20 min on CPU):
    python ml/train.py --n-per-class 5000 --epochs 15 --ensemble 1 --out ml/results_small

Example (one GPU on TACC, see ml/tacc_job.slurm):
    python ml/train.py --n-per-class 50000 --epochs 30 --ensemble 5 --out ml/results

Writes to --out: metrics.json, MODEL_CARD.md, figures (PNG), and model weights.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate import make_dataset, CLASSES          # noqa: E402
from model import LightCurveTransformer, count_parameters   # noqa: E402

CLASS_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")     # blue, orange, aqua (colorblind-safe order)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-per-class", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--ensemble", type=int, default=1, help="number of independently seeded models")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--no-context", action="store_true", help="ignore host distance information")
    p.add_argument("--sim-seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--out", default="ml/results")
    return p.parse_args()


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_tensors(d, idx):
    return [torch.as_tensor(d[k][idx]) for k in ("t", "band", "x", "mask", "ctx", "y")]


def batches(tensors, batch_size, shuffle, generator=None):
    n = len(tensors[0])
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for k in range(0, n, batch_size):
        sel = order[k:k + batch_size]
        yield [x[sel] for x in tensors]


def run_epoch(model, tensors, device, use_ctx, opt=None, sched=None, batch_size=256, gen=None):
    train = opt is not None
    model.train(train)
    total, n, logits_all = 0.0, 0, []
    amp = device.type == "cuda"
    for t, b, x, m, c, y in batches(tensors, batch_size, shuffle=train, generator=gen):
        t, b, x, m, c, y = (v.to(device, non_blocking=True) for v in (t, b, x, m, c, y))
        with torch.set_grad_enabled(train), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(t, b, x, m, c if use_ctx else None)
            loss = F.cross_entropy(logits.float(), y)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        else:
            logits_all.append(logits.float().cpu())
        total += loss.item() * len(y)
        n += len(y)
    return total / n, (torch.cat(logits_all) if logits_all else None)


def expected_calibration_error(prob, y, n_bins=15):
    conf = prob.max(1)
    pred = prob.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            acc, cf = (pred[s] == y[s]).mean(), conf[s].mean()
            ece += s.mean() * abs(acc - cf)
            rows.append((cf, acc, int(s.sum())))
    return float(ece), rows


def make_figures(out, prob, y, meta_test, cm, rel_rows_single, rel_rows_ens):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # confusion matrix (row-normalised = recall per true class)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.6 else "#1a1a19", fontsize=10)
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title("Test-set confusion matrix (fraction of true class)", fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    # impostor recall vs outburst amplitude and vs distance
    imp = (y == 0)
    pred = prob.argmax(1)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, key, edges, label in (
            (axes[0], "amplitude", np.arange(1.0, 4.01, 0.5), "outburst amplitude (mag)"),
            (axes[1], "mu", np.arange(28.5, 33.01, 0.75), "host distance modulus")):
        v = meta_test[key]
        centers, rec, err = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = imp & (v >= lo) & (v < hi)
            if s.sum() >= 10:
                r = (pred[s] == 0).mean()
                centers.append(0.5 * (lo + hi)); rec.append(r)
                err.append(np.sqrt(r * (1 - r) / s.sum()))
        ax.errorbar(centers, rec, yerr=err, color=CLASS_COLORS[0], marker="o", ms=5, lw=2, capsize=0)
        ax.set_xlabel(label)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color="#e4e3dc", lw=0.8)
    axes[0].set_ylabel("impostor recall")
    fig.suptitle("Recovery of simulated impostor eruptions (test set)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "impostor_recall.png", dpi=200)
    plt.close(fig)

    # reliability diagram: single model vs ensemble
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    ax.plot([0, 1], [0, 1], color="#9a9890", lw=1, ls="--", label="perfect calibration")
    for rows, color, lab in ((rel_rows_single, CLASS_COLORS[1], "single model"),
                             (rel_rows_ens, CLASS_COLORS[0], "ensemble")):
        rows = [r for r in rows if r[2] >= 20]       # skip sparsely populated bins
        if rows:
            cf, acc, _ = zip(*rows)
            ax.plot(cf, acc, marker="o", ms=4, lw=2, color=color, label=lab)
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_xlim(0.3, 1.0); ax.set_ylim(0.3, 1.0)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Calibration (reliability diagram)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=200)
    plt.close(fig)


def example_light_curves(out, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    band_colors = ("#1baf7a", "#eb6834", "#2a78d6")
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.6), sharey=False)
    for k, ax in enumerate(axes):
        i = int(np.flatnonzero(d["y"] == k)[0])
        m = d["mask"][i]
        for b, name in enumerate("gri"):
            s = m & (d["band"][i] == b)
            ax.errorbar(d["t"][i][s], d["x"][i][s, 0], yerr=d["x"][i][s, 1], fmt="o", ms=3,
                        color=band_colors[b], label=name, lw=0.8)
        ax.invert_yaxis()
        ax.set_title(f"{CLASSES[k]} ({d['subtype'][i]})", fontsize=9)
        ax.set_xlabel("days")
    axes[0].set_ylabel("mag - median")
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="best")
    fig.tight_layout()
    fig.savefig(out / "example_light_curves.png", dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    use_ctx = not args.no_context
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))

    t0 = time.time()
    d = make_dataset(args.n_per_class, seed=args.sim_seed)
    t_sim = time.time() - t0
    n = len(d["y"])
    print(f"simulated {n} light curves in {t_sim:.0f} s")
    example_light_curves(out, d)

    rng = np.random.default_rng(12345)
    perm = rng.permutation(n)
    n_tr, n_va = int(0.8 * n), int(0.1 * n)
    i_tr, i_va, i_te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
    tr, va, te = to_tensors(d, i_tr), to_tensors(d, i_va), to_tensors(d, i_te)
    y_te = d["y"][i_te]

    member_probs, members, train_seconds, samples_seen = [], [], 0.0, 0
    for k in range(args.ensemble):
        torch.manual_seed(1000 + k)
        gen = torch.Generator().manual_seed(2000 + k)
        model = LightCurveTransformer(n_ctx=2 if use_ctx else 0, d_model=args.d_model,
                                      n_layers=args.layers).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        steps = args.epochs * ((len(i_tr) + args.batch_size - 1) // args.batch_size)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
        best, best_state = np.inf, None
        for ep in range(args.epochs):
            ts = time.time()
            tl, _ = run_epoch(model, tr, device, use_ctx, opt, sched, args.batch_size, gen)
            if device.type == "cuda":
                torch.cuda.synchronize()
            train_seconds += time.time() - ts
            samples_seen += len(i_tr)
            vl, _ = run_epoch(model, va, device, use_ctx, batch_size=1024)
            if vl < best:
                best, best_state = vl, {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
            print(f"model {k + 1}/{args.ensemble} epoch {ep + 1:3d}: train {tl:.4f}  val {vl:.4f}")
        model.load_state_dict(best_state)
        _, logits = run_epoch(model, te, device, use_ctx, batch_size=1024)
        member_probs.append(torch.softmax(logits, 1).numpy())
        torch.save(best_state, out / f"model_member{k}.pt")
        members.append(count_parameters(model))

    prob_single = member_probs[0]
    prob = np.mean(member_probs, axis=0)
    pred = prob.argmax(1)
    cm = np.zeros((3, 3), int)
    np.add.at(cm, (y_te, pred), 1)
    recall = (np.diag(cm) / cm.sum(1)).tolist()
    precision = (np.diag(cm) / np.maximum(cm.sum(0), 1)).tolist()
    ece_single, rel_single = expected_calibration_error(prob_single, y_te)
    ece_ens, rel_ens = expected_calibration_error(prob, y_te)
    meta_test = {k: d[k][i_te] for k in ("amplitude", "mu", "period")}
    sub_te = d["subtype"][i_te]
    recall_by_subtype = {s: float((pred[(sub_te == s) & (y_te == 0)] == 0).mean())
                         for s in ("periodic", "giant") if np.any((sub_te == s) & (y_te == 0))}
    contaminants = {s: float((pred[sub_te == s] == 0).mean()) for s in np.unique(sub_te[y_te != 0])}

    metrics = {
        "n_train": int(len(i_tr)), "n_val": int(len(i_va)), "n_test": int(len(i_te)),
        "classes": CLASSES, "accuracy": float((pred == y_te).mean()),
        "recall": dict(zip(CLASSES, recall)), "precision": dict(zip(CLASSES, precision)),
        "impostor_recall_by_subtype": recall_by_subtype,
        "fraction_of_contaminant_subtype_called_impostor": contaminants,
        "confusion_matrix": cm.tolist(),
        "ece_single_model": ece_single, "ece_ensemble": ece_ens,
        "ensemble_size": args.ensemble, "uses_host_context": use_ctx,
        "parameters_per_model": members[0], "epochs": args.epochs,
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "train_seconds_total": round(train_seconds, 1),
        "train_throughput_samples_per_s": round(samples_seen / max(train_seconds, 1e-9), 1),
        "simulation_seconds": round(t_sim, 1),
        "torch": torch.__version__, "python": platform.python_version(),
        "args": vars(args),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    make_figures(out, prob, y_te, meta_test, cm, rel_single, rel_ens)
    write_model_card(out, metrics)
    print(json.dumps({k: metrics[k] for k in ("accuracy", "recall", "precision", "ece_single_model",
                                              "ece_ensemble", "train_seconds_total",
                                              "train_throughput_samples_per_s")}, indent=2))


def write_model_card(out, m):
    r, p = m["recall"], m["precision"]
    text = f"""# Model card: light-curve transformer for SN impostor candidates (prototype)

**Task.** Classify LSST-like g, r, i light curves as SN impostor, supernova, or AGN-like variability.

**Training data.** {m['n_train']} simulated light curves (`ml/simulate.py`), with {m['n_val']} for validation and {m['n_test']} for testing. All model parameters and ranges are assumptions; this is not real survey data.

**Model.** Transformer encoder, {m['parameters_per_model']:,} parameters per model, {m['ensemble_size']} ensemble member(s), host-distance context {"used" if m['uses_host_context'] else "not used"}.

**Test results (simulated data).**
- Accuracy: {m['accuracy']:.3f}
- Impostor recall / precision: {r['impostor']:.3f} / {p['impostor']:.3f}
- Impostor recall by type: {', '.join(f'{k} {v:.3f}' for k, v in m['impostor_recall_by_subtype'].items())}
- Expected calibration error: single model {m['ece_single_model']:.3f}, ensemble {m['ece_ensemble']:.3f}

**Compute.** {m['device']}{' (' + m['gpu'] + ')' if m['gpu'] else ''}; training time {m['train_seconds_total']:.0f} s; throughput {m['train_throughput_samples_per_s']:.0f} light curves/s.

**Limitations.** Performance on simulations is an upper limit on performance on real data. The simulations omit many real effects (other variable classes, extinction, image artefacts, realistic LSST cadence and depth variations), and the impostor model is built on a handful of well-studied objects.
"""
    (out / "MODEL_CARD.md").write_text(text)


if __name__ == "__main__":
    main()
