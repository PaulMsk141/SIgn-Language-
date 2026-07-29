"""Trainer: reads raw landmarks from STDIN, trains an MLP on the GPU, saves joblib.

Fresh process with NO MediaPipe -> clean CUDA context, so the GPU is usable.
Reads the in-memory .npz piped from extract.py on STDIN (never touches disk),
featurizes it per config.FEATURES, trains with PyTorch (val accuracy per epoch),
and exports the weights into a pure-numpy classifier saved as joblib that app.py
loads unchanged. The output path is whatever config.MODEL selects
(config.MODEL_WEIGHTS).

    python extract.py --sources 6 | python train.py

fit() holds the whole training recipe and takes plain arrays, so experiments.ipynb
trains through the exact same code path as this CLI instead of a copy of it. This CLI
runs config.ITERATIONS times, prints the spread of the final validation accuracy and
saves the median run's weights.
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "neural_network"))

import config as cfg                                     # noqa: E402
from asl_mlp import ASLClassifier                        # noqa: E402  (numpy only)
from extract import load_payload                         # noqa: E402  (no MediaPipe at import)


def fit(X, y, epochs=None, batch=None, progress=True, desc="train"):
    """Train the config.HIDDEN_LAYERS MLP on (X, y) -> (clf, history, y_val, y_val_pred).

    clf is the pure-numpy ASLClassifier (exactly what joblib saves and app.py loads),
    so callers can predict without torch. y_val/y_val_pred are the held-out split's
    true/predicted LABELS, for a confusion matrix.
    """
    import torch
    import torch.nn as nn
    from tqdm.auto import tqdm

    epochs = cfg.EPOCHS if epochs is None else epochs
    batch = cfg.BATCH_SIZE if batch is None else batch
    classes = np.unique(y)
    yi = np.searchsorted(classes, y)                     # label -> index

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    X, yi = X[perm], yi[perm]
    nv = max(1, int(cfg.VAL_SPLIT * len(X)))
    Xva, yva, Xtr, ytr = X[:nv], yi[:nv], X[nv:], yi[nv:]
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-6

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev} | samples: {len(X)} | classes: {len(classes)}")
    tt = lambda arr: torch.tensor(arr, device=dev)
    Xtr_t, ytr_t = tt((Xtr - mean) / std), tt(ytr).long()
    Xva_t, yva_t = tt((Xva - mean) / std), tt(yva).long()

    layers, prev = [], X.shape[1]                        # MLP from config.HIDDEN_LAYERS
    for h in cfg.HIDDEN_LAYERS:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        if cfg.BATCH_NORM:
            layers.append(nn.BatchNorm1d(h))
        if cfg.DROPOUT > 0:
            layers.append(nn.Dropout(cfg.DROPOUT))
        prev = h
    layers.append(nn.Linear(prev, len(classes)))
    net = nn.Sequential(*layers).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.LEARNING_RATE)
    lossf = nn.CrossEntropyLoss()

    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    acc = 0.0
    bar = tqdm(range(epochs), desc=desc, disable=not progress)
    for _ in bar:
        net.train()
        idx = torch.randperm(len(Xtr_t), device=dev)
        tl, tc, tn = 0.0, 0, 0                            # running train loss / correct / seen
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            if len(b) < 2:
                continue          # BatchNorm1d needs >1 sample; a 1-sample step is noise
            tn += len(b)
            opt.zero_grad()
            logits = net(Xtr_t[b])
            loss = lossf(logits, ytr_t[b])
            loss.backward()
            opt.step()
            tl += loss.item() * len(b)
            tc += (logits.argmax(1) == ytr_t[b]).sum().item()
        net.eval()
        with torch.no_grad():
            vo = net(Xva_t)
            v_loss = lossf(vo, yva_t).item()
            acc = (vo.argmax(1) == yva_t).float().mean().item()
        history["loss"].append(tl / tn)
        history["accuracy"].append(tc / tn)
        history["val_loss"].append(v_loss)
        history["val_accuracy"].append(acc)
        bar.set_postfix(loss=f"{history['loss'][-1]:.3f}", val_acc=f"{acc:.3f}")

    net.eval()
    with torch.no_grad():
        vpred = net(Xva_t).argmax(1).cpu().numpy()

        # Export to numpy (nn.Linear weight is (out,in); ASLClassifier uses x@W).
        # At eval a BatchNorm is just y = a*x + b, and it always sits between two
        # Linears, so fold it into the NEXT one instead of teaching ASLClassifier
        # (and app.py) about a third layer type.
        Ws, bs, fold = [], [], None
        for m in net:
            if isinstance(m, nn.BatchNorm1d):
                a = (m.weight / torch.sqrt(m.running_var + m.eps)).cpu().numpy()
                fold = (a, m.bias.cpu().numpy() - a * m.running_mean.cpu().numpy())
            elif isinstance(m, nn.Linear):
                W, b = m.weight.cpu().numpy().T, m.bias.cpu().numpy()
                if fold is not None:
                    a, shift = fold
                    W, b, fold = a[:, None] * W, shift @ W + b, None
                Ws.append(W)
                bs.append(b)
        clf = ASLClassifier(Ws, bs, list(classes), mean, std)

        # A wrong fold would fail silently and only show up as bad app.py accuracy.
        ref = net(Xva_t[:256]).cpu().numpy()
    assert np.allclose(clf._logits(Xva[:256]), ref, atol=1e-3, rtol=1e-3), \
        "numpy export does not reproduce the torch net"
    return clf, history, classes[yva], classes[vpred]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    ap.add_argument("--batch", type=int, default=cfg.BATCH_SIZE)
    a = ap.parse_args()
    out = str(cfg.MODEL_WEIGHTS)                          # model chosen in config.MODEL

    raw = sys.stdin.buffer.read()
    if not raw:
        sys.exit("train.py: no data on stdin (did extract.py fail?)")
    X, y, _ = load_payload(raw)          # raw landmarks -> features per config.FEATURES
    print(f"features: {cfg.FEATURES} -> {X.shape[1]} dims")

    n = max(1, cfg.ITERATIONS)
    runs = [fit(X, y, a.epochs, a.batch, desc=f"train {i + 1}/{n}") for i in range(n)]
    accs = np.array([h["val_accuracy"][-1] for _, h, _, _ in runs])

    # Keep the MEDIAN run, not the best: identical data and recipe, so the only thing
    # separating them is the seedless init/shuffle/dropout draw, and shipping the top
    # of n draws reports an accuracy the next retrain won't reproduce.
    clf, history, yva, vpred = runs[int(np.argsort(accs)[n // 2])]
    if n > 1:
        print(f"val acc over {n} runs: {accs.mean():.4f} +/- {accs.std(ddof=1):.4f} "
              f"(min {accs.min():.4f}, max {accs.max():.4f})")
    print("final val acc:", round(history["val_accuracy"][-1], 4), "(median run, saved)")

    import joblib
    from plots import save_history, save_val_predictions
    stem = os.path.splitext(os.path.basename(out))[0]
    print("plots ->", ", ".join(save_history(history, str(cfg.MODEL_DIR), stem)))
    print("val predictions ->",
          save_val_predictions(vpred, yva, str(cfg.MODEL_DIR), stem))
    joblib.dump(clf, out)
    print(f"saved -> {out}")

    # Sidecar for experiments.ipynb: every iteration's loss history and model. test.py
    # scores the models on test_set right after this and writes the predictions back,
    # so the notebook only ever reads -- it never retrains.
    sidecar = os.path.join(cfg.MODEL_DIR, f"{stem}_runs.joblib")
    joblib.dump(dict(features=cfg.FEATURES, dims=X.shape[1],
                     hidden_layers=tuple(cfg.HIDDEN_LAYERS),   # record the architecture
                     batch_norm=cfg.BATCH_NORM,                # actually trained, so a
                     dropout=cfg.DROPOUT,                      # later config edit cannot
                     experiment=cfg.EXPERIMENT if cfg.EXPERIMENT_MODE else None,
                     histories=[h for _, h, _, _ in runs],
                     models=[c for c, _, _, _ in runs]), sidecar)
    print(f"{n} iterations -> {sidecar}")


if __name__ == "__main__":
    main()
