





import copy
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from data import ROOT, inner_split, stratified_folds
from metrics import binary_metrics, bootstrap_ci, delong_ci


OUT = ROOT / "results/full_session_mil"
CONFIGS = [
    dict(name="mean_gated_32", dim=32, pooling="mean_gated"),
    dict(name="mean_gated_64", dim=64, pooling="mean_gated"),
    dict(name="mean_32", dim=32, pooling="mean"),
]
COMMON = dict(segment_feature_frames=16, epochs=40, patience=10, batch_size=8,
              lr=5e-4, weight_decay=0.03, token_dropout=0.2, bag_dropout=0.5,
              max_cpu_threads=2)


class SessionMIL(nn.Module):
    def __init__(self, dim, pooling):
        super().__init__()
        self.pooling = pooling
        self.proj = nn.Sequential(nn.Linear(512, dim), nn.GELU(),
                                  nn.Dropout(COMMON["token_dropout"]))
        self.v = nn.Linear(dim, dim)
        self.u = nn.Linear(dim, dim)
        self.a = nn.Linear(dim, 1)
        self.drop = nn.Dropout(COMMON["bag_dropout"])
        self.head = nn.Linear(dim * (2 if pooling == "mean_gated" else 1), 1)

    def forward(self, x, mask):
        h = self.proj(x)
        mean = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        if self.pooling == "mean_gated":
            score = self.a(torch.tanh(self.v(h)) * torch.sigmoid(self.u(h))).squeeze(-1)
            score = score.masked_fill(~mask, -torch.inf)
            attention = score.softmax(1)
            gated = (h * attention[..., None]).sum(1)
            pooled = torch.cat([mean, gated], 1)
        else:
            attention = mask.float() / mask.sum(1, keepdim=True)
            pooled = mean
        return self.head(self.drop(pooled)).squeeze(-1), attention


def read_sequences(ids):
    sequences = {}
    for c in ids:
        raw = np.load(ROOT / f"data_cache/feat_vgg16/{c}.npy").astype(np.float32)
        if raw.ndim != 2 or raw.shape[1] != 512 or len(raw) == 0 or not np.isfinite(raw).all():
            raise ValueError(f"Invalid VGG feature cache for {c}")
        n = COMMON["segment_feature_frames"]
        sequences[c] = np.stack([raw[s:s+n].mean(0) for s in range(0, len(raw), n)])
    return sequences


def fit_normalization(sequences, optimization_ids):


    x = np.concatenate([sequences[c] for c in optimization_ids]).astype(np.float64)
    mean = x.mean(0)
    scale = x.std(0)
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def pack(sequences, ids, mean, scale):
    max_len = max(len(sequences[c]) for c in ids)
    x = torch.zeros(len(ids), max_len, 512)
    mask = torch.zeros(len(ids), max_len, dtype=torch.bool)
    for i, c in enumerate(ids):
        n = len(sequences[c])
        x[i, :n] = torch.from_numpy((sequences[c] - mean) / scale)
        mask[i, :n] = True
    if not torch.isfinite(x).all() or not mask.any(1).all():
        raise ValueError("Invalid normalized sequences or masks")
    return x, mask


@torch.no_grad()
def evaluate(model, x, mask, y):
    model.eval()
    logits, attention = model(x, mask)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y).item()
    return loss, logits.sigmoid().numpy(), attention.numpy()


def fit_candidate(cfg, train_pack, validation_pack, train_labels, validation_labels, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SessionMIL(cfg["dim"], cfg["pooling"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=COMMON["lr"],
                                 weight_decay=COMMON["weight_decay"])
    x, mask = train_pack
    xv, mv = validation_pack
    history, best, best_epoch, best_state, stale = [], float("inf"), None, None, 0
    for epoch in range(1, COMMON["epochs"] + 1):
        model.train()
        order = rng.permutation(len(train_labels))
        losses = []
        for s in range(0, len(order), COMMON["batch_size"]):
            b = order[s:s + COMMON["batch_size"]]
            logits, _ = model(x[b], mask[b])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, train_labels[b])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(loss.item())
        val_loss, probabilities, _ = evaluate(model, xv, mv, validation_labels)
        metrics = binary_metrics(validation_labels.numpy(), probabilities)
        rec = dict(epoch=epoch, training_loss=float(np.mean(losses)),
                   validation_loss=val_loss, validation_accuracy=metrics["acc"],
                   validation_auc=metrics["auc"])
        if val_loss < best:
            best, best_epoch, best_state, stale = val_loss, epoch, copy.deepcopy(model.state_dict()), 0
            rec["best"] = True
        else:
            stale += 1
        history.append(rec)
        if stale >= COMMON["patience"]:
            break
    model.load_state_dict(best_state)
    return model, history, best, best_epoch


def main():
    torch.set_num_threads(COMMON["max_cpu_threads"])
    torch.set_num_interop_threads(COMMON["max_cpu_threads"])
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(ROOT / "data_cache/manifest.csv")
    ids = man.child_id.tolist()
    if len(ids) != len(set(ids)) or set(man.label) != {0, 1}:
        raise ValueError("Manifest has duplicate IDs or invalid labels")
    labels = dict(zip(ids, man.label.astype(int)))
    protocol = dict(outer_folds=5, outer_seed=0, inner_validation_fraction=0.2,
                    candidate_configs=CONFIGS, common=COMMON, device="cpu",
                    selection="lowest inner child validation BCE, then fixed candidate order",
                    epoch_selection="lowest inner child validation BCE",
                    threshold=0.5, auxiliary_clip_labels=False,
                    normalization_fit="optimization children only, tokenwise feature mean and std",
                    training="whole-session tokens, all sessions contribute one supervised BCE",
                    backbone="frozen ImageNet VGG16; pretrained, not from scratch",
                    note="Exploratory adaptive cohort reuse; internal validation only. No external validation.")
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Output directory already has a different protocol")
    protocol_path.write_text(json.dumps(protocol, indent=2))

    (OUT / "environment.json").write_text(json.dumps(dict(python=platform.python_version(),
        torch=torch.__version__, numpy=np.__version__, cpu_threads=torch.get_num_threads()), indent=2))
    sequences = read_sequences(ids)
    folds = stratified_folds(ids, [labels[c] for c in ids], 5, 0)
    rows = []
    for k, test_ids in enumerate(folds):
        outer_train = [c for c in ids if c not in test_ids]
        optimization_ids, validation_ids = inner_split(outer_train, [labels[c] for c in outer_train], 0.2, k)
        groups = [set(optimization_ids), set(validation_ids), set(test_ids)]
        assert all(not groups[i] & groups[j] for i in range(3) for j in range(i+1, 3))
        assert set().union(*groups) == set(ids)
        fold_out = OUT / f"fold{k}"
        fold_out.mkdir(exist_ok=True)
        split = dict(outer_train=outer_train, optimization=optimization_ids,
                     inner_validation=validation_ids, outer_test=test_ids, outer_seed=0, inner_seed=k,
                     normalization_fit_ids=optimization_ids)
        (fold_out / "split.json").write_text(json.dumps(split, indent=2))
        if (fold_out / "predictions.csv").exists():
            saved = pd.read_csv(fold_out / "predictions.csv")
            if set(saved.child_id) != set(test_ids):
                raise ValueError("Saved fold does not match the frozen protocol")
            rows.extend(saved.to_dict("records"))
            print(f"Fold {k}: reused completed result", flush=True)
            continue
        mean, scale = fit_normalization(sequences, optimization_ids)
        np.savez(fold_out / "normalization.npz", mean=mean, scale=scale)
        train_pack = pack(sequences, optimization_ids, mean, scale)
        validation_pack = pack(sequences, validation_ids, mean, scale)
        yt = torch.tensor([labels[c] for c in optimization_ids], dtype=torch.float32)
        yv = torch.tensor([labels[c] for c in validation_ids], dtype=torch.float32)
        candidate_scores = []
        selected_model, selected_index, selected_score, selected_epoch = None, None, float("inf"), None
        t0 = time.monotonic()
        for index, cfg in enumerate(CONFIGS):
            model, history, score, epoch = fit_candidate(cfg, train_pack, validation_pack, yt, yv, seed=1000+k)
            (fold_out / f"history_{cfg['name']}.json").write_text(json.dumps(history, indent=2))
            candidate_scores.append(dict(config=cfg, best_validation_bce=score,
                                         best_epoch=epoch, executed_epochs=len(history)))
            if score < selected_score:
                selected_model, selected_index, selected_score, selected_epoch = model, index, score, epoch
            print(f"Fold {k} {cfg['name']}: inner BCE {score:.4f} at epoch {epoch} ({time.monotonic()-t0:.1f}s)", flush=True)
        selected_cfg = CONFIGS[selected_index]
        (fold_out / "selection.json").write_text(json.dumps(dict(candidates=candidate_scores,
            selected_index=selected_index, selected_config=selected_cfg, best_epoch=selected_epoch,
            selected_validation_bce=selected_score), indent=2))
        torch.save(dict(state=selected_model.state_dict(), selected_config=selected_cfg,
                        common=COMMON, feature_mean=mean, feature_scale=scale,
                        normalization_fit_ids=optimization_ids, best_epoch=selected_epoch,
                        parameter_count=sum(p.numel() for p in selected_model.parameters())),
                   fold_out / "model.pt")

        test_pack = pack(sequences, test_ids, mean, scale)
        ytest = torch.tensor([labels[c] for c in test_ids], dtype=torch.float32)
        _, probabilities, attention = evaluate(selected_model, *test_pack, ytest)
        np.save(fold_out / "test_attention.npy", attention)
        fold_rows = [dict(child_id=c, label=labels[c], prob=float(p), fold=k)
                     for c, p in zip(test_ids, probabilities)]
        pd.DataFrame(fold_rows).to_csv(fold_out / "predictions.csv", index=False)
        rows.extend(fold_rows)
        print(f"Fold {k}: outer accuracy {binary_metrics(ytest.numpy(), probabilities)['acc']:.4f}", flush=True)
    frame = pd.DataFrame(rows)
    if frame.child_id.duplicated().any() or set(frame.child_id) != set(ids):
        raise ValueError("Outer predictions are incomplete or duplicated")
    if not np.isfinite(frame.prob).all() or not frame.prob.between(0, 1).all():
        raise ValueError("Invalid outer prediction scores")
    if not np.array_equal(frame.label.to_numpy(), np.array([labels[c] for c in frame.child_id])):
        raise ValueError("Outer prediction label mismatch")
    for k, expected in enumerate(folds):
        assert set(frame[frame.fold == k].child_id) == set(expected)
    frame.to_csv(OUT / "predictions.csv", index=False)
    y, probabilities = frame.label.to_numpy(), frame.prob.to_numpy()
    result = binary_metrics(y, probabilities)
    result["accuracy_ci"] = bootstrap_ci(y, probabilities, lambda yy, pp: binary_metrics(yy, pp)["acc"]).tolist()
    for metric in ("f1", "precision", "sensitivity", "specificity"):
        result[metric + "_ci"] = bootstrap_ci(y, probabilities,
            lambda yy, pp, key=metric: binary_metrics(yy, pp)[key]).tolist()
    result["auc_ci"] = list(delong_ci(y, probabilities)[1:])
    result["n"] = len(frame)
    result["interpretation"] = "Exploratory internally selected pretrained-feature MIL. Conditional prediction CIs omit training, selection, adaptive cohort reuse and overlapping-fold dependence. No independent external validation or clinical diagnostic claim."
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
