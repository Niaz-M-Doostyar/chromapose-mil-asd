





import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from data import ROOT, stratified_folds
from metrics import binary_metrics, bootstrap_ci, delong_ci
from palette import COLORS_RGB

OUT = ROOT / "results/rendered_motion"
FRAME_STRIDE = 4
HUE_BINS = 512
MIN_RELATIVE_MASS = .002
LAGS = (1, 4, 16)
REPRESENTATIONS = ("absolute_state", "normalized_state", "absolute_motion", "normalized_motion")
CANDIDATES = [(f, kind, c) for f in REPRESENTATIONS for kind in ("lr", "rbf")
              for c in (.01, .1, 1., 10.)]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def opponent_components(rgb):
    rgb = np.asarray(rgb, dtype=np.float32) / 255.
    r, g, b = np.moveaxis(rgb, -1, 0)
    a, q = r - .5 * (g + b), np.sqrt(3.) / 2. * (g - b)
    return np.arctan2(q, a), np.sqrt(a * a + q * q)


def prototype_lookup():
    mu, _ = opponent_components(COLORS_RGB)
    angles = -np.pi + (np.arange(HUE_BINS) + .5) * 2 * np.pi / HUE_BINS
    similarities = 30. * np.cos(angles[:, None] - mu[None, :])
    similarities -= similarities.max(axis=1, keepdims=True)
    weights = np.exp(similarities)
    return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)


LOOKUP = prototype_lookup()


def density_moments(frames):






    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError("Expected uint8 (T,H,W,3) frames")
    n, h, w, _ = frames.shape
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    coords = (np.ones((h, w)), xx, yy, xx * xx, yy * yy, xx * yy)
    hue, chroma = opponent_components(frames)

    chroma = np.where(frames.max(axis=-1) > 5, chroma, 0.)
    bins = np.floor((hue + np.pi) * HUE_BINS / (2 * np.pi)).astype(np.int32)
    bins = np.clip(bins, 0, HUE_BINS - 1)
    index = (bins + np.arange(n)[:, None, None] * HUE_BINS).ravel()
    moments = []
    for coordinate in coords:
        grouped = np.bincount(index, weights=(chroma * coordinate).ravel(),
                              minlength=n * HUE_BINS).reshape(n, HUE_BINS)
        moments.append(grouped @ LOOKUP)
    return np.stack(moments, axis=-1), h * w


def descriptors(moments, area, normalized=False):






    mass = moments[..., 0]
    denominator = np.maximum(mass, 1e-12)
    cx, cy = moments[..., 1] / denominator, moments[..., 2] / denominator
    vx = np.maximum(moments[..., 3] / denominator - cx * cx, 0.)
    vy = np.maximum(moments[..., 4] / denominator - cy * cy, 0.)
    cov = moments[..., 5] / denominator - cx * cy
    total = moments.sum(axis=1)
    relative = mass / np.maximum(total[:, :1], 1e-12)
    valid = (relative >= MIN_RELATIVE_MASS) & (mass > 1e-8)
    density = mass / area
    sx, sy = np.sqrt(vx), np.sqrt(vy)
    if normalized:
        all_mass = np.maximum(total[:, 0], 1e-12)
        gx, gy = total[:, 1] / all_mass, total[:, 2] / all_mass
        gvx = np.maximum(total[:, 3] / all_mass - gx * gx, 0.)
        gvy = np.maximum(total[:, 4] / all_mass - gy * gy, 0.)
        scale = np.maximum(np.sqrt(gvx + gvy), 1e-3)
        cx, cy = (cx - gx[:, None]) / scale[:, None], (cy - gy[:, None]) / scale[:, None]
        sx, sy = sx / scale[:, None], sy / scale[:, None]
        cov = cov / scale[:, None] ** 2
        density = relative
    states = np.stack((density, cx, cy, sx, sy, cov), axis=-1)
    mask = np.repeat(valid[..., None], 6, axis=-1)

    mask[..., 0] = True
    states = np.where(mask, states, 0.)
    return states, mask, valid


def masked_summary(x, mask):
    flat, valid = x.reshape(len(x), -1), mask.reshape(len(x), -1)
    values = []
    for j in range(flat.shape[1]):
        v = flat[valid[:, j], j]
        values.append(np.array([v.mean(), v.std(), *np.quantile(v, [.1, .5, .9])])
                      if len(v) else np.zeros(5))
    return np.asarray(values).ravel()


def session_features(moments, area, normalized=False):
    states, mask, valid = descriptors(moments, area, normalized)
    base = np.concatenate((masked_summary(states, mask), valid.mean(axis=0)))
    motion = []
    for lag in LAGS:
        delta = states[lag:] - states[:-lag]
        both = mask[lag:] & mask[:-lag]
        count = both.sum(axis=0)
        denominator = np.maximum(count, 1)
        signed = np.where(both, delta, 0.).sum(axis=0) / denominator
        absolute = np.where(both, np.abs(delta), 0.).sum(axis=0) / denominator
        rms = np.sqrt(np.where(both, delta * delta, 0.).sum(axis=0) / denominator)
        fraction = (valid[lag:] & valid[:-lag]).mean(axis=0) if len(delta) else np.zeros(valid.shape[1])
        motion.extend((signed.ravel(), absolute.ravel(), rms.ravel(), fraction))
    return base.astype(np.float32), np.concatenate((base, *motion)).astype(np.float32)


def extract_session(path):
    with np.load(path) as cache:
        frames = cache["frames"][::FRAME_STRIDE]
    chunks = [density_moments(frames[start:start + 64]) for start in range(0, len(frames), 64)]
    moments = np.concatenate([chunk[0] for chunk in chunks])
    area = chunks[0][1]
    absolute = session_features(moments, area, normalized=False)
    normalized = session_features(moments, area, normalized=True)
    return dict(zip(REPRESENTATIONS, (absolute[0], normalized[0], absolute[1], normalized[1]))), len(frames)


def extract_streamed_session(directory):





    directory = Path(directory)
    metadata_path, complete_path = directory / "metadata.json", directory / "complete.json"
    metadata, complete = json.loads(metadata_path.read_text()), json.loads(complete_path.read_text())
    if (metadata.get("schema") != "higherres-rgb-chunked-npz-v1"
            or complete.get("schema") != metadata["schema"]
            or metadata.get("status") != "complete"
            or metadata.get("child_id") != complete.get("child_id")
            or metadata.get("n_frames") != complete.get("n_frames")):
        raise ValueError("Invalid completed compressed session")
    metadata_sha = sha256(metadata_path)
    if metadata_sha != complete["output_sha256"].get("metadata.json"):
        raise ValueError("Metadata SHA-256 mismatch")
    all_moments, sampled, position, chunk_hashes = [], 0, 0, {}
    shape = tuple(metadata["frame_shape"])
    area = shape[1] * shape[2]
    for chunk in metadata["chunks"]:
        name = chunk["file"]
        if Path(name).name != name or name in chunk_hashes or chunk["start"] != position:
            raise ValueError("Chunk sequence must be safe, unique, and contiguous")
        path = directory / name
        digest = sha256(path)
        if digest != chunk["sha256"] or digest != complete["output_sha256"].get(name):
            raise ValueError(f"Chunk SHA-256 mismatch: {name}")
        chunk_hashes[name] = digest
        with np.load(path, allow_pickle=False) as value:
            frames = value["frames"]
            if frames.dtype != np.uint8 or frames.shape != (chunk["n_frames"], *shape[1:]):
                raise ValueError("Invalid compressed frame dtype/shape")
            frames = frames[(-position) % FRAME_STRIDE::FRAME_STRIDE]

            for start in range(0, len(frames), 64):
                moments, chunk_area = density_moments(frames[start:start + 64])
                if chunk_area != area:
                    raise ValueError("Inconsistent frame dimensions")
                all_moments.append(moments)
            sampled += len(frames)
        position += chunk["n_frames"]
    if position != metadata["n_frames"] or sampled != len(range(0, position, FRAME_STRIDE)):
        raise ValueError("Compressed chunks do not cover the declared session")
    moments = np.concatenate(all_moments)
    absolute = session_features(moments, area, normalized=False)
    normalized = session_features(moments, area, normalized=True)
    source = dict(metadata_sha256=metadata_sha, complete_sha256=sha256(complete_path),
                  chunks_sha256=chunk_hashes, sampled_frames=sampled,
                  source_mp4_sha256={s["path"]: s["sha256"] for s in metadata.get("sources", [])})
    return dict(zip(REPRESENTATIONS, (absolute[0], normalized[0], absolute[1], normalized[1]))), sampled, source


def make_model(kind, c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000,
                          random_state=0) if kind == "lr" else SVC(C=c, kernel="rbf"))


def write_protocol(ids, out=OUT, higherres=False):
    protocol = dict(name="rendered_motion", designation="Fixed exploratory adaptive-cohort follow-up",
                    outer_folds=5, outer_seed=0, inner_folds=4, inner_seeds=[991+k for k in range(5)],
                    selection="mean inner child accuracy; deterministic first-candidate tie-break",
                    threshold="fixed zero classifier margin; sigmoid score >= 0.5",
                    candidate_grid=CANDIDATES, input="existing 64x48 RGB rendered-skeleton cache",
                    cached_fps=8, descriptor_fps=2, frame_stride=FRAME_STRIDE,
                    hue_bins=HUE_BINS, kappa=30., minimum_relative_hue_mass=MIN_RELATIVE_MASS,
                    lags_frames=LAGS, lags_seconds=[v / 2 for v in LAGS], chunk_frames=64,
                    representations=REPRESENTATIONS, summary="mean, population std, quantiles .1/.5/.9, hue-valid fraction",
                    temporal_summary="signed, absolute and RMS lag differences, pair-valid fraction",
                    normalization="centroid/scale of all chromatic pixels; no anatomical or person-identity claim",
                    normalization_floor=.001, classifier_preprocessing="StandardScaler fitted within every inner/outer fit",
                    cpu_threads=2, outer_ids=ids,
                    caveat="Existing cohort and prior scores informed this follow-up; exploratory, not independent confirmation.")
    if higherres:
        protocol.update(name="rendered_motion128", input="128x96 RGB compressed-chunk cache; no dilation",
                        decoded_chunk_bound=128,
                        input_variant="Combined spatial resolution, no-dilation, and preprocessing/input variant; not an isolated resolution ablation.",
                        source_hash_verification="completion and metadata identity plus every actual compressed chunk SHA-256")
    path = out / "protocol.json"

    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError("Existing protocol differs; preserve completed exploratory runs")
    path.write_text(json.dumps(protocol, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--higherres", action="store_true", help="Use fixed 128x96 no-dilation compressed cache")
    args = parser.parse_args()
    out = ROOT / "results/rendered_motion128" if args.higherres else OUT
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "data_cache/manifest.csv"
    man = pd.read_csv(manifest)
    ids, y = man.child_id.tolist(), man.label.to_numpy(dtype=int)
    assert len(ids) == 160 and len(set(ids)) == 160 and (y == 0).sum() == (y == 1).sum() == 80
    write_protocol(ids, out, args.higherres)
    (out / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    provenance = dict(manifest_sha256=sha256(manifest), code_sha256=sha256(__file__),
                      palette_sha256=sha256(ROOT / "src/palette.py"), environment=dict(
                      python=sys.version, numpy=np.__version__, sklearn=sklearn.__version__,
                      platform=platform.platform(), threads=2), children={})
    features = {f: [] for f in REPRESENTATIONS}
    with threadpool_limits(limits=2):
        if args.higherres:
            from lazy_cohort import LazyCohort
            cache_root = ROOT / "data_cache/frames128_compressed"


            with LazyCohort("cpu", frames_dir=cache_root, manifest=manifest, max_open=1) as cohort:
                if cohort.ids != ids or any(cohort.frame_shape[c][1:] != (96, 128, 3) for c in ids):
                    raise ValueError("Higher-resolution cohort/dimensions disagree with fixed protocol")
                for c in ids:
                    parameters = cohort.cache_parameters[c]
                    if parameters.get("fps") != 8 or parameters.get("dilation") != "none":
                        raise ValueError("Higher-resolution cache must be 8 fps without dilation")
            provenance["cache_validator_sha256"] = sha256(ROOT / "src/lazy_cohort.py")
            provenance["source_preprocessor_sha256"] = sha256(ROOT / "src/preprocess_higherres.py")
        for n, child in enumerate(ids, 1):
            if args.higherres:
                row, sampled, source = extract_streamed_session(cache_root / child)
            else:
                path = ROOT / f"data_cache/frames/{child}.npz"
                row, sampled = extract_session(path)
                source = dict(input_sha256=sha256(path), sampled_frames=sampled)
            for f in features:
                features[f].append(row[f])
            provenance["children"][child] = source
            if n % 10 == 0 or n == 1:
                print(f"Extraction {n}/160; elapsed {time.monotonic()-start:.1f}s", flush=True)
        features = {f: np.stack(rows) for f, rows in features.items()}
        assert all(np.isfinite(x).all() for x in features.values())
        np.savez_compressed(out / "features.npz", child_ids=np.array(ids), **features)
        provenance["features_sha256"] = sha256(out / "features.npz")
        provenance["feature_dimensions"] = {f: x.shape[1] for f, x in features.items()}
        (out / "input_provenance.json").write_text(json.dumps(provenance, indent=2))
        extraction_seconds = time.monotonic() - start
        folds = stratified_folds(ids, y, 5, 0)
        rows, replay_errors, scaler_errors = [], [], []
        fold_accuracy = []
        for k, test_ids in enumerate(folds):
            test_set = set(test_ids)
            tr = np.array([i for i, child in enumerate(ids) if child not in test_set])
            te = np.array([ids.index(child) for child in test_ids])
            assert not set(tr) & set(te) and len(tr) == 128 and len(te) == 32
            inner = list(StratifiedKFold(4, shuffle=True, random_state=991 + k).split(tr, y[tr]))
            for a, b in inner:
                assert not set(tr[a]) & set(tr[b]) and not (set(tr[a]) | set(tr[b])) & set(te)
            scores = []
            for f, kind, c in CANDIDATES:
                model = make_model(kind, c)
                per_fold = cross_val_score(model, features[f][tr], y[tr], cv=inner,
                                           scoring="accuracy", n_jobs=1)
                scores.append(dict(representation=f, classifier=kind, c=c,
                                   inner_accuracy=float(per_fold.mean()),
                                   inner_fold_accuracy=per_fold.tolist()))
            best = int(np.argmax([s["inner_accuracy"] for s in scores]))
            f, kind, c = CANDIDATES[best]
            model = make_model(kind, c)
            model.fit(features[f][tr], y[tr])
            scaler_errors.append(float(np.max(np.abs(model[0].mean_ - features[f][tr].astype(float).mean(axis=0)))))
            decision = model.decision_function(features[f][te])
            score = expit(decision)
            prediction = (decision >= 0).astype(int)
            path = out / f"fold{k}.joblib"
            joblib.dump(model, path)
            replay = joblib.load(path).decision_function(features[f][te])
            replay_errors.append(float(np.max(np.abs(decision - replay))))
            assert np.allclose(decision, replay, atol=1e-12, rtol=1e-12)
            for i, p, d, yh in zip(te, score, decision, prediction):
                rows.append(dict(child_id=ids[i], label=int(y[i]), prob=float(p), decision=float(d),
                                 prediction=int(yh), fold=k))
            fold_accuracy.append(float(np.mean(prediction == y[te])))
            (out / f"fold{k}.json").write_text(json.dumps(dict(train=[ids[i] for i in tr], test=test_ids,
                inner_splits=[dict(train=[ids[i] for i in tr[a]], validation=[ids[i] for i in tr[b]]) for a, b in inner],
                selected=CANDIDATES[best], selected_inner_accuracy=scores[best]["inner_accuracy"],
                candidate_scores=scores, model_sha256=sha256(path)), indent=2))
            print(f"Fold {k}: {CANDIDATES[best]}, inner={scores[best]['inner_accuracy']:.4f}; completed", flush=True)
    frame = pd.DataFrame(rows)
    assert len(frame) == 160 and set(frame.child_id) == set(ids) and not frame.child_id.duplicated().any()
    assert np.isfinite(frame[["prob", "decision"]].to_numpy()).all()
    truth = dict(zip(ids, y))
    for row in rows:
        assert row["label"] == truth[row["child_id"]] and row["child_id"] in folds[row["fold"]]
    frame.to_csv(out / "predictions.csv", index=False)
    yy, pp = frame.label.to_numpy(), frame.prob.to_numpy()
    result = binary_metrics(yy, pp)
    result["accuracy_ci"] = bootstrap_ci(yy, pp, lambda a, b: binary_metrics(a, b)["acc"]).tolist()
    result["auc_ci"] = list(delong_ci(yy, pp)[1:])
    result["fold_accuracy"] = fold_accuracy
    result["extraction_seconds"] = extraction_seconds
    result["total_seconds"] = time.monotonic() - start
    result["interpretation"] = ("Exploratory nested CV after adaptive cohort reuse. Pooled AUC combines differently scaled "
        "LR/SVM margins; accuracy at the fixed classifier boundary is primary. Sigmoid SVM margins are not calibrated "
        "probabilities. Bootstrap/DeLong intervals condition on these predictions and omit model-selection/training "
        "variability and CV dependence. Color moments are not anatomical joints, and all rendered people contribute.")
    if args.higherres:
        result["interpretation"] += " This is a combined 128x96 no-dilation/input variant, not an isolated resolution ablation."
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    (out / "verification.json").write_text(json.dumps(dict(unique_children=160, labels_checked=True,
        outer_folds_match_primary_seed0=True, inner_splits_disjoint_from_outer_test=True,
        scaler_training_mean_max_errors=scaler_errors, checkpoint_replay_max_errors=replay_errors,
        predictions_sha256=sha256(out / "predictions.csv"), candidate_count=32,
        compressed_cache_hashes_verified=args.higherres), indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
