








import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

_figure_cache = Path(tempfile.gettempdir()) / "asd_followups_figure_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_figure_cache / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_figure_cache / "fontconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import bootstrap_ci


ROOT = Path(__file__).resolve().parents[1]
FOLLOWUPS = [
    ("nested_features", "Nested pooled features"),
    ("nested_temporal", "Nested temporal features"),
    ("full_session_mil", "Full-session feature MIL"),
    ("rendered_motion", "Rendered motion (64 × 48)"),
    ("rendered_motion128", "Rendered motion (128 × 96)"),
]
CAPTION = (
    "Child-level accuracy across completed exploratory internal cross-validation "
    "analyses (160 participants). Points denote pooled out-of-fold accuracy; bars "
    "show conditional 95% percentile bootstrap intervals from 2,000 participant "
    "resamples of the fixed predictions. All displayed methods use the same five "
    "seed-0 outer participant folds. The intervals omit training and model-selection "
    "variability, dependence from overlapping training folds, and adaptive cohort "
    "reuse. The 128 × 96 rendered-motion input also omits RGB dilation and is a "
    "combined preprocessing variant; differences between rows do not isolate "
    "causal component effects. These are internal results without independent "
    "external or clinical validation."
)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def outer_folds(manifest):

    rng = np.random.default_rng(0)
    result = [set() for _ in range(5)]
    for label in (0, 1):
        children = manifest.loc[manifest.label == label, "child_id"].tolist()
        rng.shuffle(children)
        for index, child in enumerate(children):
            result[index % 5].add(child)
    return result


def validate_predictions(frame, manifest, folds, name):
    require({"child_id", "label", "prob", "fold"} <= set(frame.columns),
            f"{name}: predictions lack required columns")
    require(len(frame) == 160 and not frame.child_id.duplicated().any(),
            f"{name}: require exactly 160 unique predictions")
    require(set(frame.child_id) == set(manifest.child_id),
            f"{name}: prediction cohort differs from manifest")
    truth = manifest.set_index("child_id").label
    require(np.array_equal(frame.label.to_numpy(), truth.loc[frame.child_id].to_numpy()),
            f"{name}: prediction labels differ from manifest")
    require(np.isfinite(frame.prob).all() and frame.prob.between(0, 1).all(),
            f"{name}: invalid prediction scores")
    require(set(frame.fold) == set(range(5)), f"{name}: require all five outer folds")
    for index, expected in enumerate(folds):
        require(set(frame.loc[frame.fold == index, "child_id"]) == expected,
                f"{name}: outer fold {index} differs from seed-0 manifest split")
    if "prediction" in frame:
        require(np.array_equal(frame.prediction.to_numpy(), (frame.prob >= 0.5).astype(int)),
                f"{name}: stored decisions disagree with fixed score boundary")


def validate_split(split, expected_test, expected_all, name):
    if "outer_test" in split:
        test, train = split["outer_test"], split["optimization"]
        validation = split["inner_validation"]
        require(set(split["outer_train"]) == set(train) | set(validation),
                f"{name}: optimization and validation differ from outer training")
        require(set(split["normalization_fit_ids"]) == set(train),
                f"{name}: normalization fit is not confined to optimization participants")
    else:
        test = split["test"]
        train = split["train"]
        validation = split.get("validation", [])
    groups = [set(train), set(validation), set(test)]
    require(all(len(values) == len(set(values)) for values in (train, validation, test)),
            f"{name}: duplicate split participant")
    require(groups[2] == expected_test and set.union(*groups) == expected_all,
            f"{name}: split does not cover expected participants")
    require(not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]),
            f"{name}: optimization/validation/test participant overlap")
    for inner in split.get("inner_splits", []):
        left, right = set(inner["train"]), set(inner["validation"])
        require(not left & right and left | right == groups[0],
                f"{name}: inner split overlaps or differs from outer training set")


def check_numeric_row(name, label, frame, accuracy, interval, sources):
    y, p = frame.label.to_numpy(), frame.prob.to_numpy()
    decisions = p >= 0.5
    measured = float(np.mean(decisions == y))
    interval = np.asarray(interval, dtype=float)
    require(np.isfinite(accuracy) and abs(measured - accuracy) < 1e-12,
            f"{name}: recorded accuracy disagrees with predictions")
    require(interval.shape == (2,) and np.isfinite(interval).all()
            and 0 <= interval[0] <= measured <= interval[1] <= 1,
            f"{name}: invalid recorded accuracy interval")
    replay = bootstrap_ci(y, p, lambda yy, pp: np.mean((pp >= 0.5) == yy),
                          n=2000, seed=0, alpha=0.05)
    require(np.allclose(replay, interval, atol=1e-12, rtol=0),
            f"{name}: recorded conditional bootstrap interval did not replay")
    return {
        "model": name, "display_label": label, "protocol": "cv5", "seed": 0,
        "n": len(frame), "n_correct": int((decisions == y).sum()),
        "accuracy": float(accuracy), "accuracy_ci": interval.tolist(),
        "accuracy_percent": float(Decimal(str(float(accuracy))) * Decimal(100)),
        "accuracy_ci_percent": [float(Decimal(str(float(value))) * Decimal(100)) for value in interval],
        "checks": {"unique_complete_manifest": True, "labels_match": True,
                   "five_outer_folds_match": True, "accuracy_replayed": True,
                   "conditional_bootstrap_ci_replayed": True},
        "sources": sources,
    }


def source_record(path, root):
    return {"path": str(Path(path).relative_to(root)), "sha256": sha256(path)}


def audited_row(name, label, table, manifest, folds, root):
    selected = table[(table.model == name) & (table.protocol == "cv5") & (table.seed == 0)]
    require(len(selected) == 1, f"{name}: require one audited CV5 seed-0 metrics row")
    metric = selected.iloc[0]
    require(metric["n"] == 160, f"{name}: audited metrics cohort is incomplete")
    directory = root / "results/runs" / name
    frames, sources = [], [source_record(root / "results/audited/metrics.csv", root)]
    if name == "kinematic_rf":
        combined_path = directory / "cv5_s0"
        prediction_path = combined_path / "preds.csv"
        protocol_path = combined_path / "protocol.json"
        protocol = json.loads(protocol_path.read_text())
        require(protocol["model"] == name and protocol["protocol"] == "cv5" and protocol["seed"] == 0,
                f"{name}: recorded protocol differs from audited row")
        frames.append(pd.read_csv(prediction_path))
        sources.extend(source_record(path, root) for path in (prediction_path, protocol_path))
        for index, expected in enumerate(folds):
            split_path = combined_path / f"outer_split{index}.json"
            validate_split(json.loads(split_path.read_text()), expected, set(manifest.child_id), name)
            sources.append(source_record(split_path, root))
    else:
        for index, expected in enumerate(folds):
            fold_path = directory / f"cv5_s0_f{index}"
            prediction_path = fold_path / "preds.csv"
            split_path = fold_path / "split.json"
            require(prediction_path.is_file() and split_path.is_file(),
                    f"{name}: completed prediction/split files missing for fold {index}")
            frame = pd.read_csv(prediction_path)
            require(set(frame.fold) == {index} and set(frame.child_id) == expected,
                    f"{name}: fold {index} prediction file is incomplete")
            validate_split(json.loads(split_path.read_text()), expected, set(manifest.child_id), name)
            frames.append(frame)
            sources.extend(source_record(path, root) for path in (prediction_path, split_path))
    frame = pd.concat(frames, ignore_index=True).sort_values("child_id").reset_index(drop=True)
    validate_predictions(frame, manifest, folds, name)
    return check_numeric_row(name, label, frame, float(metric.acc),
                             [metric.acc_lo, metric.acc_hi], sources)


def followup_row(name, label, manifest, folds, root):
    directory = root / "results" / name
    metric_path, prediction_path = directory / "metrics.json", directory / "predictions.csv"
    protocol_path = directory / "protocol.json"
    metric = json.loads(metric_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    require(protocol["outer_folds"] == 5 and protocol["outer_seed"] == 0,
            f"{name}: incompatible recorded outer protocol")
    frame = pd.read_csv(prediction_path)
    validate_predictions(frame, manifest, folds, name)
    sources = [source_record(path, root) for path in (metric_path, prediction_path, protocol_path)]
    for index, expected in enumerate(folds):
        split_path = (directory / f"fold{index}" / "split.json" if name == "full_session_mil"
                      else directory / f"fold{index}.json")
        validate_split(json.loads(split_path.read_text()), expected, set(manifest.child_id), name)
        sources.append(source_record(split_path, root))
    for filename in ("verification.json", "validation.json"):
        path = directory / filename
        if path.is_file():
            sources.append(source_record(path, root))
    result = check_numeric_row(name, label, frame, metric["acc"], metric["accuracy_ci"], sources)
    result["recorded_interpretation"] = metric.get("interpretation", "")
    if name == "rendered_motion128":
        result["input_note"] = "128 × 96 with area resizing and no dilation; combined preprocessing variant"
    return result


def plot(rows, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "axes.linewidth": 0.6, "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    height = 2.85 + max(0, len(rows) - 7) * 0.28
    fig, ax = plt.subplots(figsize=(3.45, height))
    fig.subplots_adjust(left=0.435, right=0.875, bottom=0.185, top=0.875)
    positions = np.arange(len(rows))[::-1]
    for position, row in zip(positions, rows):
        value = row["accuracy_percent"]
        low, high = row["accuracy_ci_percent"]
        ax.errorbar(value, position, xerr=[[value - low], [high - value]],
                    fmt="o", color="#243d4b", markersize=3.8, capsize=2.5,
                    elinewidth=1.0, capthick=0.85, zorder=3)
        displayed = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        ax.text(1.05, position, str(displayed), transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=7)
    ax.set_yticks(positions, [row["display_label"] for row in rows])
    ax.set(xlim=(40, 80), ylim=(-0.55, len(rows) - 0.45), xlabel="Accuracy (%)")
    ax.set_xticks([40, 50, 60, 70, 80])
    ax.tick_params(axis="y", length=0, pad=5, labelsize=6.9)
    ax.tick_params(axis="x", length=3, labelsize=7)
    ax.grid(axis="x", color="#d8dce0", linewidth=0.5, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.text(0.035, 0.957, "Exploratory internal CV (n = 160)", ha="left",
             va="top", fontsize=8)
    ax.text(1.05, 1.012, "Acc.", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5)
    fig.text(0.5, 0.026, "Bars: conditional 95% bootstrap CI", ha="center", va="bottom", fontsize=6.7)

    fig.savefig(output / "followup_accuracy.png", dpi=600, facecolor="white")
    fig.savefig(output / "followup_accuracy.pdf", facecolor="white")
    plt.close(fig)
    return {"width_inches": 3.45, "height_inches": height, "png_dpi": 600,
            "accuracy_axis_percent": [40, 80], "pdf_fonttype": 42}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output_dir or root / "results/audited").resolve()
    manifest_path = root / "data_cache/manifest.csv"
    manifest = pd.read_csv(manifest_path)
    require(len(manifest) == 160 and not manifest.child_id.duplicated().any()
            and manifest.label.value_counts().to_dict() == {1: 80, 0: 80},
            "Require the complete balanced 160-child manifest")
    folds = outer_folds(manifest)
    table = pd.read_csv(root / "results/audited/metrics.csv")
    rows = [audited_row("audited_full", "ChromaPose-MIL", table, manifest, folds, root),
            audited_row("kinematic_rf", "Hue dynamics RF", table, manifest, folds, root)]
    rows.extend(followup_row(name, label, manifest, folds, root) for name, label in FOLLOWUPS)
    optional_name = "higherres_compact"
    try:
        optional = audited_row(optional_name, "Compact MIL (128 × 96)", table, manifest, folds, root)
    except (ValueError, FileNotFoundError) as error:
        optional_status = {"model": optional_name, "included": False, "reason": str(error),
                           "required_gate": "Audited CV5 seed-0 metrics row and all five independently validated complete outer prediction/split files"}
    else:
        rows.append(optional)
        optional_status = {"model": optional_name, "included": True,
                           "required_gate": "Audited CV5 seed-0 metrics row and all five independently validated complete outer prediction/split files"}
    output.mkdir(parents=True, exist_ok=True)
    layout = plot(rows, output)
    evidence = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Completed exploratory analyses using an adaptively reused internal cohort; CPU-only figure generation",
        "caption": CAPTION, "runtime_command": "venv/bin/python src/figures_followups.py",
        "refresh_instruction": "Rerun after audited metrics.csv refresh when higherres_compact completes all five folds.",
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "script_sha256": sha256(Path(__file__)), "metrics_module_sha256": sha256(Path(__file__).with_name("metrics.py")),
        "manifest": source_record(manifest_path, root), "layout": layout,
        "interval_method": {"type": "percentile bootstrap conditional on fixed out-of-fold predictions",
                            "resampling_unit": "participant", "n_resamples": 2000, "seed": 0,
                            "quantiles": [0.025, 0.975], "threshold": 0.5,
                            "replayed_against_recorded_interval": True,
                            "excluded_uncertainty": ["training", "model selection", "overlapping CV training folds", "adaptive cohort reuse"]},
        "rows": rows, "optional_higherres_compact": optional_status,
        "outputs": [{"path": str(output / f"followup_accuracy.{extension}"),
                     "sha256": sha256(output / f"followup_accuracy.{extension}")}
                    for extension in ("png", "pdf")],
    }
    target = output / "followup_accuracy_evidence.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(evidence, indent=2) + "\n")
    temporary.replace(target)
    print(json.dumps({"figure": str(output / "followup_accuracy.png"), "rows": len(rows),
                      "optional_higherres_compact": optional_status}, indent=2))


if __name__ == "__main__":
    main()
