





import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
APPEARANCE_DIM = 512
MIN_FRAMES = {
    'appearance_mean': 1, 'appearance_dynamics': 2, 'fusion': 2,
    'temporal_pyramid': 4, 'multiscale_motion': 65, 'temporal_fusion': 65,
}
HUE_REPRESENTATIONS = {'hue_dynamics', 'fusion', 'temporal_fusion'}
REPRESENTATIONS = set(MIN_FRAMES) | {'hue_dynamics'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _numeric_array(value, name):
    x = np.asarray(value)
    if x.dtype.kind not in 'fiu' or not np.isfinite(x).all():
        raise ValueError(f'{name} must contain finite real numeric values')
    with np.errstate(over='ignore', invalid='ignore'):
        x = x.astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f'{name} must remain finite after conversion to float32')
    return x


def feature_vector(representation, appearance=None, hue=None, expected_dim=None):






    if representation not in REPRESENTATIONS:
        raise ValueError(f'Unknown representation: {representation}')
    if representation != 'hue_dynamics':
        if appearance is None:
            raise ValueError(f'{representation} requires VGG features')
        x = _numeric_array(appearance, 'VGG features')
        if x.ndim != 2 or x.shape[1] != APPEARANCE_DIM:
            raise ValueError('VGG features must have shape (N, 512)')
        minimum = MIN_FRAMES[representation]
        if x.shape[0] < minimum:
            raise ValueError(f'{representation} requires at least {minimum} feature frames')
        if representation == 'appearance_mean':
            vector = x.mean(0)
        elif representation in {'appearance_dynamics', 'fusion'}:
            vector = np.concatenate([x.mean(0), x.std(0),
                                     np.abs(np.diff(x, axis=0)).mean(0)])
        elif representation == 'temporal_pyramid':
            vector = np.concatenate([
                np.concatenate([part.mean(0), part.std(0)])
                for n in [1, 2, 4] for part in np.array_split(x, n)
            ])
        else:
            vector = np.concatenate([x.mean(0), x.std(0)] + [
                np.abs(x[lag:] - x[:-lag]).mean(0) for lag in [1, 4, 16, 64]
            ])
    else:
        vector = np.empty(0, np.float32)
    if representation in HUE_REPRESENTATIONS:
        if hue is None:
            raise ValueError(f'{representation} requires a hue vector')
        hue = _numeric_array(hue, 'Hue vector')
        if hue.ndim != 1 or not len(hue):
            raise ValueError('Hue vector must be a nonempty flat array')
        vector = np.concatenate([vector, hue])
    if not np.isfinite(vector).all():
        raise ValueError('Feature statistics are nonfinite')
    if expected_dim is not None and vector.shape != (expected_dim,):
        raise ValueError(f'Feature dimension {len(vector)} differs from saved scaler dimension {expected_dim}')
    return vector


def load_hue_cache(path):

    with np.load(path, allow_pickle=True) as saved:
        if 'kin' not in saved or saved['kin'].shape != ():
            raise ValueError('Hue cache must contain a scalar kin dictionary')
        values = saved['kin'].item()
    if not isinstance(values, dict):
        raise ValueError('Hue cache kin value must be a dictionary')
    return values


def _sigmoid(margin):

    if margin >= 0:
        return 1.0 / (1.0 + math.exp(-margin))
    v = math.exp(margin)
    return v / (1.0 + v)


class FoldPredictor:

    def __init__(self, run, fold):
        if not isinstance(fold, int) or not 0 <= fold < 5:
            raise ValueError('Fold must be an integer from zero through four')
        self.run, self.fold = Path(run).resolve(), fold
        self.model_path = self.run / f'fold{fold}.joblib'
        self.metadata_path = self.run / f'fold{fold}.json'
        self.protocol_path = self.run / 'protocol.json'
        self.metadata = json.loads(self.metadata_path.read_text())
        self.protocol = json.loads(self.protocol_path.read_text())
        selected = self.metadata.get('selected')
        if not isinstance(selected, list) or len(selected) != 3:
            raise ValueError('Fold metadata has no valid selected configuration')
        self.representation, self.kind, self.c = selected
        if self.representation not in REPRESENTATIONS or self.kind not in {'lr', 'rbf'}:
            raise ValueError('Unsupported selected representation or classifier')
        if not isinstance(self.c, (int, float)) or not math.isfinite(self.c) or self.c <= 0:
            raise ValueError('Selected classifier C must be finite and positive')
        if selected not in self.protocol.get('candidates', []):
            raise ValueError('Selected configuration is absent from the saved candidate grid')
        if self.protocol.get('outer_folds') != 5 or self.protocol.get('outer_seed') != 0:
            raise ValueError('Expected the executed five-fold seed-zero protocol')
        self.train, self.test = self.metadata.get('train'), self.metadata.get('test')
        if (not isinstance(self.train, list) or not isinstance(self.test, list)
                or not all(isinstance(c, str) for c in self.train + self.test)
                or len(self.train) != len(set(self.train))
                or len(self.test) != len(set(self.test)) or set(self.train) & set(self.test)):
            raise ValueError('Invalid or overlapping saved train/test memberships')
        self.model_hash = sha256(self.model_path)
        self.model = joblib.load(self.model_path)
        if sha256(self.model_path) != self.model_hash:
            raise ValueError('Model file changed while loading')
        if (not isinstance(self.model, Pipeline) or len(self.model.steps) != 2
                or not isinstance(self.model.steps[0][1], StandardScaler)):
            raise ValueError('Expected the fitted scaler/classifier pipeline')
        self.scaler, self.classifier = [stage for _, stage in self.model.steps]
        expected_class = LogisticRegression if self.kind == 'lr' else SVC
        if not isinstance(self.classifier, expected_class) or self.classifier.C != self.c:
            raise ValueError('Saved model does not match selected classifier metadata')
        if self.kind == 'rbf' and self.classifier.kernel != 'rbf':
            raise ValueError('Selected RBF model has a different saved kernel')
        if not np.array_equal(self.classifier.classes_, [0, 1]):
            raise ValueError('Saved classifier must use classes zero (TD) and one (ASD)')
        self.dim = int(self.scaler.n_features_in_)
        if self.dim != int(self.classifier.n_features_in_):
            raise ValueError('Saved scaler and classifier feature dimensions differ')
        self.sources = {
            'inference_source': {'path': str(Path(__file__).resolve()), 'sha256': sha256(__file__)},
            'current_training_source': {'path': str(ROOT / 'src/nested_features.py'),
                                        'sha256': sha256(ROOT / 'src/nested_features.py')},
            'model': {'path': str(self.model_path), 'sha256': self.model_hash},
            'fold_metadata': {'path': str(self.metadata_path), 'sha256': sha256(self.metadata_path)},
            'protocol': {'path': str(self.protocol_path), 'sha256': sha256(self.protocol_path)},
        }

    @property
    def requires_hue(self):
        return self.representation in HUE_REPRESENTATIONS

    def predict(self, appearance=None, hue=None, child=None):
        vector = feature_vector(self.representation, appearance, hue, self.dim)
        with threadpool_limits(limits=2):
            margin = float(self.model.decision_function(vector[None])[0])
        if not math.isfinite(margin):
            raise ValueError('Saved classifier produced a nonfinite margin')
        membership = ('not_provided' if child is None else 'test' if child in self.test
                      else 'train' if child in self.train else 'unlisted')
        return {
            'run': str(self.run), 'fold': self.fold,
            'selected_configuration': self.metadata['selected'],
            'decision_margin': margin, 'predicted_label': int(margin >= 0),
            'predicted_class': 'ASD' if margin >= 0 else 'TD',
            'decision_rule': 'label 1 iff margin >= 0; the executed metrics use sigmoid(margin) >= 0.5',
            'monotone_score': _sigmoid(margin), 'score_is_calibrated_probability': False,
            'feature_dimension': len(vector),
            'feature_vector_sha256': hashlib.sha256(vector.tobytes(order='C')).hexdigest(),
            'child_id': child, 'child_split_status': membership,
            'child_id_is_held_out_in_this_fold': membership == 'test',
            'split_interpretation': 'Membership concerns the supplied identifier, not independently verified input identity. Training-child scores are not out-of-fold predictions.',
            'cpu_thread_limit': 2, 'numpy_version': np.__version__,
            'sklearn_version': sklearn.__version__, 'sources': dict(self.sources),
            'interpretation': 'Uncalibrated sigmoid-transformed decision margin, not an ASD probability or clinical diagnosis. Each fold has its own learned score scale.',
        }


def infer(run, fold, features=None, hue_vector=None, child=None,
          hue_cache=ROOT / 'data_cache/handcrafted.npz',
          features_dir=ROOT / 'data_cache/feat_vgg16'):
    predictor = FoldPredictor(run, fold)
    if child is not None and (not child or Path(child).name != child or child in {'.', '..'}):
        raise ValueError('Child identifier must be a plain identifier, not a path')
    sources = {}
    appearance = None
    if predictor.representation != 'hue_dynamics':
        if features is None:
            if child is None:
                raise ValueError('Provide --features or --child for VGG input')
            features = Path(features_dir) / f'{child}.npy'
        features = Path(features).resolve()
        appearance = np.load(features, allow_pickle=False)
        sources['vgg_features'] = {'path': str(features), 'sha256': sha256(features)}
    hue = None
    if predictor.requires_hue:
        if hue_vector is not None:
            hue_vector = Path(hue_vector).resolve()
            hue = np.load(hue_vector, allow_pickle=False)
            sources['hue_vector'] = {'path': str(hue_vector), 'sha256': sha256(hue_vector)}
        elif child is not None:
            hue_cache = Path(hue_cache).resolve()
            values = load_hue_cache(hue_cache)
            if child not in values:
                raise ValueError(f'Hue cache has no vector for child {child}')
            hue = values[child]
            sources['hue_cache'] = {'path': str(hue_cache), 'sha256': sha256(hue_cache)}
        else:
            raise ValueError(f'{predictor.representation} requires --hue-vector or --child with --hue-cache')
    result = predictor.predict(appearance, hue, child)
    result['sources'].update(sources)
    result['feature_frames'] = len(appearance) if appearance is not None else None
    return result


def replay_run(run, features_dir=ROOT / 'data_cache/feat_vgg16',
               hue_cache=ROOT / 'data_cache/handcrafted.npz', tolerance=2e-6):

    run = Path(run).resolve()
    predictions_path = run / 'predictions.csv'
    with predictions_path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 160 or len({r['child_id'] for r in rows}) != 160:
        raise ValueError('Replay requires exactly 160 unique saved child predictions')
    predictors = {k: FoldPredictor(run, k) for k in range(5)}
    seen_test = [c for predictor in predictors.values() for c in predictor.test]
    if len(seen_test) != 160 or len(set(seen_test)) != 160:
        raise ValueError('The five saved test memberships must partition 160 children')
    if set(seen_test) != {r['child_id'] for r in rows}:
        raise ValueError('Prediction children differ from saved fold memberships')
    hues = load_hue_cache(hue_cache) if any(p.requires_hue for p in predictors.values()) else None
    max_margin_error = max_score_error = 0.0
    correct = mismatches = 0
    feature_hashes, selected, models = {}, {}, {}
    for row in rows:
        child = row['child_id']
        k = int(row['fold'])
        if k not in predictors or child not in predictors[k].test:
            raise ValueError(f'{child} is not held out in its saved prediction fold')
        predictor = predictors[k]
        path = Path(features_dir) / f'{child}.npy'
        x = np.load(path, allow_pickle=False)
        hue = hues[child] if predictor.requires_hue else None
        result = predictor.predict(x, hue, child)
        margin_error = abs(result['decision_margin'] - float(row['decision']))
        score_error = abs(result['monotone_score'] - float(row['prob']))
        saved_label = int(float(row['prob']) >= .5)
        mismatches += result['predicted_label'] != saved_label
        correct += result['predicted_label'] == int(row['label'])
        max_margin_error = max(max_margin_error, margin_error)
        max_score_error = max(max_score_error, score_error)
        feature_hashes[child] = sha256(path)
        selected[str(k)] = predictor.metadata['selected']
        models[str(k)] = predictor.sources['model']
    if mismatches or max_margin_error > tolerance or max_score_error > tolerance:
        raise ValueError(f'Replay differs: {mismatches} labels, max margin error {max_margin_error}, max score error {max_score_error}')
    return {
        'run': str(run), 'n_children': len(rows), 'n_folds': 5, 'passed': True,
        'max_absolute_margin_error': max_margin_error,
        'max_absolute_score_error': max_score_error,
        'label_mismatches': int(mismatches), 'correct': int(correct),
        'accuracy': correct / len(rows), 'comparison_tolerance': tolerance,
        'selected_configurations': selected, 'models': models,
        'saved_predictions_sha256': sha256(predictions_path),
        'vgg_feature_file_sha256': feature_hashes,
        'hue_cache_sha256': sha256(hue_cache) if hues is not None else None,
        'inference_source_sha256': sha256(__file__),
        'original_executed_script_current_sha256': sha256(ROOT / 'src/nested_features.py'),
        'numeric_note': 'Single-session float32 logistic-regression matrix-vector evaluation can differ slightly from the original batched matrix multiplication. The saved transforms and learned weights are unchanged.',
        'interpretation': 'Every child was scored only by its saved held-out fold model. No fitting, calibration, model selection, or all-fold ensemble was performed.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--features', type=Path, help='Frame-wise VGG array with shape (N,512)')
    parser.add_argument('--hue-vector', type=Path, help='Flat hue descriptor array, needed only for selected hue/fusion heads')
    parser.add_argument('--child', help='Existing cohort child ID; defaults to its local VGG and hue caches')
    parser.add_argument('--hue-cache', type=Path, default=ROOT / 'data_cache/handcrafted.npz')
    parser.add_argument('--features-dir', type=Path, default=ROOT / 'data_cache/feat_vgg16')
    parser.add_argument('--output', type=Path, help='Optional JSON file; JSON also prints to stdout')
    args = parser.parse_args()
    try:
        result = infer(args.run, args.fold, args.features, args.hue_vector,
                       args.child, args.hue_cache, args.features_dir)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n')
    print(rendered)


if __name__ == '__main__':
    main()
