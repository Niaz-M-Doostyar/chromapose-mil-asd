
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from predict_nested import FoldPredictor, feature_vector, infer, replay_run


class NestedFeatureContractTests(unittest.TestCase):
    def test_ordered_pyramid_retains_unequal_final_partition(self):
        x = np.repeat(np.arange(7, dtype=np.float32)[:, None], 512, axis=1)
        result = feature_vector('temporal_pyramid', x).reshape(14, 512)
        expected = [3, 2, 1.5, np.sqrt(1.25), 5, np.sqrt(2 / 3),
                    .5, .5, 2.5, .5, 4.5, .5, 6, 0]
        np.testing.assert_allclose(result[:, 0], expected, atol=1e-7)
        np.testing.assert_array_equal(result, np.repeat(result[:, :1], 512, axis=1))

    def test_multiscale_descriptor_preserves_lag_order_and_float32(self):
        x = np.repeat(np.arange(100, dtype=np.float16)[:, None], 512, axis=1)
        result = feature_vector('multiscale_motion', x).reshape(6, 512)
        expected = [49.5, np.sqrt((100**2 - 1) / 12), 1, 4, 16, 64]
        np.testing.assert_allclose(result[:, 0], expected, rtol=1e-7)
        self.assertEqual(result.dtype, np.float32)

    def test_missing_hue_is_rejected_only_for_representations_that_need_it(self):
        x = np.ones((65, 512), np.float32)
        for representation in ['fusion', 'temporal_fusion', 'hue_dynamics']:
            with self.subTest(representation=representation):
                with self.assertRaisesRegex(ValueError, 'requires a hue vector'):
                    feature_vector(representation, x)
        result = feature_vector('appearance_dynamics', x)
        self.assertEqual(result.shape, (1536,))
        np.testing.assert_array_equal(result[:512], np.ones(512, np.float32))
        np.testing.assert_array_equal(result[512:], np.zeros(1024, np.float32))

    def test_nonfinite_shape_hue_and_scaler_dimension_failures(self):
        x = np.ones((65, 512), np.float32)
        for invalid in [float('nan'), float('inf')]:
            bad = x.copy(); bad[0, 1] = invalid
            with self.assertRaisesRegex(ValueError, 'finite'):
                feature_vector('appearance_mean', bad)
        with self.assertRaisesRegex(ValueError, 'shape'):
            feature_vector('appearance_mean', x[:, :511])
        with self.assertRaisesRegex(ValueError, 'flat'):
            feature_vector('fusion', x, np.ones((1, 51)))
        with self.assertRaisesRegex(ValueError, 'finite'):
            feature_vector('fusion', x, np.full(51, np.nan))
        with self.assertRaisesRegex(ValueError, 'scaler dimension'):
            feature_vector('fusion', x, np.ones(50), expected_dim=1587)
        with self.assertRaisesRegex(ValueError, 'finite after conversion'):
            feature_vector('appearance_mean', np.full((2, 512), 1e100))

    def test_short_sequences_are_rejected_before_empty_means(self):
        for representation, count, minimum in [('appearance_mean', 0, 1),
                                               ('appearance_dynamics', 1, 2),
                                               ('temporal_pyramid', 3, 4),
                                               ('multiscale_motion', 64, 65),
                                               ('temporal_fusion', 64, 65)]:
            with self.subTest(representation=representation):
                with self.assertRaisesRegex(ValueError, f'at least {minimum}'):
                    feature_vector(representation, np.zeros((count, 512)), np.zeros(51))


@unittest.skipUnless((ROOT / 'results/nested_features/fold0.joblib').is_file()
                     and (ROOT / 'results/nested_temporal/fold0.joblib').is_file()
                     and (ROOT / 'data_cache/feat_vgg16/trASD001.npy').is_file(),
                     'Complete local research models and feature caches are not present')
class NestedSavedModelReplayTests(unittest.TestCase):
    def test_every_saved_oof_row_for_both_completed_methods(self):
        for name in ['nested_features', 'nested_temporal']:
            with self.subTest(run=name):
                result = replay_run(ROOT / 'results' / name)
                self.assertTrue(result['passed'])
                self.assertEqual(result['n_children'], 160)
                self.assertEqual(result['label_mismatches'], 0)
                self.assertEqual(result['correct'], 108)
                self.assertLess(result['max_absolute_margin_error'], 2e-6)
                self.assertLess(result['max_absolute_score_error'], 2e-6)

    def test_vgg_only_model_does_not_load_irrelevant_hue_input(self):
        result = infer(ROOT / 'results/nested_features', 0, child='teTD131',
                       hue_cache=ROOT / 'does-not-exist.npz',
                       hue_vector=ROOT / 'does-not-exist.npy')
        self.assertEqual(result['child_split_status'], 'test')
        self.assertTrue(result['child_id_is_held_out_in_this_fold'])
        self.assertFalse(result['score_is_calibrated_probability'])
        self.assertEqual(result['feature_dimension'], 1536)
        self.assertNotIn('hue_cache', result['sources'])
        self.assertNotIn('hue_vector', result['sources'])

    def test_required_hue_cannot_be_silently_omitted_at_inference(self):
        features = ROOT / 'data_cache/feat_vgg16/teTD131.npy'
        with self.assertRaisesRegex(ValueError, 'requires --hue-vector'):
            infer(ROOT / 'results/nested_temporal', 0, features=features)

    def test_membership_marks_training_child_as_not_held_out(self):
        result = infer(ROOT / 'results/nested_features', 1, child='teTD131')
        self.assertEqual(result['child_split_status'], 'train')
        self.assertFalse(result['child_id_is_held_out_in_this_fold'])

    def test_explicit_hue_vector_matches_trusted_cache(self):
        with np.load(ROOT / 'data_cache/handcrafted.npz', allow_pickle=True) as z:
            hue = z['kin'].item()['teTD131']
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'hue.npy'; np.save(path, hue)
            direct = infer(ROOT / 'results/nested_temporal', 0,
                           features=ROOT / 'data_cache/feat_vgg16/teTD131.npy', hue_vector=path)
        cohort = infer(ROOT / 'results/nested_temporal', 0, child='teTD131')
        self.assertEqual(direct['decision_margin'], cohort['decision_margin'])
        self.assertEqual(direct['feature_vector_sha256'], cohort['feature_vector_sha256'])

    def test_wrong_saved_membership_and_selected_model_metadata_fail(self):
        source = ROOT / 'results/nested_features'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'protocol.json').write_text((source / 'protocol.json').read_text())
            metadata = json.loads((source / 'fold0.json').read_text())
            (root / 'fold0.joblib').symlink_to(source / 'fold0.joblib')
            overlap = dict(metadata, test=metadata['test'] + [metadata['train'][0]])
            (root / 'fold0.json').write_text(json.dumps(overlap))
            with self.assertRaisesRegex(ValueError, 'overlapping'):
                FoldPredictor(root, 0)
            altered = dict(metadata, selected=['appearance_dynamics', 'lr', .1])
            (root / 'fold0.json').write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, 'does not match'):
                FoldPredictor(root, 0)


if __name__ == '__main__':
    unittest.main()
