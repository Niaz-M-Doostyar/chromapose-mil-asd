
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from model import ChromaPoseMIL
from predict_session import infer
from train import config_overrides


class SessionInferenceTests(unittest.TestCase):
    def test_complete_session_score_is_independent_of_encoding_chunk_size(self):
        torch.set_num_threads(2)
        torch.manual_seed(12)
        cfg = config_overrides(['input=rgb', 'motion=false', 'width=[4,8,12,16]',
                                'dim=16', 'temporal=mean', 'clip_len=4', 'amp=false'])
        model = ChromaPoseMIL(cfg).eval()
        frames = np.random.default_rng(8).integers(0, 256, (36, 48, 64, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, session = root/'model.pt', root/'one.npz'
            torch.save({'cfg': cfg, 'state': model.state_dict()}, checkpoint)
            np.savez_compressed(session, frames=frames, occupancy=np.ones(36, np.float32))
            one, all_at_once = infer(checkpoint, session, chunk=1), infer(checkpoint, session, chunk=8)
            self.assertEqual(one['clip_start_frames'], [0, 16, 32])
            self.assertAlmostEqual(one['research_score'], all_at_once['research_score'], places=6)
            np.testing.assert_allclose(one['attention'], all_at_once['attention'], atol=1e-6)
            self.assertAlmostEqual(sum(one['attention']), 1, places=6)
            with self.assertRaises(ValueError):
                infer(checkpoint, session, chunk=0)

    def test_lazy_checkpoint_requires_matching_preprocessing_and_provenance(self):
        torch.set_num_threads(2)
        cfg = config_overrides(['input=rgb', 'motion=false', 'width=[4,8,12,16]',
                                'dim=16', 'temporal=mean', 'clip_len=4', 'amp=false'])
        cfg['_loader'] = 'lazy'
        model = ChromaPoseMIL(cfg).eval()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root/'model.pt'
            torch.save({'cfg': cfg, 'state': model.state_dict()}, checkpoint)
            session = root/'child'; session.mkdir()
            frames = np.random.default_rng(3).integers(0, 256, (20, 48, 64, 3), dtype=np.uint8)
            np.save(session/'frames.npy', frames)
            np.save(session/'occupancy.npy', np.ones(20, np.float32))
            params = {'width':64, 'height':48, 'fps':8, 'color':'RGB24',
                      'dilation':'none', 'resize':'cv2.INTER_AREA',
                      'source_occupancy_threshold':40, 'empty_occupancy_threshold':0.0001}
            metadata = {'schema':'higherres-rgb-memmap-v1', 'status':'complete',
                        'child_id':'child', 'n_frames':20, 'frame_shape':list(frames.shape),
                        'frame_dtype':'uint8', 'occupancy_dtype':'float32',
                        'empty_frame_fraction':0.0, 'parameters':params}
            (session/'metadata.json').write_text(json.dumps(metadata))
            hashes = {name:hashlib.sha256((session/name).read_bytes()).hexdigest()
                      for name in ['frames.npy','occupancy.npy','metadata.json']}
            (session/'complete.json').write_text(json.dumps(dict(
                schema=metadata['schema'], child_id='child', n_frames=20, output_sha256=hashes)))
            manifest = root/'manifest.csv'
            manifest.write_text('child_id,label\nchild,0\n')
            with self.assertRaisesRegex(ValueError, 'missing cache provenance'):
                infer(checkpoint, session, manifest=manifest)
            provenance = root/'cache_provenance.json'
            provenance.write_text(json.dumps({'parameters':{'child':params}}))
            result = infer(checkpoint, session, manifest=manifest)
            self.assertEqual(result['spatial_shape'], [48,64,3])
            self.assertEqual(result['clip_start_frames'], [0,16])
            self.assertTrue(np.isfinite(result['research_score']))
            wrong = dict(params, dilation='3x3')
            provenance.write_text(json.dumps({'parameters':{'child':wrong}}))
            with self.assertRaisesRegex(ValueError, 'preprocessing differs'):
                infer(checkpoint, session, manifest=manifest)
            with self.assertRaisesRegex(ValueError, 'session directory'):
                infer(checkpoint, root/'compact.npz', manifest=manifest)


if __name__ == '__main__':
    unittest.main()
