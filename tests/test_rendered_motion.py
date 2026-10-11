
import sys
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rendered_motion import (CANDIDATES, LOOKUP, descriptors, density_moments,
                             session_features, masked_summary, extract_session,
                             extract_streamed_session, sha256)


def transform_moments(m, scale, tx, ty):
    z = m.copy()
    mass, mx, my, mx2, my2, mxy = np.moveaxis(m, -1, 0)
    z[..., 1] = scale * mx + tx * mass
    z[..., 2] = scale * my + ty * mass
    z[..., 3] = scale * scale * mx2 + 2 * scale * tx * mx + tx * tx * mass
    z[..., 4] = scale * scale * my2 + 2 * scale * ty * my + ty * ty * mass
    z[..., 5] = scale * scale * mxy + scale * ty * mx + scale * tx * my + tx * ty * mass
    return z


class RenderedMotionTests(unittest.TestCase):
    def compressed_session(self, root, frames):
        directory = root / "child"
        directory.mkdir()
        chunks = []

        for start in range(0, len(frames), 7):
            name = f"chunk_{len(chunks):06d}.npz"
            path = directory / name
            np.savez_compressed(path, frames=frames[start:start+7], occupancy=np.ones(len(frames[start:start+7]), np.float32))
            chunks.append(dict(file=name, start=start, n_frames=len(frames[start:start+7]), sha256=sha256(path)))
        metadata = dict(schema="higherres-rgb-chunked-npz-v1", status="complete", child_id="child",
                        n_frames=len(frames), frame_shape=list(frames.shape), chunks=chunks)
        meta_path = directory / "metadata.json"
        meta_path.write_text(json.dumps(metadata))
        complete = dict(schema=metadata["schema"], child_id="child", n_frames=len(frames),
                        output_sha256={**{c["file"]: c["sha256"] for c in chunks}, "metadata.json":sha256(meta_path)})
        (directory / "complete.json").write_text(json.dumps(complete))
        return directory

    def test_streamed_global_sampling_matches_compact_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = np.random.default_rng(21).integers(0, 256, (101, 8, 12, 3), dtype=np.uint8)
            compact = root / "compact.npz"
            np.savez_compressed(compact, frames=frames)
            expected, sampled = extract_session(compact)
            directory = self.compressed_session(root, frames)
            actual, streamed_sampled, source = extract_streamed_session(directory)
            self.assertEqual(sampled, 26)
            self.assertEqual(sampled, streamed_sampled)
            self.assertEqual(len(source["chunks_sha256"]), 15)
            for key in expected:
                np.testing.assert_allclose(actual[key], expected[key], rtol=1e-6, atol=1e-7)

    def test_streaming_rejects_changed_chunk_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.compressed_session(root, np.zeros((40, 8, 8, 3), np.uint8))
            chunk = directory / "chunk_000000.npz"
            original = chunk.read_bytes()
            chunk.write_bytes(original + b"tampering")
            with self.assertRaisesRegex(ValueError, "Chunk SHA-256 mismatch"):
                extract_streamed_session(directory)
            chunk.write_bytes(original)
            metadata = directory / "metadata.json"
            metadata.write_text(metadata.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "Metadata SHA-256 mismatch"):
                extract_streamed_session(directory)

    def test_fixed_candidate_grid_and_lookup(self):
        self.assertEqual(len(CANDIDATES), 32)
        self.assertEqual(len(set(CANDIDATES)), 32)
        np.testing.assert_allclose(LOOKUP.sum(axis=1), 1., atol=2e-7)
        self.assertTrue((LOOKUP >= 0).all())

    def test_blank_frames_finite(self):
        moments, area = density_moments(np.zeros((40, 8, 8, 3), dtype=np.uint8))
        for norm in (False, True):
            a, b = session_features(moments, area, norm)
            self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())
            np.testing.assert_array_equal(a, np.zeros_like(a))
            np.testing.assert_array_equal(b, np.zeros_like(b))

    def test_layout_normalization_translation_and_scale(self):
        frames = np.zeros((20, 16, 16, 3), dtype=np.uint8)
        frames[:, 3:6, 3:5] = [255, 0, 0]
        frames[:, 8:13, 10:13] = [0, 255, 0]
        moments, area = density_moments(frames)
        transformed = transform_moments(moments, 1.7, .25, -.3)
        original, _, valid = descriptors(moments, area, True)
        shifted, _, shifted_valid = descriptors(transformed, area, True)
        np.testing.assert_array_equal(valid, shifted_valid)
        np.testing.assert_allclose(original, shifted, atol=1e-7)
        absolute, _, _ = descriptors(moments, area, False)
        shifted_absolute, _, _ = descriptors(transformed, area, False)
        self.assertGreater(np.max(np.abs(absolute - shifted_absolute)), .1)

    def test_spatial_position_is_recorded(self):
        frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)
        frames[0, 4:7, 2:4] = [255, 0, 0]
        frames[1, 4:7, 9:11] = [255, 0, 0]
        moments, area = density_moments(frames)
        state, _, valid = descriptors(moments, area)
        red = moments[0, :, 0].argmax()
        self.assertTrue(valid[:, red].all())
        self.assertAlmostEqual(state[1, red, 1] - state[0, red, 1], 14 / 15, places=5)
        self.assertAlmostEqual(state[1, red, 0], state[0, red, 0], places=7)

    def test_masked_spatial_summary_does_not_impute_absent_zero(self):
        states = np.array([[[.2, .7]], [[0., 0.]], [[.2, .7]]])
        mask = np.array([[[True, True]], [[True, False]], [[True, True]]])
        summary = masked_summary(states, mask).reshape(-1, 5)
        self.assertAlmostEqual(summary[1, 0], .7)
        self.assertEqual(summary[1, 1], 0.)
        self.assertAlmostEqual(summary[0, 0], .4 / 3)

    def test_motion_constant_session(self):
        frames = np.zeros((40, 8, 8, 3), dtype=np.uint8)
        frames[:, 2:5, 2:5] = [255, 0, 0]
        moments, area = density_moments(frames)
        for norm in (False, True):
            base, combined = session_features(moments, area, norm)
            offset = len(base)

            np.testing.assert_allclose(combined[offset:offset + 3 * 18 * 6], 0., atol=1e-7)


if __name__ == "__main__":
    unittest.main()
