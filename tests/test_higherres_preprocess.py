
import sys
import json
import subprocess
import tempfile
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from preprocess_higherres import ffconcat_path, process_child, raw_to_npy, resize_and_occupancy


class HigherResolutionTests(unittest.TestCase):
    def test_no_cross_channel_max_dilation(self):
        source = np.zeros((1, 2, 2, 3), dtype=np.uint8)
        source[0, 0, 0] = [255, 0, 0]
        source[0, 0, 1] = [0, 255, 0]
        resized, occupancy = resize_and_occupancy(source, width=1, height=1)
        np.testing.assert_array_equal(resized[0, 0, 0], [64, 64, 0])
        self.assertEqual(occupancy[0], 0.5)

    def test_exact_rgb_block_and_empty_source(self):
        source = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        source[0, :4, :4] = [245, 27, 16]
        resized, occupancy = resize_and_occupancy(source, width=4, height=4)
        np.testing.assert_array_equal(resized[0, :2, :2],
                                      np.broadcast_to([245, 27, 16], (2, 2, 3)))
        np.testing.assert_array_equal(occupancy, [0.25, 0])
        self.assertFalse(resized[1].any())

    def test_exact_shape_memmap_without_padding(self):
        source = np.arange(7 * 2 * 3 * 3, dtype=np.uint8).reshape(7, 2, 3, 3)
        with tempfile.TemporaryDirectory() as directory:
            raw, target = Path(directory) / "raw", Path(directory) / "frames.npy"
            source.tofile(raw)
            raw_to_npy(raw, target, np.uint8, source.shape, batch_frames=3)
            result = np.load(target, mmap_mode="r")
            np.testing.assert_array_equal(result, source)
            self.assertIsInstance(result, np.memmap)
            self.assertFalse(raw.exists())
            del result

    def test_concat_apostrophe_and_newline(self):
        escaped = ffconcat_path(Path("child's clip.mp4"))
        self.assertIn("'\\''", escaped)
        with self.assertRaises(ValueError):
            ffconcat_path(Path("bad\nclip.mp4"))

    def test_invalid_video_does_not_publish_partial_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "part_1.mp4"
            source.write_bytes(b"invalid mp4")
            output = root / "output"; output.mkdir()
            row = {"child_id": "sample", "n_parts": 1, "path": str(root / "part_{}.mp4")}
            with self.assertRaises(subprocess.CalledProcessError):
                process_child((row, output, {}, root / "old"))
            self.assertEqual(list(output.iterdir()), [])

    def test_existing_config_mismatch_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "part_1.mp4").write_bytes(b"source")
            output = root / "output"; output.mkdir()
            completed = output / "sample"; completed.mkdir()
            marker = completed / "complete.json"; marker.write_text("{}")
            metadata = completed / "metadata.json"
            metadata.write_text(json.dumps({"parameters": {"width": 64}, "sources": []}))
            original = metadata.read_bytes()
            row = {"child_id": "sample", "n_parts": 1, "path": str(root / "part_{}.mp4")}
            with self.assertRaisesRegex(RuntimeError, "configuration differs"):
                process_child((row, output, {"width": 128}, root / "old"))
            self.assertEqual(metadata.read_bytes(), original)
            self.assertEqual(marker.read_text(), "{}")

    def test_disk_reserve_failure_does_not_publish_partial_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "part_1.mp4").write_bytes(b"source")
            output = root / "output"; output.mkdir()
            row = {"child_id": "sample", "n_parts": 1, "path": str(root / "part_{}.mp4")}
            with self.assertRaisesRegex(OSError, "disk reserve"):
                process_child((row, output, {}, root / "old", 10 ** 22))
            self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
