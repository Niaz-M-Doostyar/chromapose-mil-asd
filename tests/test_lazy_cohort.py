
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import Cohort
from lazy_cohort import LazyCohort, SCHEMA, CHUNKED_SCHEMA


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LazyCohortTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cache, self.old = self.root / "new", self.root / "old"
        self.cache.mkdir(); self.old.mkdir()
        self.manifest = self.root / "manifest.csv"
        self.rows = []

    def tearDown(self):
        self.temp.cleanup()

    def child(self, child, count=128, label=0, occupancy=None, dtype=np.uint8,
              occupancy_dtype=np.float32, storage="memmap", chunk_frames=16):
        frames = (np.arange(count * 4 * 6 * 3) % 251).astype(dtype).reshape(count, 4, 6, 3)
        quality = np.full(count, 0.2, dtype=occupancy_dtype) if occupancy is None else np.asarray(occupancy, dtype=occupancy_dtype)
        directory = self.cache / child; directory.mkdir()
        np.save(directory / "frames.npy", frames)
        np.save(directory / "occupancy.npy", quality)
        np.savez_compressed(self.old / f"{child}.npz", frames=frames, occupancy=quality)
        row = {"child_id": child, "label": label, "split": "train"}
        self.rows.append(row)
        metadata = {"schema": SCHEMA, "status": "complete", "child_id": child,
                    "n_frames": count, "frame_shape": list(frames.shape),
                    "frame_dtype": "uint8", "occupancy_dtype": "float32", "manifest_row": row,
                    "parameters": {"width": 6, "height": 4, "fps": 8, "color": "RGB24"}}
        (directory / "metadata.json").write_text(json.dumps(metadata))
        if storage == "compressed":
            chunks = []
            for start in range(0, count, chunk_frames):
                name = f"chunk_{len(chunks):06d}.npz"
                path = directory / name
                np.savez_compressed(path, frames=frames[start:start + chunk_frames],
                                    occupancy=quality[start:start + chunk_frames])
                chunks.append({"file": name, "start": start,
                               "n_frames": len(frames[start:start + chunk_frames]),
                               "sha256": digest(path), "size_bytes": path.stat().st_size})
            metadata.update(schema=CHUNKED_SCHEMA, chunks=chunks)
            metadata["parameters"].update(storage="chunked_npz", chunk_frames=chunk_frames)
            (directory / "metadata.json").write_text(json.dumps(metadata))
            (directory / "frames.npy").unlink(); (directory / "occupancy.npy").unlink()
        self.marker(child)
        self.write_manifest()
        return frames, quality

    def marker(self, child):
        directory = self.cache / child
        metadata = json.loads((directory / "metadata.json").read_text())
        files = [c["file"] for c in metadata["chunks"]] if metadata["schema"] == CHUNKED_SCHEMA else ["frames.npy", "occupancy.npy"]
        (directory / "complete.json").write_text(json.dumps({"schema": metadata["schema"],
            "child_id": child, "n_frames": metadata["n_frames"],
            "output_sha256": {name: digest(directory / name) for name in
                              [*files, "metadata.json"]}}))

    def write_manifest(self):
        pd.DataFrame(self.rows).to_csv(self.manifest, index=False)

    def cohort(self, **options):
        return LazyCohort("cpu", frames_dir=self.cache, manifest=self.manifest, **options)

    def test_original_start_gather_and_sampling_parity(self):
        for index, count in enumerate((128, 139)):
            quality = np.full(count, 0.2, dtype=np.float32)
            quality[:20] = 0
            self.child(f"child{index}", count, index, quality)
        original = Cohort(torch.device("cpu"), frames_dir=self.old, manifest=self.manifest,
                          clip_len=8, hop=4, min_valid=0.5)
        with self.cohort(clip_len=8, hop=4, verify_hashes=True) as lazy:
            self.assertEqual(lazy.ids, original.ids)
            self.assertEqual(lazy.label, original.label)
            self.assertEqual(lazy.length, original.length)
            self.assertEqual(lazy.offset, original.offset)
            self.assertFalse(hasattr(lazy, "frames"))
            for child in lazy.ids:
                np.testing.assert_array_equal(lazy.all_clips(child), original.all_clips(child))
                starts = np.array([0, 11, lazy.length[child] - 3])
                torch.testing.assert_close(lazy.gather(child, starts), original.gather(child, starts))
                first, picks = lazy.sample_bag(child, 12, np.random.default_rng(63), jitter=3)
                second, other_picks = original.sample_bag(child, 12, np.random.default_rng(63), jitter=3)
                np.testing.assert_array_equal(picks, other_picks)
                torch.testing.assert_close(first, second)

    def test_short_empty_and_tail_padding(self):
        frames, _ = self.child("short", count=3, occupancy=[0, 0, 0])
        with self.cohort(clip_len=8, hop=2) as lazy:
            np.testing.assert_array_equal(lazy.all_clips("short"), [0])
            batch, starts = lazy.sample_bag("short", 5, np.random.default_rng(4), jitter=16)
            np.testing.assert_array_equal(starts, np.zeros(5))
            expected = frames[np.minimum(np.arange(8), 2)]
            np.testing.assert_array_equal(batch[0].numpy(), expected)
            np.testing.assert_array_equal(lazy.gather("short", [100]).numpy()[0], np.repeat(frames[-1:], 8, axis=0))
            self.assertEqual(tuple(lazy.gather("short", []).shape), (0, 8, 4, 6, 3))
            with self.assertRaises(ValueError):
                lazy.gather("short", [-1])
            with self.assertRaises(ValueError):
                lazy.gather("short", [0.5])

    def test_mappings_are_bounded_and_tensor_copy_is_independent(self):
        self.child("one"); self.child("two")
        with self.cohort(clip_len=4, max_open=1) as lazy:
            tensor = lazy.gather("one", [0])
            before = tensor.clone()
            mapped = lazy._maps["one"]
            tensor.zero_()
            torch.testing.assert_close(lazy.gather("one", [0]), before)
            lazy.gather("two", [0])
            self.assertEqual(list(lazy._maps), ["two"])
            self.assertTrue(mapped._mmap.closed)
            self.assertEqual(lazy._maps["two"].mode, "r")

    def test_missing_child_is_not_silently_excluded(self):
        self.child("present")
        self.rows.append({"child_id": "missing", "label": 1, "split": "test"})
        self.write_manifest()
        with self.assertRaises(FileNotFoundError):
            self.cohort()
        with self.cohort(child_ids=["present"]) as selected:
            self.assertEqual(selected.ids, ["present"])
        with self.assertRaises(ValueError):
            self.cohort(child_ids=["unknown"])

    def test_zero_length_cache_is_invalid(self):
        self.child("empty", count=0)
        with self.assertRaisesRegex(ValueError, "at least one frame"):
            self.cohort()

    def test_wrong_array_dtype_and_quality_are_invalid(self):
        for frame_type, quality_type, quality in (
                (np.float32, np.float32, [0.2] * 128),
                (np.uint8, np.float64, [0.2] * 128),
                (np.uint8, np.float32, [float("nan")] * 128),
                (np.uint8, np.float32, [1.2] * 128)):
            with self.subTest(frame_type=frame_type, quality_type=quality_type, quality=quality[0]):
                child = f"bad{len(self.rows)}"
                self.child(child, dtype=frame_type, occupancy_dtype=quality_type, occupancy=quality)
                with self.assertRaises(ValueError):
                    self.cohort(child_ids=[child])

    def test_metadata_tampering_and_provenance_mismatch(self):
        self.child("one")
        path = self.cache / "one" / "metadata.json"
        metadata = json.loads(path.read_text()); metadata["parameters"]["fps"] = 9
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.cohort()
        self.marker("one")
        metadata["manifest_row"]["label"] = 1
        path.write_text(json.dumps(metadata)); self.marker("one")
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.cohort()

    def test_changed_frame_file_after_validation_is_rejected(self):
        frames, _ = self.child("one")
        with self.cohort() as lazy:
            np.save(self.cache / "one" / "frames.npy", frames[:10])
            with self.assertRaisesRegex(RuntimeError, "changed"):
                lazy.gather("one", [0])

    def test_full_hash_check_detects_data_mutation(self):
        self.child("one")
        path = self.cache / "one" / "frames.npy"
        mapped = np.load(path, mmap_mode="r+"); mapped[0, 0, 0, 0] ^= 1
        mapped.flush(); mapped._mmap.close()
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.cohort(verify_hashes=True)

    def test_compressed_cross_chunk_and_tail_gather(self):
        frames, _ = self.child("one", count=35, storage="compressed", chunk_frames=8)
        self.child("two", count=39, storage="compressed", chunk_frames=8)
        with self.cohort(clip_len=12, hop=4, max_open=2, verify_hashes=True) as lazy:
            picks = np.array([5, 7, 16, 28, 34])
            expected = frames[np.minimum(picks[:, None] + np.arange(12), 34)]
            np.testing.assert_array_equal(lazy.gather("one", picks).numpy(), expected)
            self.assertLessEqual(len(lazy._maps), 2)
            self.assertLessEqual(sum(value.nbytes for value in lazy._maps.values()), 2 * 8 * 4 * 6 * 3)
            result = lazy.gather("one", [5]); result.zero_()
            np.testing.assert_array_equal(lazy.gather("one", [5]).numpy(), expected[:1])
            self.assertEqual(tuple(lazy.gather("one", []).shape), (0, 12, 4, 6, 3))
            lazy.gather("two", [0]); self.assertLessEqual(len(lazy._maps), 2)

    def test_compressed_clip_filter_and_sampling_parity(self):
        quality = np.full(137, 0.2, dtype=np.float32); quality[:28] = 0
        self.child("one", count=137, occupancy=quality, storage="compressed", chunk_frames=16)
        original = Cohort(torch.device("cpu"), frames_dir=self.old, manifest=self.manifest,
                          clip_len=8, hop=4)
        with self.cohort(clip_len=8, hop=4) as lazy:
            np.testing.assert_array_equal(lazy.all_clips("one"), original.all_clips("one"))
            actual, picks = lazy.sample_bag("one", 12, np.random.default_rng(82))
            expected, other = original.sample_bag("one", 12, np.random.default_rng(82))
            np.testing.assert_array_equal(picks, other)
            torch.testing.assert_close(actual, expected)

    def test_compressed_missing_and_discontiguous_chunks(self):
        self.child("one", count=35, storage="compressed", chunk_frames=8)
        path = self.cache / "one" / "metadata.json"
        metadata = json.loads(path.read_text()); metadata["chunks"][1]["start"] = 9
        path.write_text(json.dumps(metadata)); self.marker("one")
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.cohort()
        metadata["chunks"][1]["start"] = 8
        path.write_text(json.dumps(metadata)); self.marker("one")
        (self.cache / "one" / "chunk_000001.npz").unlink()
        with self.assertRaises(FileNotFoundError):
            self.cohort()

    def test_compressed_wrong_dtype_and_invalid_quality(self):
        self.child("dtype", storage="compressed", dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "dtype"):
            self.cohort(child_ids=["dtype"])
        self.child("quality", storage="compressed", occupancy=[-0.1] * 128)
        with self.assertRaisesRegex(ValueError, "occupancy"):
            self.cohort(child_ids=["quality"])

    def test_compressed_short_session_padding(self):
        frames, _ = self.child("short", count=3, occupancy=[0, 0, 0], storage="compressed", chunk_frames=2)
        with self.cohort(clip_len=8) as lazy:
            actual, picks = lazy.sample_bag("short", 2, np.random.default_rng(4))
            np.testing.assert_array_equal(picks, [0, 0])
            np.testing.assert_array_equal(actual.numpy()[0], frames[np.minimum(np.arange(8), 2)])


if __name__ == "__main__":
    unittest.main()
