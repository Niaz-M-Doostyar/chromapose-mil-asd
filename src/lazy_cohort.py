




from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "higherres-rgb-memmap-v1"
CHUNKED_SCHEMA = "higherres-rgb-chunked-npz-v1"


def _digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _close(array):
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _signature(path):
    value = path.stat()
    return value.st_size, value.st_mtime_ns


def _read_array(path):
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (ValueError, OSError) as error:
        raise ValueError(f"Invalid memory-mapped cache array: {path}") from error
    if not isinstance(array, np.memmap):
        if hasattr(array, "close"):
            array.close()
        raise ValueError(f"Expected standalone .npy array: {path}")
    if path.stat().st_size != array.offset + array.nbytes:
        _close(array)
        raise ValueError(f"Unexpected array file size: {path}")
    return array


def _chunk_headers(path):

    result = {}
    try:
        with zipfile.ZipFile(path) as archive:
            if sorted(archive.namelist()) != ["frames.npy", "occupancy.npy"]:
                raise ValueError("Chunk must contain exactly frames and occupancy")
            for name in ("frames.npy", "occupancy.npy"):
                with archive.open(name) as member:
                    version = np.lib.format.read_magic(member)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(member)
                    elif version == (2, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(member)
                    else:
                        raise ValueError(f"Unsupported chunk NPY format: {version}")
                    result[name] = (shape, dtype, fortran)
    except (zipfile.BadZipFile, OSError, ValueError) as error:
        raise ValueError(f"Invalid compressed cache chunk: {path}") from error
    return result


class LazyCohort:










    def __init__(self, device, frames_dir=ROOT / "data_cache/frames128",
                 manifest=ROOT / "data_cache/manifest.csv", clip_len=32, hop=16,
                 min_valid=0.5, child_ids=None, max_open=4, verify_hashes=False):
        self._maps = OrderedDict()
        if not isinstance(clip_len, int) or clip_len < 1:
            raise ValueError("clip_len must be a positive integer")
        if not isinstance(hop, int) or hop < 1:
            raise ValueError("hop must be a positive integer")
        if not isinstance(max_open, int) or max_open < 1:
            raise ValueError("max_open must be a positive integer")
        if not 0 <= min_valid <= 1:
            raise ValueError("min_valid must lie between zero and one")
        self.device = torch.device(device)
        self.clip_len, self.hop, self.min_valid, self.max_open = clip_len, hop, min_valid, max_open
        self.frames_dir = Path(frames_dir)
        self.man = pd.read_csv(manifest)
        if not {"child_id", "label"} <= set(self.man.columns):
            raise ValueError("Manifest requires child_id and label columns")
        if self.man.child_id.isna().any() or self.man.child_id.duplicated().any():
            raise ValueError("Manifest child IDs must be present and unique")
        if not self.man.label.isin([0, 1]).all():
            raise ValueError("Manifest labels must be zero or one")
        if child_ids is not None:
            requested = list(child_ids)
            if len(requested) != len(set(requested)):
                raise ValueError("Repeated child ID in requested subset")
            missing = set(requested) - set(self.man.child_id)
            if missing:
                raise ValueError(f"Unknown requested child IDs: {sorted(missing)}")
            self.man = self.man[self.man.child_id.isin(requested)]
        if self.man.empty:
            raise ValueError("Cohort must include at least one child")
        self.man = self.man.set_index("child_id")
        self.ids = list(self.man.index)
        self.label = {child: int(self.man.loc[child, "label"]) for child in self.ids}
        self.length, self.starts, self.offset = {}, {}, {}
        self.frame_shape, self.metadata_paths, self.cache_parameters = {}, {}, {}
        self._paths, self._signatures, self._storage, self._chunk_paths, self._chunk_sizes = {}, {}, {}, {}, {}
        position, shared_spatial_shape, shared_fps = 0, None, None
        for child in self.ids:
            directory = self.frames_dir / child
            required = [directory / name for name in ("complete.json", "metadata.json")]
            if any(not path.is_file() for path in required):
                raise FileNotFoundError(f"Missing completed cache for {child}: {directory}")
            marker_path, meta_path = required
            try:
                marker = json.loads(marker_path.read_text())
                metadata = json.loads(meta_path.read_text())
                config = metadata["parameters"]
                hashes = marker["output_sha256"]
                count = metadata["n_frames"]
                shape = tuple(metadata["frame_shape"])
                schema = metadata["schema"]
                compressed = schema == CHUNKED_SCHEMA
                if (schema not in (SCHEMA, CHUNKED_SCHEMA) or marker["schema"] != schema
                        or metadata["status"] != "complete" or metadata["child_id"] != child
                        or marker["child_id"] != child or marker["n_frames"] != count):
                    raise ValueError("Completion record or metadata does not match this child")
                if type(count) is not int or count < 1:
                    raise ValueError("A completed child must contain at least one frame")
                if shape != (count, config["height"], config["width"], 3):
                    raise ValueError("Metadata shape disagrees with dimensions")
                if any(type(config[key]) is not int or config[key] <= 0
                       for key in ("height", "width", "fps")):
                    raise ValueError("Cache dimensions/rate must be positive integers")
                if config.get("color") != "RGB24":
                    raise ValueError("Cache must declare RGB24 channel order")
                if metadata.get("frame_dtype") != "uint8" or metadata.get("occupancy_dtype") != "float32":
                    raise ValueError("Unexpected metadata array dtype")
                row = metadata.get("manifest_row", {})
                if row and (row.get("child_id") != child or row.get("label") != self.label[child]):
                    raise ValueError("Cache manifest provenance disagrees with current child/label")
                if compressed:
                    chunk_size = config["chunk_frames"]
                    if config.get("storage") != "chunked_npz" or type(chunk_size) is not int or chunk_size < 1:
                        raise ValueError("Compressed cache must declare its positive chunk size")
                    chunks = metadata["chunks"]
                    if not isinstance(chunks, list) or not chunks:
                        raise ValueError("Compressed cache has no chunks")
                    chunk_position, filenames = 0, []
                    for index, chunk in enumerate(chunks):
                        filename, chunk_count = chunk["file"], chunk["n_frames"]
                        if (not isinstance(filename, str) or Path(filename).name != filename
                                or not filename.endswith(".npz") or filename in filenames):
                            raise ValueError("Invalid or duplicate compressed chunk filename")
                        if (type(chunk_count) is not int or not 1 <= chunk_count <= chunk_size
                                or chunk["start"] != chunk_position
                                or (index < len(chunks) - 1 and chunk_count != chunk_size)):
                            raise ValueError("Compressed chunks must be contiguous and have consistent sizes")
                        if hashes[filename] != chunk["sha256"]:
                            raise ValueError("Compressed chunk and completion hashes disagree")
                        chunk_position += chunk_count
                        filenames.append(filename)
                    if chunk_position != count:
                        raise ValueError("Compressed chunk counts do not cover the session")
                else:
                    filenames = ["frames.npy", "occupancy.npy"]
                for name in ["metadata.json", *filenames]:
                    digest = hashes[name]
                    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                        raise ValueError("Missing or malformed output SHA-256 digest")
                if _digest(meta_path) != hashes["metadata.json"]:
                    raise ValueError("Metadata SHA-256 digest mismatch")
                for name in filenames:
                    path = directory / name
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing cache data for {child}: {path}")
                    if verify_hashes:
                        if _digest(path) != hashes[path.name]:
                            raise ValueError(f"SHA-256 digest mismatch: {path.name}")
            except (KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"Invalid cache metadata for {child}: {error}") from error
            frames, occupancy = None, None
            try:
                if compressed:
                    quality = []
                    for chunk in chunks:
                        path = directory / chunk["file"]
                        if path.stat().st_size != chunk["size_bytes"]:
                            raise ValueError(f"Compressed chunk file size mismatch for {child}")
                        headers = _chunk_headers(path)
                        chunk_shape = (chunk["n_frames"], *shape[1:])
                        if headers["frames.npy"][:2] != (chunk_shape, np.dtype(np.uint8)):
                            raise ValueError(f"Frame dtype/shape mismatch for {child}")
                        if headers["occupancy.npy"][:2] != ((chunk["n_frames"],), np.dtype(np.float32)):
                            raise ValueError(f"Occupancy dtype/shape mismatch for {child}")
                        with np.load(path, allow_pickle=False) as value:
                            quality.append(value["occupancy"])
                    occupancy = np.concatenate(quality)
                else:
                    frame_path, occ_path = directory / "frames.npy", directory / "occupancy.npy"
                    frames = _read_array(frame_path)
                    occupancy = _read_array(occ_path)
                    if frames.dtype != np.uint8 or frames.shape != shape:
                        raise ValueError(f"Frame dtype/shape mismatch for {child}")
                if occupancy.dtype != np.float32 or occupancy.shape != (count,):
                    raise ValueError(f"Occupancy dtype/shape mismatch for {child}")
                if not np.isfinite(occupancy).all() or (occupancy < 0).any() or (occupancy > 1).any():
                    raise ValueError(f"Invalid source occupancy for {child}")
                valid = np.asarray(occupancy > 1e-4, dtype=np.float32)
                candidates = np.arange(0, max(count - clip_len, 0) + 1, hop)
                fractions = np.array([valid[start:start + clip_len].mean() for start in candidates])
                keep = candidates[fractions >= min_valid]
                self.starts[child] = keep if len(keep) >= 8 else candidates
            finally:
                _close(frames)
                _close(occupancy)
            if shared_spatial_shape is None:
                shared_spatial_shape, shared_fps = shape[1:], config["fps"]
            elif shape[1:] != shared_spatial_shape or config["fps"] != shared_fps:
                raise ValueError("All selected children must use identical spatial dimensions and sample rate")
            self.offset[child], self.length[child] = position, count
            position += count
            self.frame_shape[child] = shape
            self.metadata_paths[child] = meta_path
            self.cache_parameters[child] = config.copy()
            self._storage[child] = "chunked_npz" if compressed else "memmap"
            if compressed:
                self._chunk_paths[child] = [directory / name for name in filenames]
                self._chunk_sizes[child] = config["chunk_frames"]
                for index, path in enumerate(self._chunk_paths[child]):
                    self._signatures[(child, index)] = _signature(path)
            else:
                self._paths[child], self._signatures[child] = frame_path, _signature(frame_path)
        self.total_frames = position
        self.spatial_shape, self.fps = shared_spatial_shape, shared_fps

    def _frames(self, child):
        if child not in self._paths:
            raise KeyError(f"Unknown cohort child: {child}")
        if _signature(self._paths[child]) != self._signatures[child]:
            raise RuntimeError(f"Frame cache changed after validation: {child}")
        if child in self._maps:
            self._maps.move_to_end(child)
            return self._maps[child]
        array = _read_array(self._paths[child])
        self._maps[child] = array
        if len(self._maps) > self.max_open:
            _, previous = self._maps.popitem(last=False)
            _close(previous)
        return array

    def _chunk(self, child, index):
        key, path = (child, index), self._chunk_paths[child][index]
        if _signature(path) != self._signatures[key]:
            raise RuntimeError(f"Frame cache changed after validation: {child} chunk {index}")
        if key in self._maps:
            self._maps.move_to_end(key)
            return self._maps[key]
        with np.load(path, allow_pickle=False) as value:
            array = value["frames"]
        self._maps[key] = array
        if len(self._maps) > self.max_open:
            _, previous = self._maps.popitem(last=False)
            _close(previous)
        return array

    def gather(self, child, starts):
        if child not in self.length:
            raise KeyError(f"Unknown cohort child: {child}")
        starts = np.asarray(starts)
        if starts.ndim != 1 or (starts.size and not np.issubdtype(starts.dtype, np.integer)):
            raise ValueError("Clip starts must be a one-dimensional integer sequence")
        if (starts < 0).any():
            raise ValueError("Clip starts must be nonnegative")
        capped_starts = np.minimum(starts, self.length[child] - 1).astype(np.int64)
        indices = np.minimum(capped_starts[:, None] + np.arange(self.clip_len)[None],
                             self.length[child] - 1)

        if self._storage[child] == "memmap":
            clips = self._frames(child)[indices]
        else:
            clips = np.empty((len(starts), self.clip_len, *self.spatial_shape), np.uint8)
            flat_indices = indices.reshape(-1)
            destinations = clips.reshape(-1, *self.spatial_shape)
            chunk_size = self._chunk_sizes[child]
            chunk_indices = flat_indices // chunk_size
            for index in np.unique(chunk_indices):
                positions = np.flatnonzero(chunk_indices == index)
                source = self._chunk(child, int(index))
                destinations[positions] = source[flat_indices[positions] % chunk_size]
        return torch.from_numpy(clips).to(self.device, non_blocking=True)

    def sample_bag(self, child, n, rng, jitter=8):
        if not isinstance(n, int) or n < 1 or not isinstance(jitter, int) or jitter < 0:
            raise ValueError("Bag size must be positive and jitter must be nonnegative integers")
        starts = self.starts[child]
        selected = np.sort(rng.choice(starts, size=n, replace=len(starts) < n))
        selected = np.clip(selected + rng.integers(-jitter, jitter + 1, size=n),
                           0, max(self.length[child] - self.clip_len, 0))
        return self.gather(child, selected), selected

    def all_clips(self, child):
        return self.starts[child].copy()

    def close(self):
        for array in self._maps.values():
            _close(array)
        self._maps.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_maps"] = OrderedDict()
        return state

    def __del__(self):
        if hasattr(self, "_maps"):
            self.close()
