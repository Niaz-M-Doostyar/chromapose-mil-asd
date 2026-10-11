




from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "higherres-rgb-memmap-v1"
CHUNKED_SCHEMA = "higherres-rgb-chunked-npz-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)}


def probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True)
    value = json.loads(result.stdout)
    stream = next(s for s in value["streams"] if s["codec_type"] == "video")
    return {"width": int(stream["width"]), "height": int(stream["height"]),
            "avg_frame_rate": stream.get("avg_frame_rate"),
            "duration_seconds": float(value["format"].get("duration", 0)),
            "codec": stream.get("codec_name")}


def resize_and_occupancy(frames, width=128, height=96):

    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Expected (N,H,W,3) uint8 RGB frames")
    occupancy = (frames.max(axis=-1) > 40).mean(axis=(1, 2)).astype(np.float32)
    small = np.empty((len(frames), height, width, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        small[index] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return small, occupancy


def ffconcat_path(path):

    text = str(path.resolve())
    if "\n" in text or "\r" in text:
        raise ValueError("Source filenames must not contain newlines")
    return "file '" + text.replace("'", "'\\''") + "'\n"


def stream_batches(paths, source_width, source_height, fps, max_seconds,
                   batch_frames, temporary_directory):
    temporary_directory = Path(temporary_directory)
    concat = temporary_directory / "sources.ffconcat"
    concat.write_text("".join(ffconcat_path(path) for path in paths))
    command = ["ffmpeg", "-v", "error", "-threads", "1", "-filter_threads", "1",
               "-f", "concat", "-safe", "0", "-i", str(concat), "-t", str(max_seconds),
               "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    frame_bytes = source_width * source_height * 3
    batch_bytes = frame_bytes * batch_frames
    with open(temporary_directory / "ffmpeg.stderr", "wb") as error_file:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=error_file)
        try:
            while True:
                raw = bytearray()
                while len(raw) < batch_bytes:
                    block = process.stdout.read(batch_bytes - len(raw))
                    if not block:
                        break
                    raw.extend(block)
                if not raw:
                    break
                if len(raw) % frame_bytes:
                    raise RuntimeError("ffmpeg produced a truncated RGB frame")
                yield np.frombuffer(raw, np.uint8).reshape(-1, source_height, source_width, 3)
            return_code = process.wait()
            if return_code:
                message = (temporary_directory / "ffmpeg.stderr").read_text(errors="replace")
                raise RuntimeError(f"ffmpeg exited {return_code}: {message[-4000:]}")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def raw_to_npy(raw_path, npy_path, dtype, shape, batch_frames):
    source = np.memmap(raw_path, mode="r", dtype=dtype, shape=shape)
    target = np.lib.format.open_memmap(npy_path, mode="w+", dtype=dtype, shape=shape)
    for start in range(0, shape[0], batch_frames):
        target[start:start + batch_frames] = source[start:start + batch_frames]
    target.flush()
    del target, source
    Path(raw_path).unlink()


def json_file(path, value):
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def process_child(job):
    cv2.setNumThreads(1)
    row, output, config, old_cache = job[:4]
    minimum_free_bytes = job[4] if len(job) > 4 else 0
    child = row["child_id"]
    paths = [Path(row["path"].format(part)) for part in range(1, int(row["n_parts"]) + 1)]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"{child}: incomplete source segments")
    sources = [source_fingerprint(path) for path in paths]
    destination = Path(output) / child
    if destination.exists():
        marker = destination / "complete.json"
        if not marker.is_file():
            raise RuntimeError(f"{destination} exists without an atomic completion marker")
        previous = json.loads((destination / "metadata.json").read_text())
        if previous["parameters"] != config or previous["sources"] != sources:
            raise RuntimeError(f"{child}: existing cache source/configuration differs; use a new output root")
        for filename, digest in json.loads(marker.read_text())["output_sha256"].items():
            if sha256_file(destination / filename) != digest:
                raise RuntimeError(f"{child}: output integrity check failed for {filename}")
        return {"child_id": child, "n_frames": previous["n_frames"], "status": "verified_existing"}

    temporary = Path(tempfile.mkdtemp(prefix=f".{child}.incomplete-", dir=output))
    try:
        if shutil.disk_usage(output).free < minimum_free_bytes:
            raise OSError(f"{child}: disk reserve would be violated before preprocessing")
        source_meta = probe(paths[0])
        frame_raw, occ_raw = temporary / "frames.raw", temporary / "occupancy.raw"
        n_frames, occupancy_sum, empty_frames = 0, 0.0, 0
        compressed = config.get("storage") == "chunked_npz"
        chunks, filled, chunk_start = [], 0, 0
        if compressed:
            chunk_frames = np.empty((config["chunk_frames"], config["height"], config["width"], 3), np.uint8)
            chunk_occ = np.empty(config["chunk_frames"], np.float32)

        def save_chunk(count):
            if shutil.disk_usage(output).free < minimum_free_bytes + count * config["height"] * config["width"] * 3:
                raise OSError(f"{child}: disk reserve would be violated before writing a chunk")
            filename = f"chunk_{len(chunks):06d}.npz"
            path = temporary / filename
            np.savez_compressed(path, frames=chunk_frames[:count], occupancy=chunk_occ[:count])
            chunks.append({"file": filename, "start": chunk_start, "n_frames": count,
                           "sha256": sha256_file(path), "size_bytes": path.stat().st_size})

        with open(frame_raw, "wb") as frame_file, open(occ_raw, "wb") as occ_file:
            for batch in stream_batches(paths, source_meta["width"], source_meta["height"],
                                        config["fps"], config["max_seconds"],
                                        config["batch_frames"], temporary):
                small, occupancy = resize_and_occupancy(batch, config["width"], config["height"])
                if compressed:
                    batch_start = 0
                    while batch_start < len(small):
                        take = min(len(small) - batch_start, config["chunk_frames"] - filled)
                        chunk_frames[filled:filled + take] = small[batch_start:batch_start + take]
                        chunk_occ[filled:filled + take] = occupancy[batch_start:batch_start + take]
                        filled += take; batch_start += take
                        if filled == config["chunk_frames"]:
                            save_chunk(filled)
                            chunk_start += filled
                            filled = 0
                else:
                    if shutil.disk_usage(output).free < minimum_free_bytes + small.nbytes:
                        raise OSError(f"{child}: disk reserve would be violated before writing frames")
                    frame_file.write(small.tobytes())
                    occ_file.write(occupancy.tobytes())
                n_frames += len(batch)
                occupancy_sum += float(occupancy.astype(np.float64).sum())
                empty_frames += int((occupancy < 1e-4).sum())
                if n_frames % 512 == 0:
                    print(f"{child}: {n_frames} frames streamed", flush=True)
            frame_file.flush(); os.fsync(frame_file.fileno())
            occ_file.flush(); os.fsync(occ_file.fileno())
        if not n_frames:
            raise RuntimeError(f"{child}: no decoded frames")
        if compressed and filled:
            save_chunk(filled)
        for source, path in zip(sources, paths):
            stat = path.stat()
            if stat.st_size != source["size_bytes"] or stat.st_mtime_ns != source["mtime_ns"]:
                raise RuntimeError(f"{child}: source changed during preprocessing")
        frame_shape = (n_frames, config["height"], config["width"], 3)
        if compressed:
            frame_raw.unlink(); occ_raw.unlink()
        else:
            if shutil.disk_usage(output).free < minimum_free_bytes + frame_raw.stat().st_size:
                raise OSError(f"{child}: insufficient disk reserve for temporary-to-NPY conversion")
            raw_to_npy(frame_raw, temporary / "frames.npy", np.uint8, frame_shape, config["batch_frames"])
            raw_to_npy(occ_raw, temporary / "occupancy.npy", np.float32, (n_frames,), config["batch_frames"])
        previous_comparison = None
        previous_path = Path(old_cache) / f"{child}.npz"
        if previous_path.is_file():

            with np.load(previous_path) as old:
                old_count = len(old["occupancy"])
                if compressed:
                    quality_chunks = []
                    for chunk in chunks:
                        with np.load(temporary / chunk["file"], allow_pickle=False) as value:
                            quality_chunks.append(value["occupancy"])
                    new_occ = np.concatenate(quality_chunks)
                else:
                    new_occ = np.load(temporary / "occupancy.npy", mmap_mode="r")
                previous_comparison = {"path": str(previous_path.resolve()),
                                       "old_n_frames": old_count, "frame_delta": n_frames - old_count,
                                       "occupancy_equal": bool(old_count == n_frames and
                                                               np.array_equal(old["occupancy"], new_occ))}
                del new_occ
        schema = CHUNKED_SCHEMA if compressed else SCHEMA
        metadata = {"schema": schema, "child_id": child, "status": "complete",
                    "manifest_row": row, "parameters": config, "sources": sources,
                    "first_source_probe": source_meta, "n_frames": n_frames,
                    "frame_shape": list(frame_shape), "frame_dtype": "uint8",
                    "occupancy_dtype": "float32", "duration_seconds": n_frames / config["fps"],
                    "mean_source_occupancy": occupancy_sum / n_frames,
                    "empty_frame_fraction": empty_frames / n_frames,
                    "old_cache_comparison": previous_comparison,
                    "numpy_version": np.__version__, "opencv_version": cv2.__version__,
                    "ffmpeg_version": subprocess.run(["ffmpeg", "-version"], check=True,
                                                      capture_output=True, text=True).stdout.splitlines()[0]}
        if compressed:
            metadata["chunks"] = chunks
        json_file(temporary / "metadata.json", metadata)
        files = [chunk["file"] for chunk in chunks] if compressed else ["frames.npy", "occupancy.npy"]
        json_file(temporary / "complete.json", {"schema": schema, "child_id": child,
                  "n_frames": n_frames, "output_sha256": {filename: sha256_file(temporary / filename)
                  for filename in [*files, "metadata.json"]}})
        (temporary / "sources.ffconcat").unlink()
        (temporary / "ffmpeg.stderr").unlink()
        os.rename(temporary, destination)
        return {"child_id": child, "n_frames": n_frames, "status": "created",
                "old_cache_comparison": previous_comparison}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data_cache/manifest.csv")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--storage", choices=["memmap", "compressed"], default="memmap")
    parser.add_argument("--chunk-frames", type=int, default=128)
    parser.add_argument("--min-free-gib", type=float, default=8.0,
                        help="Stop before spending this free-disk reserve; operational, not a representation parameter")
    parser.add_argument("--old-cache", type=Path, default=ROOT / "data_cache/frames")
    parser.add_argument("--children", nargs="+", help="Explicit child IDs (spaces or commas); required unless --all")
    parser.add_argument("--all", action="store_true", help="Explicitly opt into complete-cohort preprocessing")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-frames", type=int, default=16)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-seconds", type=int, default=700)
    args = parser.parse_args()
    if bool(args.children) == args.all:
        parser.error("Choose --children or --all")
    if min(args.workers, args.batch_frames, args.width, args.height, args.fps, args.max_seconds, args.chunk_frames) <= 0:
        parser.error("All numeric arguments must be positive")
    if not np.isfinite(args.min_free_gib) or args.min_free_gib < 0:
        parser.error("min-free-gib must be finite and nonnegative")
    manifest = pd.read_csv(args.manifest)
    if manifest.child_id.duplicated().any():
        raise ValueError("Manifest contains duplicate child IDs")
    ids = {item for group in (args.children or []) for item in group.split(",") if item}
    if ids - set(manifest.child_id):
        parser.error(f"Unknown child IDs: {sorted(ids - set(manifest.child_id))}")
    rows = manifest if args.all else manifest[manifest.child_id.isin(ids)]
    if args.out is None:
        args.out = ROOT / "data_cache" / ("frames128_compressed" if args.storage == "compressed" else "frames128")
    args.out.mkdir(parents=True, exist_ok=True)
    config = {"width": args.width, "height": args.height, "fps": args.fps,
              "max_seconds": args.max_seconds, "batch_frames": args.batch_frames,
              "resize": "cv2.INTER_AREA", "dilation": "none", "color": "RGB24",
              "source_occupancy_threshold": 40, "empty_occupancy_threshold": 1e-4,
              "ffmpeg_concat_order": "numeric_part_order", "ffmpeg_threads": 1,
              "opencv_threads": 1}
    if args.storage == "compressed":
        config.update(storage="chunked_npz", chunk_frames=args.chunk_frames)
    minimum_free_bytes = int(args.min_free_gib * 1024 ** 3)
    jobs = [(row, str(args.out), config, str(args.old_cache), minimum_free_bytes)
            for row in rows.to_dict("records")]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_child, job) for job in jobs]
        for future in as_completed(futures):
            print(json.dumps(future.result(), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
