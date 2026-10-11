






import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FPS = 8
OUT_W, OUT_H = 64, 48
MAX_SECONDS = 700


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams",
                        "-show_format", str(path)], capture_output=True, text=True)
    info = json.loads(r.stdout)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = v.get("avg_frame_rate", "0/1").split("/")
    return dict(width=int(v["width"]), height=int(v["height"]),
                src_fps=float(num) / max(float(den), 1.0),
                duration=float(info["format"].get("duration", 0.0)), codec=v.get("codec_name"))


def decode(paths, w, h, fps=FPS):

    lst = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    lst.write("".join(f"file '{p}'\n" for p in paths)); lst.close()
    cmd = ["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst.name,
           "-t", str(MAX_SECONDS), "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    Path(lst.name).unlink()
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def process_one(args):
    template, n_parts, out_path = args
    paths = [template.format(i) for i in range(1, n_parts + 1)]
    meta = probe(paths[0])
    frames = decode(paths, meta["width"], meta["height"])
    meta["duration"] = len(frames) / FPS
    kernel = np.ones((3, 3), np.uint8)
    small = np.empty((len(frames), OUT_H, OUT_W, 3), np.uint8)
    occ = np.empty(len(frames), np.float32)
    for i, f in enumerate(frames):
        occ[i] = (f.max(-1) > 40).mean()
        small[i] = cv2.resize(cv2.dilate(f, kernel), (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    np.savez_compressed(out_path, frames=small, occupancy=occ)
    meta.update(n_frames=len(frames), empty_frac=float((occ < 1e-4).mean()),
                mean_occupancy=float(occ.mean()))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data_cache/manifest.csv")
    ap.add_argument("--out", default="data_cache/frames")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    man = pd.read_csv(a.manifest)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(p, n, out / f"{cid}.npz") for p, n, cid in zip(man.path, man.n_parts, man.child_id)
            if not (out / f"{cid}.npz").exists()]
    with ProcessPoolExecutor(a.workers) as ex:
        for (p, n, o), meta in zip(jobs, ex.map(process_one, jobs)):
            meta["child_id"] = Path(o).stem
            with open(str(o).replace(".npz", ".json"), "w") as fh:
                json.dump(meta, fh)
            print(Path(o).stem, meta["n_frames"], "frames", round(meta["duration"]), "s", flush=True)
    rows = [json.load(open(j)) for j in sorted(out.glob("*.json"))]
    pd.DataFrame(rows).to_csv("data_cache/video_stats.csv", index=False)


if __name__ == "__main__":
    main()
