
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_model import SKEL, downsample, render

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rows = []
rng = np.random.default_rng(0)
for i in range(24):
    lab = i % 2
    n = 600
    frames = np.zeros((n, 48, 64, 3), np.uint8)
    for t in range(n):
        kp = SKEL.astype(float).copy() + rng.normal(0, 2, SKEL.shape)
        amp = 30 if lab else 8
        kp[4] += [0, amp * np.sin(t * (1.6 if lab else 0.4))]
        kp[3] += [0, amp / 2 * np.sin(t * (1.6 if lab else 0.4))]
        kp += [rng.normal(0, 3), 0]
        frames[t] = downsample(render(kp))
    np.savez_compressed(out / f"S{i:03d}.npz", frames=frames, occupancy=np.full(n, 0.05, np.float32))
    rows.append(dict(child_id=f"S{i:03d}", label=lab, split="train" if i < 16 else "test",
                     path="synthetic"))
pd.DataFrame(rows).to_csv(out.parent / "manifest_synth.csv", index=False)
print("done")
