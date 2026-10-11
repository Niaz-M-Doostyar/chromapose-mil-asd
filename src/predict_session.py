




import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from model import ChromaPoseMIL


def clip_starts(frames, occupancy, clip_len=32, hop=16):
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (48, 64, 3):
        raise ValueError('Expected uint8 frames with shape (T, 48, 64, 3)')
    if not len(frames) or occupancy.shape != (len(frames),) or not np.isfinite(occupancy).all():
        raise ValueError('Empty session or invalid occupancy array')
    candidates = np.arange(0, max(len(frames) - clip_len, 0) + 1, hop)
    keep = np.array([s for s in candidates if (occupancy[s:s+clip_len] > 1e-4).mean() >= .5])
    return keep if len(keep) >= 8 else candidates


@torch.no_grad()
def infer(checkpoint, session, device='cpu', chunk=32, manifest=None):
    if chunk < 1:
        raise ValueError('Chunk size must be positive')
    checkpoint = Path(checkpoint)
    session = Path(session)
    device = torch.device(device)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS was requested but is unavailable')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg = saved['cfg']
    model = ChromaPoseMIL(cfg).to(device).eval()
    model.load_state_dict(saved['state'])
    lazy = None
    if cfg.get('_loader') == 'lazy':
        if not session.is_dir():
            raise ValueError('This checkpoint requires a completed higher-resolution session directory')
        from lazy_cohort import LazyCohort
        options = {'manifest': manifest} if manifest is not None else {}
        lazy = LazyCohort(device, frames_dir=session.parent, child_ids=[session.name],
                          clip_len=cfg['clip_len'], verify_hashes=True, **options)
        metadata = json.loads(lazy.metadata_paths[session.name].read_text())
        provenance_path = checkpoint.parent / 'cache_provenance.json'
        if not provenance_path.is_file():
            lazy.close()
            raise ValueError('Higher-resolution checkpoint is missing cache provenance')
        training = json.loads(provenance_path.read_text())['parameters']
        reference = next(iter(training.values()))

        input_keys = ('width', 'height', 'fps', 'resize', 'dilation', 'color',
                      'source_occupancy_threshold', 'empty_occupancy_threshold')
        if any(metadata['parameters'].get(k) != reference.get(k) for k in input_keys):
            lazy.close()
            raise ValueError('Session preprocessing differs from the trained checkpoint')
        starts = lazy.all_clips(session.name)[::cfg.get('eval_hop', 1)]
        empty_fraction = metadata['empty_frame_fraction']
        spatial_shape, fps = lazy.spatial_shape, lazy.fps
        gather = lambda selected: lazy.gather(session.name, selected)
    else:
        if session.is_dir():
            raise ValueError('This checkpoint requires the original 64x48 NPZ session')
        with np.load(session, allow_pickle=False) as z:
            frames, occupancy = z['frames'], z['occupancy']
        starts = clip_starts(frames, occupancy, cfg['clip_len'])[::cfg.get('eval_hop', 1)]
        empty_fraction = float((occupancy <= 1e-4).mean())
        spatial_shape, fps = frames.shape[1:], 8
        def gather(selected):
            ix = selected[:, None] + np.arange(cfg['clip_len'])[None]
            return torch.from_numpy(frames[np.minimum(ix, len(frames)-1)]).to(device)
    amp = device.type == 'mps' and cfg.get('amp', False)
    embeddings = []
    try:
        for i in range(0, len(starts), chunk):
            x = gather(starts[i:i+chunk])
            with torch.autocast('mps', dtype=torch.float16, enabled=amp):
                embeddings.append(model.encode(x).float())
    finally:
        if lazy is not None:
            lazy.close()
    h = torch.cat(embeddings)[None]
    logit, attention = model.aggregate(h)
    return dict(checkpoint=str(checkpoint.resolve()), session=str(session.resolve()),
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                research_score=torch.sigmoid(logit).item(),
                attention=attention[0].cpu().tolist(), clip_start_frames=starts.tolist(),
                frame_rate=fps, spatial_shape=list(spatial_shape), device=str(device), mixed_precision=amp,
                empty_frame_fraction=empty_fraction,
                interpretation='Uncalibrated research score; not a clinical diagnosis.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--session', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    ap.add_argument('--chunk', type=int, default=32)
    ap.add_argument('--manifest', type=Path, help='Manifest for a higher-resolution session directory')
    a = ap.parse_args()
    result = infer(a.checkpoint, a.session, a.device, a.chunk, a.manifest)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    print(f"Research score: {result['research_score']:.6f}; output: {a.output}")


if __name__ == '__main__':
    main()
