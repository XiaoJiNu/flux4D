# Flux4D Reproduction

Research notes and reproduction scaffolding for Flux4D experiments.

## Repository layout

- `assets/`: figures, tables, and other media assets.
- `configs/`: experiment and training configuration files.
- `data/`: metadata indices and caches (avoid raw datasets).
- `docs/`: paper notes, requirements, and development logs.
- `scripts/`: preprocessing/training/inference entry points.
- `src/`: core implementation code.
- `tests/`: automated tests (pytest style).
- `tools/`: visualization and debugging utilities.
- `third_party/`: vendored external dependencies.
- `utils/`: shared helpers and utilities.

## Quick start

### Stage 1: PandaSet clip indices

Generate full/tiny clip indices from the local PandaSet root:

```bash
python scripts/preprocess_flux4d.py \
  --data-root /home/yr/yr/data/automonous/pandaset \
  --out-full data/metadata/pandaset_full_clips.pkl \
  --out-tiny data/metadata/pandaset_tiny_clips.pkl
```

Optional flags:

- `--clip-len-s 1.5` and `--stride-s 1.5`
- `--tiny-scenes 001,002` or `--tiny-num-scenes 2`
- `--val-scenes 010,020`

Makefile shortcut:

```bash
make pandaset-index
```

### Stage 2: Lift alignment sanity check

Visualize LiDAR projection alignment for a single clip/frame:

```bash
python tools/vis/vis_lift_alignment.py \
  --index-path data/metadata/pandaset_tiny_clips.pkl \
  --clip-index 0 --frame-index 0 --camera front_camera
```

Outputs default to `assets/vis/lift_alignment/` (ignored by git).

Note:

- PandaSet LiDAR frames are stored as `.pkl.gz` pickles that require `pandas` to read.
  If your default `python` cannot import a compatible `pandas`, run scripts under the `gaussianstorm`
  conda env (or set `PYTHON=/home/yr/anaconda3/envs/gaussianstorm/bin/python` when using `make`).

### Stage 3: Voxelization sanity check (numpy-only)

Validate the stage3 voxelization pipeline (world → ego0, point_cloud_range filter, mean pooling):

```bash
python scripts/inspect_stage3_voxelization.py \
  --config configs/flux4d.py \
  --index-path data/metadata/pandaset_tiny_clips.pkl \
  --clip-index 0 \
  --num-sky-points 20000
```

### Stage 3: Tiny clip overfit (training loop)

Run a minimal stage3 training loop that connects:
Lift(G_init) → Flux4D-base → gsplat rasterization → loss → backward.

```bash
python scripts/train_flux4d.py \
  --config configs/flux4d.py \
  --index-path data/metadata/pandaset_tiny_clips.pkl \
  --clip-index 0 \
  --camera front_camera \
  --device cuda:0 \
  --num-sky-points 20000 \
  --max-gaussians 200000
```

To simulate a larger effective batch size without increasing peak memory, use gradient accumulation:

```bash
python scripts/train_flux4d.py --grad-accum-steps 4 ...
```

To truly overfit a single target frame (avoid multi-frame averaging blur), pin the supervision frame:

```bash
python scripts/train_flux4d.py --fixed-target-frame 1 ...
```

If you want to skip projected LiDAR depth supervision (faster debug):

```bash
python scripts/train_flux4d.py --no-projected-depth ...
```

The trainer writes `train.log` under `output_dir/`, debug renders under `output_dir/step_XXXXXX/`, and saves
checkpoints as `ckpt_step_XXXXXX.pt` (with `ckpt_last.pt` pointing to the latest checkpoint). To resume:

```bash
python scripts/train_flux4d.py --resume-from assets/vis/stage3_overfit/ckpt_last.pt ...
```

If the model parameter set changed (e.g. turning on iterative refinement), you may need to skip loading the
optimizer state:

```bash
python scripts/train_flux4d.py --resume-from ... --resume-no-optim ...
```

### Stage 4: Rendered velocity sanity check

Visualize image-plane rendered velocity `v_r` (used for dynamic reweighting in the supplementary A.1):

```bash
python tools/vis/vis_flow.py \
  --config configs/flux4d.py \
  --index-path data/metadata/pandaset_tiny_clips.pkl \
  --clip-index 0 \
  --camera front_camera \
  --ckpt assets/vis/stage3_overfit_run2/ckpt_last.pt \
  --out-dir assets/vis/stage4_flow_sanity/clip_000 \
  --mode both --render-frames 0-15 --frame-ref 0
```

### Stage 6: Full training + evaluation (PandaSet)

Run multi-clip / multi-scene training (paper-aligned slicing preset is built into the index):

```bash
python scripts/train_stage6.py \
  --config configs/flux4d.py \
  --index-path data/metadata/pandaset_full_clips.pkl \
  --data-root /home/yr/yr/data/automonous/pandaset \
  --device cuda:0 \
  --iters 30000 \
  --output-dir assets/vis/stage6_train/run_001
```

Evaluate NVS (PSNR/SSIM/Depth RMSE; full + dynamic) + Scene Flow metrics:

```bash
python scripts/eval_stage6.py \
  --config configs/flux4d.py \
  --index-path data/metadata/pandaset_full_clips.pkl \
  --data-root /home/yr/yr/data/automonous/pandaset \
  --ckpt assets/vis/stage6_train/run_001/ckpt_last.pt \
  --out-dir assets/vis/stage6_eval/run_001 \
  --save-renders
```

The evaluator writes `metrics.json` under `--out-dir/`. Note: PandaSet cuboids annotations are stored as
`pandas.DataFrame` pickles, so `pandas` must be available in your Python env.

## Development notes

- Keep reproducible commands in this file when new scripts are added.
- Track progress and decisions under the development logs directory inside `docs/`.
- Follow `AGENTS.md` for coding style, docs, and versioning requirements.
