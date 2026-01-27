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

## Development notes

- Keep reproducible commands in this file when new scripts are added.
- Track progress and decisions under the development logs directory inside `docs/`.
- Follow `AGENTS.md` for coding style, docs, and versioning requirements.
