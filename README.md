# Flux4D Reproduction

## Stage 1: PandaSet clip indices

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
