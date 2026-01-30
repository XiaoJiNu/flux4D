PYTHON ?= python
PANDASET_ROOT ?= /home/yr/yr/data/automonous/pandaset

.PHONY: pandaset-index
pandaset-index:
	$(PYTHON) scripts/preprocess_flux4d.py \
	  --preset paper \
	  --data-root $(PANDASET_ROOT) \
	  --out-full data/metadata/pandaset_full_clips.pkl \
	  --out-tiny data/metadata/pandaset_tiny_clips.pkl

.PHONY: stage3-voxel-sanity
stage3-voxel-sanity:
	$(PYTHON) scripts/inspect_stage3_voxelization.py \
	  --config configs/flux4d.py \
	  --index-path data/metadata/pandaset_tiny_clips.pkl \
	  --clip-index 0 \
	  --num-sky-points 20000 \
	  --out-dir assets/vis/stage3_voxel_sanity/clip_000 \
	  --plot-window-m 400
