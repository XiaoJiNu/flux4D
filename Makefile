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

.PHONY: stage3-overfit
stage3-overfit:
	$(PYTHON) scripts/train_flux4d.py \
	  --config configs/flux4d.py \
	  --index-path data/metadata/pandaset_tiny_clips.pkl \
	  --clip-index 0 \
	  --camera front_camera

STAGE6_RUN ?= run_001
STAGE6_ITERS ?= 30000
STAGE6_DEVICE ?= cuda:0
STAGE6_OUT ?= assets/vis/stage6_train/$(STAGE6_RUN)
STAGE6_CKPT ?= assets/vis/stage6_train/$(STAGE6_RUN)/ckpt_last.pt
STAGE6_EVAL_OUT ?= assets/vis/stage6_eval/$(STAGE6_RUN)

.PHONY: stage6-train
stage6-train:
	$(PYTHON) scripts/train_stage6.py \
	  --config configs/flux4d.py \
	  --index-path data/metadata/pandaset_full_clips.pkl \
	  --data-root $(PANDASET_ROOT) \
	  --device $(STAGE6_DEVICE) \
	  --iters $(STAGE6_ITERS) \
	  --output-dir $(STAGE6_OUT)

.PHONY: stage6-eval
stage6-eval:
	$(PYTHON) scripts/eval_stage6.py \
	  --config configs/flux4d.py \
	  --index-path data/metadata/pandaset_full_clips.pkl \
	  --data-root $(PANDASET_ROOT) \
	  --ckpt $(STAGE6_CKPT) \
	  --out-dir $(STAGE6_EVAL_OUT) \
	  --save-renders
