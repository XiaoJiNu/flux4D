PANDASET_ROOT ?= /home/yr/yr/data/automonous/pandaset

.PHONY: pandaset-index
pandaset-index:
\tpython scripts/preprocess_flux4d.py \\
\t  --data-root $(PANDASET_ROOT) \\
\t  --out-full data/metadata/pandaset_full_clips.pkl \\
\t  --out-tiny data/metadata/pandaset_tiny_clips.pkl
