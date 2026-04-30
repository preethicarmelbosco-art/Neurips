#!/bin/bash
# Regenerate all 5 paper body figures from the result CSVs in ../../results/.
# Run from this directory:
#   bash make_all_figures.sh
set -e
cd "$(dirname "$0")"

python plot_fig1_pipeline.py
python plot_fig2_radar.py
python plot_fig3_ceiling_compression.py
python plot_fig6_discrimination.py
python plot_fig7_causal_signflip.py

echo ""
echo "All figures written to ../../figures/"
ls -lh ../../figures/*.pdf 2>/dev/null || echo "(figures directory will be created on first run)"
