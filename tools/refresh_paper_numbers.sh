#!/bin/bash
# Re-run the analysis pipeline end-to-end and rebuild the result CSVs that
# back the paper tables.
#
# Pipeline order matters because later scripts depend on earlier outputs:
#   1. bias_correction        -> writes table5_master_corrected.csv
#   2. eta2_family_ci         -> writes ca_eta2_family_ci.csv
#   3. ceiling_compression    -> writes family_primitive_modes.csv
#   4. inter_primitive_correlations -> writes inter_primitive_correlations.csv + moral_loo_sensitivity.csv
#   5. composition_regression -> writes table3_beta.csv + preliminary_regression.json
#   6. sign_stability_bootstrap -> writes table4_sign_stability.csv (B=5000)
#   7. kitchen_sink           -> writes kitchen_sink.csv
#   8. pairwise_lasso         -> writes pairwise_lasso.csv
#
# Inputs required (must already exist):
#   results/ceiling_compression/ca_all_rows.csv
#   results/cogbench/table5_master.csv
#   results/domain/domain_master.csv
#
# Run from repo root:
#   bash tools/refresh_paper_numbers.sh
set -e
cd "$(dirname "$0")/.."

echo "[1/8] bias_correction"
python code/analysis/bias_correction.py

echo "[2/8] eta2_family_ci"
python code/analysis/eta2_family_ci.py

echo "[3/8] ceiling_compression"
python code/analysis/ceiling_compression.py

echo "[4/8] inter_primitive_correlations"
python code/analysis/inter_primitive_correlations.py

echo "[5/8] composition_regression"
python code/analysis/composition_regression.py

echo "[6/8] sign_stability_bootstrap (B=5000, slow)"
python code/analysis/sign_stability_bootstrap.py

echo "[7/8] kitchen_sink"
python code/analysis/kitchen_sink.py

echo "[8/8] pairwise_lasso"
python code/analysis/pairwise_lasso.py

echo ""
echo "Done. Result tables written to results/composition/, results/cogbench/, results/ceiling_compression/."
