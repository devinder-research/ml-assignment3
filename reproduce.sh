#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

build_report() {
    "$PYTHON" scripts/make_figures.py
    "$PYTHON" report/build_report.py --commit "$(git rev-parse HEAD)" "$@"
}

case "${1:-}" in
    report)
        shift
        build_report "$@"
        ;;
    full)
        shift
        if [[ -e data/q4_sweep ]]; then
            echo "Existing sweep rows will be regenerated for the newly trained encoder."
        fi
        "$PYTHON" scripts/make_split.py
        "$PYTHON" scripts/q1a_comparison.py
        "$PYTHON" scripts/q1b_memorization.py
        "$PYTHON" scripts/q2a_cnn_vs_mlp.py
        "$PYTHON" scripts/q2b_permuted.py
        "$PYTHON" scripts/q2c_translation.py
        "$PYTHON" scripts/q3a_autoencoders.py
        "$PYTHON" scripts/q3b_denoising.py
        "$PYTHON" scripts/q3a_recon_grid.py
        "$PYTHON" scripts/q3c_representations.py
        "$PYTHON" scripts/q4_features.py
        for kernel in linear poly rbf; do
            for c in 0.01 0.1 1 10 100; do
                "$PYTHON" scripts/q4a_svm_sweep.py --kernel "$kernel" --C "$c"
            done
        done
        "$PYTHON" scripts/q4a_svm_sweep.py --finalize
        "$PYTHON" scripts/q4b_hard_pairs.py
        "$PYTHON" scripts/q4c_clustering.py
        build_report "$@"
        ;;
    *)
        echo "Usage: bash reproduce.sh {report|full} [--repo-url URL] [--output PATH]" >&2
        exit 2
        ;;
esac
