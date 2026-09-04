#!/usr/bin/env bash
# Parameter-sensitivity sweep for the evaluation.
#
# For one dataset and one perturbed knob, runs the full grid at every
# value in the knob's grid. Each run lands under
#     results/<dataset>/<timestamp>_<label>/
# so the three (or more) summaries don't trample each other.
#
# Knobs and their grids:
#     spectral   — --spectral-window-size   in {16, 32, 64}     → labels w16/w32/w64
#     train_frac — --train-fraction         in {0.2, 0.3, 0.5}  → labels tf02/tf03/tf05
#     n_folds    — --n-folds                in {3, 5, 10}       → labels k3/k5/k10
#     agg_window — --aggregate-query-window in {32, 64, 128}    → labels aw32/aw64/aw128
#     all        — run all four knob sweeps back-to-back
#
# Per-run logs land under results/_sensitivity/<timestamp>/<dataset>__<knob><val>.log.
# Each value produces its own summary.csv for the parameter-robustness analysis.
#
# Usage:
#     ./scripts/run_sensitivity.sh uci_har spectral
#     ./scripts/run_sensitivity.sh uci_har train_frac
#     ./scripts/run_sensitivity.sh uci_har n_folds
#     ./scripts/run_sensitivity.sh uci_har agg_window
#     ./scripts/run_sensitivity.sh uci_har all
#
# Fails fast on the first run that errors out.

set -eu
cd "$(dirname "$0")/.."

PY="${PYTHON:-/Library/Frameworks/Python.framework/Versions/3.13/bin/python3}"

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <dataset> <spectral|train_frac|n_folds|agg_window|all>" >&2
    exit 1
fi

DATASET="$1"
KNOB="$2"

STAMP=$(date '+%Y%m%d_%H%M%S')
LOG_DIR="results/_sensitivity/${STAMP}"
mkdir -p "$LOG_DIR"

echo "[sensitivity] stamp=$STAMP  dataset=$DATASET  knob=$KNOB"
echo "[sensitivity] logs -> $LOG_DIR"
echo ""

# ---------------------------------------------------------------------------
# Run one (knob, value) pair. Writes log; honours `set -e` on failure.
# ---------------------------------------------------------------------------
_run_one () {
    local flag="$1"        # CLI flag, e.g. --spectral-window-size
    local value="$2"       # value passed to the flag
    local label="$3"       # short tag for the output stamp (e.g. w16)
    local log="${LOG_DIR}/${DATASET}__${label}.log"
    local t0
    t0=$(date +%s)
    echo "[sensitivity] >>> ${flag} ${value}  label=${label}  (log: $log)"
    if "$PY" scripts/run_eval.py \
        --dataset "$DATASET" \
        "$flag" "$value" \
        --label "$label" \
        >"$log" 2>&1; then
        local elapsed=$(( $(date +%s) - t0 ))
        local artefact
        artefact=$(grep -E '^\[eval\] artefacts' "$log" | tail -1 || true)
        echo "[sensitivity] <<< ${label}  OK  (${elapsed}s)  $artefact"
    else
        local elapsed=$(( $(date +%s) - t0 ))
        echo "[sensitivity] <<< ${label}  FAILED  (${elapsed}s)  see $log"
        exit 1
    fi
}

_sweep_spectral () {
    for w in 16 32 64; do
        _run_one --spectral-window-size "$w" "w${w}"
    done
}

_sweep_train_frac () {
    # Label collapses the decimal point: 0.2 → tf02, 0.3 → tf03, 0.5 → tf05.
    for tf in 0.2 0.3 0.5; do
        label="tf$(echo "$tf" | tr -d '.')"
        _run_one --train-fraction "$tf" "$label"
    done
}

_sweep_n_folds () {
    for k in 3 5 10; do
        _run_one --n-folds "$k" "k${k}"
    done
}

_sweep_agg_window () {
    for aw in 32 64 128; do
        _run_one --aggregate-query-window "$aw" "aw${aw}"
    done
}

case "$KNOB" in
    spectral)   _sweep_spectral ;;
    train_frac) _sweep_train_frac ;;
    n_folds)    _sweep_n_folds ;;
    agg_window) _sweep_agg_window ;;
    all)
        _sweep_spectral
        _sweep_train_frac
        _sweep_n_folds
        _sweep_agg_window
        ;;
    *)
        echo "unknown knob: $KNOB (expected: spectral | train_frac | n_folds | agg_window | all)" >&2
        exit 1
        ;;
esac

echo ""
echo "[sensitivity] sweep complete. Per-run summaries under results/${DATASET}/<stamp>_<label>/summary.csv"
echo "[sensitivity] sweep log dir: ${LOG_DIR}"
