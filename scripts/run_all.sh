#!/usr/bin/env bash
# Sweep the evaluation across the classification-capable portfolio.
#
# Every dataset in the default set carries BOTH a forecasting head (raw
# sensor signal as target) and a classification head (native activity /
# fault label) — required so the task-aware vs task-agnostic comparison
# has a sample size > 1 on each task family.
#
# One log file per dataset under results/_sweep/<timestamp>/<dataset>.log.
# Per-dataset summary.csv lands under results/<dataset>/<eval_stamp>/.
#
# Usage:
#     ./scripts/run_all.sh
#     ./scripts/run_all.sh uci_har metro   # subset
#
# Fails fast on the first dataset that errors out; comment out `set -e` to
# keep going through partial failures.

set -eu
cd "$(dirname "$0")/.."

# Resolve the Python interpreter:
#   1. honour an explicit $PYTHON override (laptop dev workflow)
#   2. fall back to the first python3 on $PATH (works on macOS, Linux, HPC)
PY="${PYTHON:-$(command -v python3)}"
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
    echo "[sweep] ERROR: no usable python3 found (PYTHON=$PYTHON)" >&2
    exit 1
fi

# BLAS thread pinning is OPT-IN. On many-core HPC nodes (e.g. 28-core VM)
# the per-call thread-launch overhead of small cdist matrices can dominate
# the actual compute. On a laptop the opposite is true — multi-threaded
# BLAS is faster. Default to letting BLAS choose; pin to 1 thread per
# process by setting SWEEP_PIN_THREADS=1:
#     SWEEP_PIN_THREADS=1 ./scripts/run_all.sh
if [ "${SWEEP_PIN_THREADS:-0}" = "1" ]; then
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
fi

DEFAULT_DATASETS=(
    uci_har                         # HAR — smartphone IMU (50 Hz, 6 classes)
    opportunity                     # OPPORTUNITY HAR — body-worn IMUs (30 Hz, 5 classes)
    hydraulic                       # UCI Hydraulic Systems — condition monitoring (1 Hz, 4 valve states)
    beijing                         # UCI Beijing air-quality — PM2.5 forecast / wind-quadrant (1 h, 4 classes)
    metro                           # UCI Metro traffic — volume forecast / weekday-weekend (1 h, 2 classes)
)
# Resolve the dataset list. The shortcut `("${@:-${DEFAULT_DATASETS[@]}}")`
# is broken across bash versions — when $@ is empty the array-default
# substitution can collapse into a single space-joined element. Use an
# explicit branch instead so quoting is unambiguous.
if [ "$#" -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("${DEFAULT_DATASETS[@]}")
fi

STAMP=$(date '+%Y%m%d_%H%M%S')
SWEEP_DIR="results/_sweep/${STAMP}"
mkdir -p "$SWEEP_DIR"

echo "[sweep] stamp=$STAMP  datasets=${DATASETS[*]}"
echo "[sweep] logs -> $SWEEP_DIR"
echo ""

for ds in "${DATASETS[@]}"; do
    log="${SWEEP_DIR}/${ds}.log"
    t0=$(date +%s)
    echo "[sweep] >>> $ds  (log: $log)"
    if "$PY" -u scripts/run_eval.py --dataset "$ds" >"$log" 2>&1; then
        elapsed=$(( $(date +%s) - t0 ))
        # Pull the artefact path from the log if present.
        artefact=$(grep -E '^\[eval\] artefacts' "$log" | tail -1 || true)
        echo "[sweep] <<< $ds  OK  (${elapsed}s)  $artefact"
    else
        elapsed=$(( $(date +%s) - t0 ))
        echo "[sweep] <<< $ds  FAILED  (${elapsed}s)  see $log"
        exit 1
    fi
done

echo ""
echo "[sweep] all datasets complete."
