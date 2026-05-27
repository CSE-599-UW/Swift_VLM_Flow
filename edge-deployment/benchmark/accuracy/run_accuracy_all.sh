#!/usr/bin/env bash
# =============================================================================
# run_accuracy_all.sh
# Run all accuracy benchmarks (VQAv2, POPE, MME) for both HF baseline and
# TRT engine, then print a side-by-side comparison summary.
#
# Usage:
#   # Default: baseline + TRT bf16, all tasks
#   bash run_accuracy_all.sh
#
#   # TRT only, specific precision
#   bash run_accuracy_all.sh --skip-baseline --precision int4
#
#   # Quick smoke-test (small sample counts)
#   bash run_accuracy_all.sh --quick
#
#   # Custom engine path
#   bash run_accuracy_all.sh --engine-dir /workspace/trt_engines/qwen2vl_int8 \
#                             --precision int8
#
# Flags:
#   --skip-baseline        Skip HF baseline run (use if already done)
#   --skip-trt             Skip TRT run
#   --precision PREC       TRT engine precision label [default: bf16]
#   --engine-dir PATH      TRT engine directory
#   --tasks "vqa pope mme" Which tasks to run (space-separated, quoted)
#   --vqa-samples N        VQAv2 sample count  [default: 500]
#   --pope-samples N       POPE samples/split  [default: 300]
#   --quick                Sets vqa=20, pope=30, mme=50 for a fast smoke-test
#   --output-tag TAG       Append tag to output filenames
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="/workspace/results/accuracy"
ENGINE_DIR="/workspace/trt_engines/qwen2vl"
PRECISION="bf16"
TASKS="vqa pope mme"
VQA_SAMPLES=500
POPE_SAMPLES=300
MME_SAMPLES=""          # empty = full dataset
SKIP_BASELINE=false
SKIP_TRT=false
QUICK=false
OUTPUT_TAG=""

# ── Argument Parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-baseline)  SKIP_BASELINE=true ;;
        --skip-trt)       SKIP_TRT=true ;;
        --precision)      PRECISION="$2";    shift ;;
        --engine-dir)     ENGINE_DIR="$2";   shift ;;
        --tasks)          TASKS="$2";        shift ;;
        --vqa-samples)    VQA_SAMPLES="$2";  shift ;;
        --pope-samples)   POPE_SAMPLES="$2"; shift ;;
        --mme-samples)    MME_SAMPLES="$2";  shift ;;
        --quick)          QUICK=true ;;
        --output-tag)     OUTPUT_TAG="$2";   shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
    shift
done

if $QUICK; then
    VQA_SAMPLES=20
    POPE_SAMPLES=30
    MME_SAMPLES=50
    echo "[INFO] --quick mode: vqa=$VQA_SAMPLES pope=$POPE_SAMPLES mme=$MME_SAMPLES"
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
RESET='\033[0m'

log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')]${RESET} $*"; }
ok()   { echo -e "${GREEN}[OK]${RESET} $*"; }
warn() { echo -e "${YELLOW}[WARN]${RESET} $*"; }
fail() { echo -e "${RED}[FAIL]${RESET} $*"; }

# Build common CLI args shared by both runners
build_common_args() {
    local args="--tasks $TASKS"
    args+=" --vqa_samples $VQA_SAMPLES"
    args+=" --pope_samples $POPE_SAMPLES"
    [[ -n "$MME_SAMPLES" ]] && args+=" --mme_samples $MME_SAMPLES"
    [[ -n "$OUTPUT_TAG"  ]] && args+=" --output_tag $OUTPUT_TAG"
    echo "$args"
}

# Find the most recent JSON file under a directory
latest_json() {
    local dir="$1"
    find "$dir" -name "*.json" -type f 2>/dev/null \
        | xargs ls -t 2>/dev/null \
        | head -1
}

# ── Print header ──────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}============================================================${RESET}"
echo -e "${BOLD}  VLM Accuracy Benchmark Suite${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo -e "  Tasks     : ${TASKS}"
echo -e "  Precision : ${PRECISION}"
echo -e "  Engine    : ${ENGINE_DIR}"
echo -e "  VQA       : ${VQA_SAMPLES} samples"
echo -e "  POPE      : ${POPE_SAMPLES} samples/split × 3 splits"
echo -e "  MME       : ${MME_SAMPLES:-all} samples"
echo -e "  Skip base : ${SKIP_BASELINE}"
echo -e "  Skip TRT  : ${SKIP_TRT}"
echo ""

START_TIME=$(date +%s)

# ── Step 1: HF Baseline ───────────────────────────────────────────────────────
BASELINE_JSON=""
if $SKIP_BASELINE; then
    warn "Skipping HF baseline run (--skip-baseline)"
    BASELINE_JSON=$(latest_json "$RESULTS_DIR/baseline")
    [[ -n "$BASELINE_JSON" ]] && log "Using existing baseline: $BASELINE_JSON"
else
    log "Starting HF baseline benchmark..."
    COMMON_ARGS=$(build_common_args)

    if python3 "$SCRIPT_DIR/run_accuracy_baseline.py" $COMMON_ARGS; then
        BASELINE_JSON=$(latest_json "$RESULTS_DIR/baseline")
        ok "Baseline complete → $BASELINE_JSON"
    else
        fail "Baseline benchmark failed!"
        exit 1
    fi
fi

# ── Step 2: TRT Engine ────────────────────────────────────────────────────────
TRT_JSON=""
if $SKIP_TRT; then
    warn "Skipping TRT run (--skip-trt)"
    TRT_JSON=$(latest_json "$RESULTS_DIR/trt")
    [[ -n "$TRT_JSON" ]] && log "Using existing TRT result: $TRT_JSON"
else
    log "Starting TRT benchmark (precision=$PRECISION)..."

    # Verify engine directory exists before launching
    if [[ ! -d "$ENGINE_DIR" ]]; then
        fail "TRT engine directory not found: $ENGINE_DIR"
        fail "Skipping TRT run. Re-run with --skip-trt if you only want the baseline."
        TRT_JSON=""
    else
        COMMON_ARGS=$(build_common_args)

        if python3 "$SCRIPT_DIR/run_accuracy_trt.py" \
                $COMMON_ARGS \
                --engine_dir "$ENGINE_DIR" \
                --precision  "$PRECISION"; then
            TRT_JSON=$(latest_json "$RESULTS_DIR/trt")
            ok "TRT complete → $TRT_JSON"
        else
            fail "TRT benchmark failed!"
            exit 1
        fi
    fi
fi

# ── Step 3: Comparison Summary ────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
ELAPSED_FMT="$(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"

echo ""
echo -e "${BOLD}============================================================${RESET}"
echo -e "${BOLD}  RESULTS SUMMARY  (elapsed: $ELAPSED_FMT)${RESET}"
echo -e "${BOLD}============================================================${RESET}"

python3 - "$BASELINE_JSON" "$TRT_JSON" << 'PYEOF'
import json, sys

def load(path):
    if not path or path == "None":
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] Could not load {path}: {e}")
        return None

def fmt(val, suffix=""):
    return f"{val:.2f}{suffix}" if isinstance(val, float) else str(val)

def delta(a, b, suffix=""):
    """Return coloured delta string (b - a)."""
    if a is None or b is None:
        return "  n/a"
    d = b - a
    sign = "+" if d >= 0 else ""
    # ANSI: green if >= 0, red if < 0
    color = "\033[0;32m" if d >= 0 else "\033[0;31m"
    reset = "\033[0m"
    return f"{color}{sign}{d:.2f}{suffix}{reset}"

baseline = load(sys.argv[1])
trt      = load(sys.argv[2])

COL = 22   # column width

def header(title):
    print(f"\n  \033[1m── {title} {'─'*(50-len(title))}\033[0m")
    if baseline and trt:
        print(f"  {'Metric':<{COL}} {'Baseline':>10}  {'TRT':>10}  {'Δ (TRT-Base)':>14}")
        print(f"  {'-'*58}")
    else:
        print(f"  {'Metric':<{COL}} {'Value':>10}")
        print(f"  {'-'*35}")

def row(label, base_val, trt_val, suffix=""):
    if baseline and trt:
        d = delta(base_val, trt_val, suffix)
        bv = fmt(base_val, suffix) if base_val is not None else "n/a"
        tv = fmt(trt_val,  suffix) if trt_val  is not None else "n/a"
        print(f"  {label:<{COL}} {bv:>10}  {tv:>10}  {d:>14}")
    else:
        val = base_val if base_val is not None else trt_val
        print(f"  {label:<{COL}} {fmt(val, suffix):>10}")

# ── VQAv2 ────────────────────────────────────────────────────────────────────
def get_vqa(data):
    try:    return data["results"]["vqa"]["scores"]["accuracy"]
    except: return None

if baseline or trt:
    has_vqa = (
        (baseline and "vqa" in baseline.get("results", {})) or
        (trt      and "vqa" in trt.get("results", {}))
    )
    if has_vqa:
        header("VQAv2")
        row("Accuracy", get_vqa(baseline), get_vqa(trt), "%")

# ── POPE ─────────────────────────────────────────────────────────────────────
def get_pope(data, key):
    try:    return data["results"]["pope"]["scores"][key]
    except: return None

def get_pope_split(data, split, key):
    try:    return data["results"]["pope"]["scores"]["per_split"][split][key]
    except: return None

if (baseline and "pope" in baseline.get("results", {})) or \
   (trt      and "pope" in trt.get("results",      {})):
    header("POPE")
    row("Avg Accuracy",   get_pope(baseline,"avg_accuracy"), get_pope(trt,"avg_accuracy"), "%")
    row("Avg F1",         get_pope(baseline,"avg_f1"),       get_pope(trt,"avg_f1"),       "%")
    for split in ["random","popular","adversarial"]:
        row(f"  {split} / Acc",
            get_pope_split(baseline, split, "accuracy"),
            get_pope_split(trt,      split, "accuracy"), "%")
        row(f"  {split} / F1",
            get_pope_split(baseline, split, "f1"),
            get_pope_split(trt,      split, "f1"),       "%")

# ── MME ──────────────────────────────────────────────────────────────────────
def get_mme(data, key):
    try:    return float(data["results"]["mme"]["scores"][key])
    except: return None

if (baseline and "mme" in baseline.get("results", {})) or \
   (trt      and "mme" in trt.get("results",      {})):
    header("MME")
    row("Total Score",      get_mme(baseline,"total_score"),      get_mme(trt,"total_score"))
    row("Perception Score", get_mme(baseline,"perception_score"), get_mme(trt,"perception_score"))
    row("Cognition Score",  get_mme(baseline,"cognition_score"),  get_mme(trt,"cognition_score"))

# ── File paths ────────────────────────────────────────────────────────────────
print()
if baseline: print(f"  Baseline → {sys.argv[1]}")
if trt:      print(f"  TRT      → {sys.argv[2]}")
print()
PYEOF

echo ""