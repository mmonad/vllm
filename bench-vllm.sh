#!/bin/bash
# vLLM benchmark script — works with cuda/, rocm/, and remote/ configs.
#
# Benchmarks a RUNNING vLLM server by sending requests via the OpenAI API.
# Reads host/port/model from the same config YAML used by run-vllm.sh.
# Detects backend (cuda/rocm/remote) from the config path.
#
# Usage:
#   ./run-vllm.sh configs/cuda/qwen3.5-27b-fp8.yaml       # start server
#   ./bench-vllm.sh configs/cuda/qwen3.5-27b-fp8.yaml     # benchmark it
#
#   ./bench-vllm.sh configs/remote/qwen3.5-35b-a3b-fp8-spark1.yaml  # remote
#
# Results are saved to: bench-results/<backend>/<config-name>/<timestamp>/
#
# Environment variables:
#   BENCH_INPUT_LENS    Space-separated input lengths    (default: "128 512 2048")
#   BENCH_OUTPUT_LENS   Space-separated output lengths   (default: "128 512")
#   BENCH_NUM_PROMPTS   Number of prompts per run        (default: 10)
#   BENCH_REQUEST_RATES Space-separated req/s rates      (default: "inf")
#   BENCH_NUM_WARMUPS   Warmup requests                  (default: 2)
#   BENCH_HOST          Override server host              (default: from config)
#   BENCH_PORT          Override server port              (default: from config)

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml> [extra vllm bench serve args...]}"
shift

# =============================================================================
# Activate venv
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

# =============================================================================
# Detect backend from config path
# =============================================================================
case "$CONFIG" in
    configs/cuda/*)    HW_BACKEND="cuda" ;;
    configs/rocm/*)    HW_BACKEND="rocm" ;;
    configs/remote/*)  HW_BACKEND="remote" ;;
    *)
        echo "error: cannot determine backend from config path." >&2
        echo "Place configs under configs/cuda/, configs/rocm/, or configs/remote/." >&2
        exit 1
        ;;
esac

# =============================================================================
# Benchmark parameters (override via env vars)
# =============================================================================
read -ra INPUT_LENS <<< "${BENCH_INPUT_LENS:-128 512 2048}"
read -ra OUTPUT_LENS <<< "${BENCH_OUTPUT_LENS:-128 512}"
NUM_PROMPTS="${BENCH_NUM_PROMPTS:-10}"
read -ra REQUEST_RATES <<< "${BENCH_REQUEST_RATES:-inf}"
NUM_WARMUPS="${BENCH_NUM_WARMUPS:-2}"

# =============================================================================
# Parse config YAML for server connection + model info
# =============================================================================
read_config() {
    python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('host', '127.0.0.1'))
print(cfg.get('port', 8080))
print(cfg.get('served-model-name', cfg.get('model', '')))
print(cfg.get('model', ''))
print('true' if cfg.get('trust-remote-code', False) else 'false')
print(cfg.get('endpoint', '/v1/completions'))
ep = cfg.get('endpoint', '/v1/completions')
print('openai-chat' if 'chat' in ep else 'openai')
"
}

mapfile -t _cfg < <(read_config)
HOST="${BENCH_HOST:-${_cfg[0]}}"
PORT="${BENCH_PORT:-${_cfg[1]}}"
MODEL="${_cfg[2]}"
TOKENIZER="${_cfg[3]}"
TRUST_REMOTE="${_cfg[4]}"
ENDPOINT="${_cfg[5]}"
BACKEND="${_cfg[6]}"

BASE_URL="http://${HOST}:${PORT}"

# Refuse to benchmark localhost for remote configs (likely misconfigured)
if [ "$HW_BACKEND" = "remote" ] && [ "$HOST" = "127.0.0.1" ]; then
    echo "error: remote config resolved to localhost." >&2
    echo "Set 'host' in the config YAML or export BENCH_HOST=<hostname>." >&2
    exit 1
fi

# Extract config name for results directory
CONFIG_NAME=$(basename "$CONFIG" .yaml)

# =============================================================================
# GPU info helper (adapts to backend)
# =============================================================================
get_gpu_info() {
    case "$HW_BACKEND" in
        cuda)
            local gpu
            gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) || gpu="unknown"
            local driver
            driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1) || driver="unknown"
            echo "GPU:       $gpu"
            echo "Driver:    $driver"
            echo "CUDA:      $(nvcc --version 2>/dev/null | grep 'release' | sed 's/.*release //' | sed 's/,.*//' || echo 'unknown')"
            ;;
        rocm)
            echo "GPU:       $(rocm-smi --showproductname 2>/dev/null | grep 'Card Series' | head -1 | sed 's/.*: *//' || echo 'unknown')"
            echo "ROCm:      $(cat /opt/rocm/.info/version 2>/dev/null || echo 'unknown')"
            ;;
        remote)
            echo "GPU:       remote (${HOST})"
            ;;
        *)
            echo "GPU:       unknown"
            ;;
    esac
}

# =============================================================================
# Wait for server to be ready
# =============================================================================
echo "Checking server at ${BASE_URL}/v1/models ..."
if ! curl -sf --max-time 5 "${BASE_URL}/v1/models" > /dev/null 2>&1; then
    echo "ERROR: Server not reachable at ${BASE_URL}"
    echo "Start it first:  ./run-vllm.sh $CONFIG"
    exit 1
fi
echo "Server is up. Model: ${MODEL}"
echo ""

# =============================================================================
# Output directory
# =============================================================================
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="$SCRIPT_DIR/bench-results/$HW_BACKEND/$CONFIG_NAME/$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

SUMMARY="$RESULTS_DIR/summary.txt"
GIT_SHA=$(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Copy the config YAML into results for full traceability
cp "$CONFIG" "$RESULTS_DIR/config.yaml"

# Write header
{
    echo "================================================================"
    echo "vLLM Benchmark Results"
    echo "================================================================"
    echo "Config:    $CONFIG"
    echo "Name:      $CONFIG_NAME"
    echo "Backend:   $HW_BACKEND"
    echo "Date:      $(date -Iseconds)"
    echo "Git SHA:   $GIT_SHA"
    echo "Server:    ${BASE_URL}"
    echo "Model:     ${MODEL}"
    get_gpu_info
    echo ""
    echo "--- Config Contents ---"
    cat "$CONFIG"
    echo "--- End Config ---"
    echo "================================================================"
    echo ""
} | tee "$SUMMARY"

# =============================================================================
# Helper: run a single bench serve and extract metrics
# =============================================================================
run_bench() {
    local label="$1"
    local json_file="$2"
    shift 2
    local extra_args=("$@")

    echo "--- $label ---" | tee -a "$SUMMARY"

    # Build the command
    local cmd=(vllm bench serve)
    cmd+=(--backend "$BACKEND" --base-url "$BASE_URL" --model "$MODEL")
    cmd+=(--endpoint "$ENDPOINT" --tokenizer "$TOKENIZER")
    cmd+=(--save-result --result-dir "$RESULTS_DIR")
    cmd+=(--result-filename "$(basename "$json_file")")
    cmd+=(--num-warmups "$NUM_WARMUPS")
    cmd+=(--percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "50,90,99")
    if [ "$TRUST_REMOTE" = "true" ]; then
        cmd+=(--trust-remote-code)
    fi
    cmd+=("${extra_args[@]}")

    echo "  Command: ${cmd[*]}" | tee -a "$SUMMARY"

    if "${cmd[@]}" 2>&1 | tee -a "$RESULTS_DIR/full.log"; then
        # Extract key metrics from the saved JSON
        python3 -c "
import json, sys

with open('$json_file') as f:
    d = json.load(f)

lines = []
# Throughput
req_s = d.get('request_throughput', 0)
out_s = d.get('output_throughput', 0)
tot_s = d.get('total_token_throughput', 0)
lines.append(f'  Throughput:  {req_s:.2f} req/s | {out_s:.1f} out tok/s | {tot_s:.1f} total tok/s')

# Prefill (TTFT)
ttft = d.get('mean_ttft_ms')
if ttft is not None:
    p50 = d.get('p50_ttft_ms', 0)
    p99 = d.get('p99_ttft_ms', 0)
    lines.append(f'  Prefill:     mean {ttft:.1f} ms | P50 {p50:.1f} ms | P99 {p99:.1f} ms')

# Decode (TPOT)
tpot = d.get('mean_tpot_ms')
if tpot is not None:
    p50 = d.get('p50_tpot_ms', 0)
    p99 = d.get('p99_tpot_ms', 0)
    lines.append(f'  Decode:      mean {tpot:.1f} ms/tok | P50 {p50:.1f} ms | P99 {p99:.1f} ms')

# End-to-end
e2e = d.get('mean_e2el_ms')
if e2e is not None:
    p50 = d.get('p50_e2el_ms', 0)
    p99 = d.get('p99_e2el_ms', 0)
    lines.append(f'  E2E:         mean {e2e:.1f} ms | P50 {p50:.1f} ms | P99 {p99:.1f} ms')

dur = d.get('duration', 0)
n = d.get('completed', 0)
lines.append(f'  Completed:   {n} requests in {dur:.1f}s')

print('\n'.join(lines))
" | tee -a "$SUMMARY"
    else
        echo "  FAILED (see full.log for details)" | tee -a "$SUMMARY"
    fi
    echo "" | tee -a "$SUMMARY"
}

# =============================================================================
# Run benchmarks: sweep input_len x output_len x request_rate
# =============================================================================
for req_rate in "${REQUEST_RATES[@]}"; do
    rate_label=""
    if [ "$req_rate" != "inf" ]; then
        rate_label=" rate=${req_rate}rps"
    fi

    echo "================ BENCHMARKS${rate_label} ========================" | tee -a "$SUMMARY"
    echo "" | tee -a "$SUMMARY"

    for input_len in "${INPUT_LENS[@]}"; do
        for output_len in "${OUTPUT_LENS[@]}"; do
            label="in=${input_len} out=${output_len} n=${NUM_PROMPTS}${rate_label}"
            json_file="$RESULTS_DIR/serve_in${input_len}_out${output_len}_rate${req_rate}.json"

            run_bench "$label" "$json_file" \
                --dataset-name random \
                --input-len "$input_len" \
                --output-len "$output_len" \
                --num-prompts "$NUM_PROMPTS" \
                --request-rate "$req_rate" \
                --ignore-eos
        done
    done
done

# =============================================================================
# Final summary
# =============================================================================
{
    echo "================================================================"
    echo "Benchmark complete: $(date -Iseconds)"
    echo "Results saved to:   $RESULTS_DIR/"
    echo "Summary:            $SUMMARY"
    echo "================================================================"
} | tee -a "$SUMMARY"
