#!/usr/bin/env bash
# Re-run the Frontier-CS judge concurrency sweep at a given worker count, holding the
# workload constant: reuse the cpu=8 baseline's calibrated oracles.json + stratification
# (same problems, same seed -> identical phase2 picks) and vary only --judge-cpu/--judge-mem.
# Boots a fresh judge, WARMS the pool (compile checkers) so the sweep is apples-to-apples
# with the warm cpu=8 baseline, runs phase2, writes the report, tears the sandbox down.
#
#   agentic_rl/profiles/sweep_cpu.sh <CPU> <MEM_MB>
set -euo pipefail
cd /Users/junlin/Documents/Research/async-rl/slime
CPU="$1"; MEM="$2"
BASE=agentic_rl/profiles/frontier_cs_judge/20260625-113112   # the cpu=8 baseline run
RD="agentic_rl/profiles/frontier_cs_judge/cpu${CPU}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RD"
cp "$BASE/oracles.json" "$BASE/stratification.json" "$RD/"
export MODAL_ENVIRONMENT=junlin-dev
P=(uv run --no-dev python agentic_rl/profiles/judge_profile.py)
"${P[@]}" boot     --run-dir "$RD" --judge-cpu "$CPU" --judge-mem "$MEM"
"${P[@]}" warm     --run-dir "$RD"
"${P[@]}" phase2   --run-dir "$RD"
"${P[@]}" report   --run-dir "$RD"
"${P[@]}" teardown --run-dir "$RD"
echo "$RD" >> agentic_rl/profiles/frontier_cs_judge/SWEEP_RUNS
echo "DONE cpu=$CPU mem=$MEM rd=$RD"
