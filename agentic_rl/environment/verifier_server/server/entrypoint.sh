#!/usr/bin/env bash
set -uo pipefail

# Set defaults for parallelism if not provided
: "${GJ_PARALLELISM:=$(nproc)}"
: "${JUDGE_WORKERS:=$(nproc)}"
# Node's default max-old-space is ~4GB regardless of machine RAM; grading bursts
# (workers x concurrent cases x buffered stdout) need far more headroom.
: "${NODE_MAX_OLD_SPACE_MB:=65536}"
export GJ_PARALLELISM JUDGE_WORKERS

echo "[entrypoint] GJ_PARALLELISM=${GJ_PARALLELISM}, JUDGE_WORKERS=${JUDGE_WORKERS}, NODE_MAX_OLD_SPACE_MB=${NODE_MAX_OLD_SPACE_MB}"

# Both processes run under restart loops: a crash of either (e.g. node heap OOM
# under a grading burst) must degrade to a few errored submissions, not kill the
# sandbox for the rest of the training run. The tunnel URL survives restarts.
# Exception: exit code 86 is the server's idle self-termination (no requests for
# JUDGE_IDLE_EXIT_MIN — the owning run is presumed dead), so exit the entrypoint
# instead of restarting, which terminates the sandbox.
IDLE_EXIT_CODE=86
(
  while true; do
    /usr/local/bin/go-judge -parallelism "${GJ_PARALLELISM}"
    echo "[entrypoint] go-judge exited (code=$?), restarting in 1s"
    sleep 1
  done
) &
sleep 0.5
while true; do
  node --max-old-space-size="${NODE_MAX_OLD_SPACE_MB}" /app/server.js
  code=$?
  if [ "${code}" -eq "${IDLE_EXIT_CODE}" ]; then
    echo "[entrypoint] node exited with idle code ${code}; terminating sandbox"
    exit 0
  fi
  echo "[entrypoint] node exited (code=${code}), restarting in 1s"
  sleep 1
done
