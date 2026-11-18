#!/usr/bin/env bash
# Run a script inside the Isaac Lab docker image (same recipe as LunarSim).
# Usage: scripts/run_isaaclab.sh isaac/fidelity_test.py [args...]
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${DRONERACING_ISAACLAB_IMAGE:-lunarsim-isaaclab:6.0.1}"
CACHE_ROOT="${LUNARSIM_ISAAC_CACHE:-$HOME/docker/isaac-sim}"
mkdir -p "$CACHE_ROOT"/cache/{ov,pip,glcache,kit} "$CACHE_ROOT"/{data,documents,pylibs} "$PROJECT_ROOT/runs"
SCRIPT="$1"; shift
# watchdog: Isaac sometimes hangs at start-up; stop the container after ISAAC_TIMEOUT seconds so the caller can retry
NAME="dr_$$_$RANDOM"
( exec >/dev/null 2>&1 </dev/null; for fd in /proc/$BASHPID/fd/*; do n=${fd##*/}; [ "$n" -gt 2 ] && eval "exec $n>&-"; done; sleep "${ISAAC_TIMEOUT:-2400}"; docker stop "$NAME" ) &  # no inherited stdout and no inherited lock fd (flock): callers must not wait for the watchdog
WD=$!
trap 'kill $WD 2>/dev/null || true' EXIT
docker run --rm --name "$NAME" -e NOARROW="${NOARROW:-}" --gpus all --network=host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e PYTHONUNBUFFERED=1 \
  -v "$PROJECT_ROOT":/workspace/DroneRacing \
  -v "$CACHE_ROOT/cache/ov":/root/.cache/ov \
  -v "$CACHE_ROOT/cache/pip":/root/.cache/pip \
  -v "$CACHE_ROOT/cache/glcache":/root/.cache/nvidia \
  -v "$CACHE_ROOT/cache/kit":/root/.cache/kit \
  -v "$CACHE_ROOT/data":/root/.local/share/ov \
  -v "$CACHE_ROOT/documents":/root/Documents \
  -v "$CACHE_ROOT/pylibs":/root/.local \
  --entrypoint /workspace/IsaacLab/_isaac_sim/python.sh \
  -w /workspace/IsaacLab \
  "$IMAGE" "/workspace/DroneRacing/$SCRIPT" --headless "$@"
chmod -R a+rwX "$PROJECT_ROOT/runs" 2>/dev/null || true
