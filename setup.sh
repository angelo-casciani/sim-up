#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "[setup] Repository root: $REPO_ROOT"

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[setup] ERROR: '$cmd' is required but not installed." >&2
    exit 1
  fi
}

require_cmd python3.11
require_cmd poetry
require_cmd docker
require_cmd git

echo "[setup] Configuring Poetry to use Python 3.11..."
poetry env use python3.11 >/dev/null

echo "[setup] Installing project dependencies with Poetry..."
poetry install

echo "[setup] Ensuring SIMOD Docker image is available..."
if ! docker image inspect nokal/simod:latest >/dev/null 2>&1; then
  docker pull nokal/simod:latest
else
  echo "[setup] Docker image nokal/simod:latest already present."
fi

AGENT_SIM_PATH="$REPO_ROOT/src/AgentSimulator"
if [[ -f "$REPO_ROOT/.gitmodules" ]]; then
  echo "[setup] Syncing and initializing git submodules..."
  git submodule sync --recursive
  git submodule update --init --recursive src/AgentSimulator
elif [[ ! -d "$AGENT_SIM_PATH/.git" ]]; then
  echo "[setup] AgentSimulator submodule not configured; cloning fallback into src/AgentSimulator..."
  git clone https://github.com/lukaskirchdorfer/AgentSimulator "$AGENT_SIM_PATH"
else
  echo "[setup] AgentSimulator already present at src/AgentSimulator."
fi

echo "[setup] Verifying AgentSimulator import in Poetry environment..."
poetry run python - <<'PY'
import sys
sys.path.insert(0, 'src/AgentSimulator')
import source.agent_simulator
print('[setup] AgentSimulator import OK')
PY

echo "[setup] Done."
