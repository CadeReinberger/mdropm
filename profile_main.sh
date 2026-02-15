#!/usr/bin/env bash

# If sourced, avoid mutating/killing the caller's shell.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  echo "Run this script, do not source it: bash profile_main.sh [args...]" >&2
  return 1
fi

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${ROOT}/out/py-spy"
mkdir -p "$OUT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT_DIR}/profile_${TS}.svg"

PYSPY_BIN="${PYSPY_BIN:-py-spy}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TIMEOUT="${RUN_TIMEOUT:-5m}"
PROFILE_DURATION="${PROFILE_DURATION:-240}"

# Resolve py-spy path robustly (including common sudo/root PATH cases).
if ! command -v "$PYSPY_BIN" >/dev/null 2>&1; then
  if [[ "$PYSPY_BIN" == "py-spy" ]]; then
    CANDIDATE="/home/${SUDO_USER:-$USER}/.local/bin/py-spy"
    if [[ -x "$CANDIDATE" ]]; then
      PYSPY_BIN="$CANDIDATE"
    fi
  fi
fi

if ! command -v "$PYSPY_BIN" >/dev/null 2>&1 && [[ ! -x "$PYSPY_BIN" ]]; then
  echo "py-spy not found. Try one of:" >&2
  echo "  python -m pip install --user py-spy" >&2
  echo "  PYSPY_BIN=/home/\${USER}/.local/bin/py-spy ./profile_main.sh" >&2
  exit 1
fi

# In no_new_privs shells, sudo and file capabilities cannot grant ptrace access.
if [[ "$(awk '/^NoNewPrivs:/ {print $2}' /proc/self/status 2>/dev/null || echo 0)" == "1" ]]; then
  echo "This shell has NoNewPrivs=1; py-spy attach permissions cannot be elevated here." >&2
  echo "Run this script from a normal terminal session (outside this restricted shell)." >&2
  exit 1
fi

echo "Profiling main.py..."
echo "Output: $OUT"
echo "Run timeout: $RUN_TIMEOUT"
echo "Profile duration: ${PROFILE_DURATION}s"

set +e
"$PYTHON_BIN" "$ROOT/main.py" "$@" &
APP_PID=$!
set -e

cleanup_signal() {
  if kill -0 "$APP_PID" >/dev/null 2>&1; then
    kill "$APP_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup_signal INT TERM

# Enforce timeout without wrapping the python PID.
TIMEOUT_FLAG="$(mktemp "${OUT_DIR}/.timeout.${APP_PID}.XXXXXX")"
rm -f "$TIMEOUT_FLAG"
(
  sleep "$RUN_TIMEOUT"
  if kill -0 "$APP_PID" >/dev/null 2>&1; then
    echo 1 > "$TIMEOUT_FLAG"
    kill "$APP_PID" >/dev/null 2>&1 || true
  fi
) &
TIMER_PID=$!

set +e
"$PYSPY_BIN" record -o "$OUT" --duration "$PROFILE_DURATION" --pid "$APP_PID"
SPY_RC=$?
set -e

if [[ "$SPY_RC" -ne 0 ]]; then
  SCOPE="$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo unknown)"
  echo "py-spy attach failed (exit $SPY_RC)." >&2
  if kill -0 "$APP_PID" >/dev/null 2>&1; then
    echo "Target process is still alive." >&2
    echo "If this persists with a known Python PID, it is likely ptrace restriction (kernel.yama.ptrace_scope=${SCOPE})." >&2
  else
    echo "Target process already exited before py-spy could attach." >&2
    echo "Check earlier stderr from main.py/MPI for the first failure." >&2
  fi
  echo "Options:" >&2
  echo "  1) Run as root: sudo -E ./profile_main.sh [args...]" >&2
  echo "  2) Relax ptrace policy (root): sudo sysctl -w kernel.yama.ptrace_scope=0" >&2
  echo "  3) Grant py-spy ptrace capability (root): sudo setcap cap_sys_ptrace+ep \"$(command -v "$PYSPY_BIN")\"" >&2
  if kill -0 "$APP_PID" >/dev/null 2>&1; then
    kill "$APP_PID" >/dev/null 2>&1 || true
  fi
  wait "$APP_PID" >/dev/null 2>&1 || true
  exit "$SPY_RC"
fi

set +e
wait "$APP_PID"
APP_RC=$?
set -e

kill "$TIMER_PID" >/dev/null 2>&1 || true
wait "$TIMER_PID" >/dev/null 2>&1 || true

if [[ -s "$TIMEOUT_FLAG" ]]; then
  echo "Stopped after timeout (${RUN_TIMEOUT})."
  rm -f "$TIMEOUT_FLAG"
  exit 0
fi

rm -f "$TIMEOUT_FLAG"
exit "$APP_RC"
# exec "$PYSPY_BIN" top -- "$PYTHON_BIN" "$ROOT/main.py" "$@"
