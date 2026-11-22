#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/bridge/build"
JULIA_BIN="$(command -v julia)"
JULIAC_BIN="${HOME}/.julia/bin/juliac"

mkdir -p "${BUILD_DIR}"

echo "[JClp] Building libjclp.dylib via JuliaC (--trim=safe)..."
JULIA_APPS_JULIA_CMD="${JULIA_BIN}" "${JULIAC_BIN}" \
  --output-lib libjclp \
  --project "${ROOT_DIR}" \
  --bundle "${BUILD_DIR}" \
  --trim=safe \
  --experimental \
  --compile-ccallable \
  --jl-option handle-signals=no \
  --export-abi "${BUILD_DIR}/jclp_abi.json" \
  "${ROOT_DIR}/src/JClp.jl"

echo "[JClp] Build complete:"
ls -lh "${BUILD_DIR}/lib/"libjclp*
