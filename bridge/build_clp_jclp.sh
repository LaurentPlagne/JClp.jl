#!/usr/bin/env bash
set -euo pipefail

JCLP_ROOT="/Users/laurentplagne/Projects/JClp.jl"
CLP_ROOT="/Users/laurentplagne/Projects/Clp"
BRIDGE_DIR="${JCLP_ROOT}/bridge"
BRIDGE_LIB_DIR="${BRIDGE_DIR}/build/lib"
INSTALL_PREFIX="${CLP_ROOT}/parity/install_jclp"
COINUTILS_SRC="${CLP_ROOT}/parity/deps/CoinUtils"
COINUTILS_BUILD="${CLP_ROOT}/parity/build/coinutils_jclp"
CLP_BUILD="${CLP_ROOT}/parity/build/clp_jclp"
NPROC="$(sysctl -n hw.ncpu)"

mkdir -p "${BRIDGE_LIB_DIR}" "${COINUTILS_BUILD}" "${CLP_BUILD}" "${INSTALL_PREFIX}"

echo "[JClp] 1/3 Building libjclp_bridge.dylib..."
clang++ -O3 -std=c++17 -fPIC -dynamiclib \
  -install_name "${BRIDGE_LIB_DIR}/libjclp_bridge.dylib" \
  -I"${BRIDGE_DIR}" \
  "${BRIDGE_DIR}/jclp_bridge.cpp" \
  -o "${BRIDGE_LIB_DIR}/libjclp_bridge.dylib"

echo "[JClp] 2/3 Configuring & building CoinUtils (jclp-bridge)..."
if [[ ! -f "${COINUTILS_BUILD}/Makefile" ]]; then
  (
    cd "${COINUTILS_BUILD}"
    "${COINUTILS_SRC}/configure" \
      --prefix="${INSTALL_PREFIX}" \
      CXX=clang++ CC=clang \
      CPPFLAGS="-I${BRIDGE_DIR} -DJCLP_BRIDGE_ENABLED" \
      CXXFLAGS="-O3 -DNDEBUG -std=c++17" \
      CFLAGS="-O3 -DNDEBUG" \
      LDFLAGS="-L${BRIDGE_LIB_DIR} -ljclp_bridge" \
      --disable-dependency-tracking
  )
fi
make -C "${COINUTILS_BUILD}" -j"${NPROC}" install

echo "[JClp] 3/3 Configuring & building Clp (jclp-bridge)..."
if [[ ! -f "${CLP_BUILD}/Makefile" ]]; then
  (
    cd "${CLP_BUILD}"
    PKG_CONFIG_PATH="${INSTALL_PREFIX}/lib/pkgconfig" \
    "${CLP_ROOT}/configure" \
      --prefix="${INSTALL_PREFIX}" \
      CXX=clang++ CC=clang \
      CPPFLAGS="-I${BRIDGE_DIR} -DJCLP_BRIDGE_ENABLED" \
      CXXFLAGS="-O3 -DNDEBUG -std=c++17" \
      CFLAGS="-O3 -DNDEBUG" \
      LDFLAGS="-L${BRIDGE_LIB_DIR} -ljclp_bridge" \
      --disable-dependency-tracking
  )
fi
make -C "${CLP_BUILD}" -j"${NPROC}" install

echo "[JClp] Done! Bridged clp binary: ${INSTALL_PREFIX}/bin/clp"
otool -L "${INSTALL_PREFIX}/bin/clp"
