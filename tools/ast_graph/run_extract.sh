#!/usr/bin/env bash
set -euo pipefail

BASE="/Users/laurentplagne/Projects/JClp.jl/tools/ast_graph"
OUT_DIR="$BASE/out"
SHARDS_DIR="$OUT_DIR/shards"
BIN="$BASE/extract_ast"
SDK="$(xcrun --show-sdk-path)"

mkdir -p "$SHARDS_DIR"
rm -f "$SHARDS_DIR"/*.nodes.tsv "$SHARDS_DIR"/*.edges.tsv "$OUT_DIR/tu_status.tsv"

# Compile extract_ast if needed or source is newer
if [[ ! -x "$BIN" || "$BASE/extract_ast.cpp" -nt "$BIN" ]]; then
  clang++ -std=c++17 -O2 -Wall -Wextra \
    -I/opt/homebrew/opt/llvm/include \
    -L/opt/homebrew/opt/llvm/lib -Wl,-rpath,/opt/homebrew/opt/llvm/lib -lclang \
    "$BASE/extract_ast.cpp" -o "$BIN"
fi

# Generate the exact job list from the two Makefiles
JOBS_FILE="$OUT_DIR/jobs.tsv"
python3 - "$JOBS_FILE" <<'PY'
import re, sys
from pathlib import Path

def extract_cpp_list(makefile_path: Path, vars_to_expand: list[str]) -> list[str]:
    text = makefile_path.read_text()
    text = re.sub(r"\\\n", " ", text)
    var_map = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9_]+)\s*=\s*(.*)$", line)
        if m:
            var_map[m.group(1)] = m.group(2)

    def expand(val: str) -> list[str]:
        tokens = []
        for tok in val.split():
            m = re.match(r"^\$\(([A-Za-z0-9_]+)\)$", tok)
            if m and m.group(1) in var_map:
                tokens.extend(expand(var_map[m.group(1)]))
            else:
                tokens.append(tok)
        return tokens

    cpps = []
    seen = set()
    for v in vars_to_expand:
        for tok in expand(var_map.get(v, "")):
            if tok.endswith(".lo") or tok.endswith(".$(OBJEXT)"):
                cpp = tok.replace(".lo", ".cpp").replace(".$(OBJEXT)", ".cpp")
                if cpp not in seen:
                    seen.add(cpp)
                    cpps.append(cpp)
    return cpps

clp_cpps = extract_cpp_list(
    Path("/Users/laurentplagne/Projects/Clp/parity/build/clp_clean/src/Makefile"),
    ["am_libClp_la_OBJECTS", "am_clp_OBJECTS"],
)
cu_cpps = extract_cpp_list(
    Path("/Users/laurentplagne/Projects/Clp/parity/build/coinutils/src/Makefile"),
    ["am_libCoinUtils_la_OBJECTS"],
)

out_path = Path(sys.argv[1])
with out_path.open("w") as f:
    for cpp in clp_cpps:
        full = Path("/Users/laurentplagne/Projects/Clp/src") / cpp
        assert full.exists(), f"Missing Clp source: {full}"
        f.write(f"clp\t{cpp}\t{full}\n")
    for cpp in cu_cpps:
        full = Path("/Users/laurentplagne/Projects/Clp/parity/deps/CoinUtils/src") / cpp
        assert full.exists(), f"Missing CoinUtils source: {full}"
        f.write(f"coinutils\t{cpp}\t{full}\n")

print(f"Prepared {len(clp_cpps)} Clp TUs + {len(cu_cpps)} CoinUtils TUs = {len(clp_cpps) + len(cu_cpps)} total TUs.")
PY

run_one() {
  local repo="$1"
  local cpp_name="$2"
  local full_path="$3"
  local shard_stem="${repo}__${cpp_name%.cpp}"
  local out_n="$SHARDS_DIR/${shard_stem}.nodes.tsv"
  local out_e="$SHARDS_DIR/${shard_stem}.edges.tsv"

  if [[ "$repo" == "clp" ]]; then
    "$BIN" --tu "$full_path" --repo clp --out-nodes "$out_n" --out-edges "$out_e" -- \
      -x c++ -std=c++11 -O3 -DNDEBUG -DHAVE_CONFIG_H -DCLPLIB_BUILD \
      -I/Users/laurentplagne/Projects/Clp/parity/build/clp_clean/src \
      -I/Users/laurentplagne/Projects/Clp/src \
      -I/Users/laurentplagne/Projects/Clp/parity/build/clp_clean/src/OsiClp \
      -I/Users/laurentplagne/Projects/Clp/parity/install/include/coin-or \
      -nostdinc++ \
      -isystem /opt/homebrew/opt/llvm/include/c++/v1 \
      -resource-dir /opt/homebrew/opt/llvm/lib/clang/17 \
      -isysroot "$SDK"
  else
    "$BIN" --tu "$full_path" --repo coinutils --out-nodes "$out_n" --out-edges "$out_e" -- \
      -x c++ -std=c++11 -O3 -DNDEBUG -DHAVE_CONFIG_H -DCOINUTILSLIB_BUILD \
      -I/Users/laurentplagne/Projects/Clp/parity/build/coinutils/src \
      -I/Users/laurentplagne/Projects/Clp/parity/deps/CoinUtils/src \
      -nostdinc++ \
      -isystem /opt/homebrew/opt/llvm/include/c++/v1 \
      -resource-dir /opt/homebrew/opt/llvm/lib/clang/17 \
      -isysroot "$SDK"
  fi
}

export -f run_one
export BIN SHARDS_DIR SDK

xargs -P 8 -n 3 bash -c 'run_one "$1" "$2" "$3"' _ < <(tr '\t' ' ' < "$JOBS_FILE") > "$OUT_DIR/tu_status.tsv"

python3 - "$OUT_DIR/tu_status.tsv" <<'PY'
import sys
rows = [line.rstrip("\n").split("\t") for line in open(sys.argv[1]) if line.strip()]
total_tus = len(rows)
clean_tus = sum(1 for r in rows if int(r[2]) == 0 and int(r[3]) == 0)
fatal_sum = sum(int(r[2]) for r in rows)
err_sum = sum(int(r[3]) for r in rows)
warn_sum = sum(int(r[4]) for r in rows)
print(f"TUs parsed: {total_tus}, clean (0 fatal/0 error): {clean_tus}, fatal: {fatal_sum}, errors: {err_sum}, warnings: {warn_sum}")
if fatal_sum > 0 or err_sum > 0:
    for r in rows:
        if int(r[2]) > 0 or int(r[3]) > 0:
            print("  DIAG FAILURE:", r)
    sys.exit(1)
PY

python3 "$BASE/merge_ast.py"
