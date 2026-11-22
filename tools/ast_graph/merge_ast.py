#!/usr/bin/env python3
"""Merge per-TU libclang AST shards into ast_nodes.tsv and ast_edges.tsv."""

from __future__ import annotations

import glob
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


def is_header(path: str) -> bool:
    return path.endswith((".hpp", ".h", ".hxx", ".hh"))


def stem(path: str) -> str:
    return Path(path).stem


def merge_shards(shards_dir: Path, out_dir: Path) -> None:
    node_files = sorted(
        p for p in glob.glob(str(shards_dir / "*.nodes.tsv"))
        if not os.path.basename(p).startswith("test_")
    )
    edge_files = sorted(
        p for p in glob.glob(str(shards_dir / "*.edges.tsv"))
        if not os.path.basename(p).startswith("test_")
    )

    if not node_files:
        print("ERROR: No shard .nodes.tsv files found.", file=sys.stderr)
        sys.exit(1)

    # usr -> best node tuple + candidate decl_files
    best_nodes: dict[str, list[str]] = {}
    nondef_decl_files: dict[str, list[str]] = defaultdict(list)
    max_virtual: dict[str, int] = defaultdict(int)

    for nf in node_files:
        with open(nf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                cols = line.split("\t")
                if len(cols) < 14:
                    continue
                usr = cols[0]
                is_def = int(cols[10])
                is_virt = int(cols[11])
                loc = int(cols[9])
                file_path = cols[6]
                decl_file = cols[13]

                if is_virt > max_virtual[usr]:
                    max_virtual[usr] = is_virt

                if is_def == 0 and decl_file:
                    nondef_decl_files[usr].append(decl_file)

                cur = best_nodes.get(usr)
                if cur is None:
                    best_nodes[usr] = cols
                else:
                    cur_def = int(cur[10])
                    cur_loc = int(cur[9])
                    if (is_def, loc, file_path) > (cur_def, cur_loc, cur[6]):
                        best_nodes[usr] = cols

    # Finalize decl_file and is_virtual on best_nodes
    type_kinds = {"class", "struct", "enum", "typedef"}
    for usr, cols in best_nodes.items():
        cols[11] = str(max_virtual[usr])
        kind = cols[1]
        win_file = cols[6]
        is_def = int(cols[10])
        if kind in type_kinds:
            # For types, the defining header IS the declaration file
            cols[13] = win_file
        else:
            # For callables, prefer a header declaration file if defined in .cpp
            if is_def == 1 and not is_header(win_file):
                cands = [p for p in nondef_decl_files.get(usr, []) if is_header(p)]
                if cands:
                    win_stem = stem(win_file)
                    matching = [p for p in cands if stem(p) == win_stem]
                    cols[13] = matching[0] if matching else cands[0]
                else:
                    cols[13] = win_file
            else:
                cols[13] = win_file

    # Merge edges: (src_usr, dst_usr, edge_kind) -> max (src_is_def, count)
    best_edges: dict[tuple[str, str, str], tuple[int, int]] = {}
    dropped_external = 0

    for ef in edge_files:
        with open(ef, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 5:
                    continue
                src_usr, dst_usr, edge_kind, cnt_s, src_def_s = parts[:5]
                if src_usr not in best_nodes or dst_usr not in best_nodes:
                    dropped_external += 1
                    continue
                cnt = int(cnt_s)
                src_def = int(src_def_s)
                key = (src_usr, dst_usr, edge_kind)
                prev = best_edges.get(key)
                if prev is None or (src_def, cnt) > prev:
                    best_edges[key] = (src_def, cnt)

    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_out = out_dir / "ast_nodes.tsv"
    edges_out = out_dir / "ast_edges.tsv"

    with open(nodes_out, "w", encoding="utf-8") as f:
        f.write(
            "usr\tkind\trepo\tname\tqualname\towner_usr\tfile\tline_start\t"
            "line_end\tloc\tis_def\tis_virtual\tsignature\tdecl_file\n"
        )
        for usr in sorted(best_nodes.keys()):
            f.write("\t".join(best_nodes[usr]) + "\n")

    with open(edges_out, "w", encoding="utf-8") as f:
        f.write("src_usr\tdst_usr\tedge_kind\tcount\n")
        for (src_usr, dst_usr, edge_kind), (_, cnt) in sorted(best_edges.items()):
            f.write(f"{src_usr}\t{dst_usr}\t{edge_kind}\t{cnt}\n")

    # Summary stats
    by_repo_kind: Counter[tuple[str, str]] = Counter()
    by_repo: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    for cols in best_nodes.values():
        kind, repo = cols[1], cols[2]
        by_repo_kind[(repo, kind)] += 1
        by_repo[repo] += 1
        by_kind[kind] += 1

    edge_counts: Counter[str] = Counter()
    edge_weight_sums: Counter[str] = Counter()
    for (_, _, ek), (_, cnt) in best_edges.items():
        edge_counts[ek] += 1
        edge_weight_sums[ek] += cnt

    print(f"Merged {len(node_files)} node shards and {len(edge_files)} edge shards.")
    print(f"Total unique nodes: {len(best_nodes)} (clp={by_repo['clp']}, coinutils={by_repo['coinutils']})")
    print("Nodes by repo and kind:")
    for repo in ("clp", "coinutils"):
        for kind in ("class", "struct", "enum", "typedef", "method", "ctor", "dtor", "function"):
            print(f"  {repo:10s} {kind:10s}: {by_repo_kind[(repo, kind)]}")
    print(f"Total unique edges: {len(best_edges)} (dropped non-node edges: {dropped_external})")
    print("Edges by edge_kind (unique_edges / total_weighted_count):")
    for ek in sorted(edge_counts.keys()):
        print(f"  {ek:20s}: {edge_counts[ek]:6d}  (sum_count={edge_weight_sums[ek]})")

    # Sanity checks on key classes
    key_classes = [
        "ClpSimplexDual",
        "ClpFactorization",
        "CoinFactorization",
        "CoinIndexedVector",
        "ClpDualRowSteepest",
        "ClpPackedMatrix",
    ]
    qual_to_usrs: dict[str, list[str]] = defaultdict(list)
    owner_to_methods: dict[str, list[str]] = defaultdict(list)
    for usr, cols in best_nodes.items():
        qual_to_usrs[cols[4]].append(usr)
        if cols[5]:
            owner_to_methods[cols[5]].append(usr)

    out_edges_by_src: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    in_edges_by_dst: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for (s, d, ek), (_, cnt) in best_edges.items():
        out_edges_by_src[s].append((d, ek, cnt))
        in_edges_by_dst[d].append((s, ek, cnt))

    print("\nSanity check on key classes:")
    for cls in key_classes:
        usrs = [u for u in qual_to_usrs.get(cls, []) if best_nodes[u][1] in ("class", "struct")]
        if not usrs:
            print(f"  MISSING CLASS: {cls}")
            continue
        u = usrs[0]
        c = best_nodes[u]
        methods = owner_to_methods.get(u, [])
        calls_out = sum(
            1 for m in methods for (_, ek, _) in out_edges_by_src.get(m, []) if ek == "calls"
        )
        overrides_out = sum(
            1 for m in methods for (_, ek, _) in out_edges_by_src.get(m, []) if ek == "overrides"
        )
        inherits_out = [
            best_nodes[d][4] for (d, ek, _) in out_edges_by_src.get(u, []) if ek == "inherits"
        ]
        print(
            f"  {cls:20s} repo={c[2]} file={c[6]}:{c[7]}-{c[8]} "
            f"methods={len(methods)} calls_out={calls_out} overrides={overrides_out} inherits={inherits_out}"
        )


if __name__ == "__main__":
    base = Path("/Users/laurentplagne/Projects/JClp.jl/tools/ast_graph")
    merge_shards(base / "out" / "shards", base / "out")
