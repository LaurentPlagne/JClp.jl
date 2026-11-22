#!/usr/bin/env julia
#=
Analyze the extracted C++ AST graph (ast_nodes.tsv, ast_edges.tsv) for the
V1 LP + CoinUtils scope of Clp:
  1. Load & index all nodes and edges + ownership mapping.
  2. Check ClpFactorization -> Coin{Dense,Osl,Simp,Other}Factorization edges.
  3. Compute V1 reachability (full vs strict call-only) from V1 entry points.
  4. Build Graph 1 (Type/Module dependency graph): Tarjan SCC + bottom-up
     topological layers + cross-class cycle edge analysis.
  5. Build Graph 2 (Fine-grained MF call graph): Tarjan SCC + bottom-up
     topological layers + layer distribution.
  6. Compute per-cluster internal + external dependency profiles.
  7. Write v1_inventory_types.tsv, v1_inventory_mf.tsv, and v1_scc_report.txt.
=#

using Printf

const OUT_DIR = joinpath(@__DIR__, "out")
const NODES_TSV = joinpath(OUT_DIR, "ast_nodes.tsv")
const EDGES_TSV = joinpath(OUT_DIR, "ast_edges.tsv")

struct AstNode
    usr::String
    kind::String
    repo::String
    name::String
    qualname::String
    owner_usr::String
    file::String
    line_start::Int
    line_end::Int
    loc::Int
    is_def::Bool
    is_virtual::Bool
    signature::String
    decl_file::String
end

struct AstEdge
    src::String
    dst::String
    kind::String
    count::Int
end

function load_graph(nodes_path::String, edges_path::String)
    nodes = Dict{String, AstNode}()
    open(nodes_path, "r") do io
        _header = readline(io)
        for line in eachline(io)
            isempty(line) && continue
            c = split(line, '\t'; keepempty=true)
            node = AstNode(
                String(c[1]), String(c[2]), String(c[3]), String(c[4]),
                String(c[5]), String(c[6]), String(c[7]),
                parse(Int, c[8]), parse(Int, c[9]), parse(Int, c[10]),
                c[11] == "1", c[12] == "1", String(c[13]), String(c[14])
            )
            nodes[node.usr] = node
        end
    end

    edges = AstEdge[]
    open(edges_path, "r") do io
        _header = readline(io)
        for line in eachline(io)
            isempty(line) && continue
            c = split(line, '\t'; keepempty=true)
            push!(edges, AstEdge(String(c[1]), String(c[2]), String(c[3]), parse(Int, c[4])))
        end
    end
    return nodes, edges
end

function build_module_mapper(nodes::Dict{String, AstNode})
    struct_by_qn = Dict{String, String}()
    for (u, n) in nodes
        if n.kind in ("class", "struct") && isempty(n.owner_usr)
            struct_by_qn[n.qualname] = u
        end
    end

    function top_owner(usr::String)
        cur = usr
        while haskey(nodes, cur) && !isempty(nodes[cur].owner_usr)
            cur = nodes[cur].owner_usr
        end
        return cur
    end

    function module_id(usr::String)::Union{String, Nothing}
        n = get(nodes, usr, nothing)
        n === nothing && return nothing
        if n.kind in ("method", "ctor", "dtor")
            isempty(n.owner_usr) && return nothing
            top = top_owner(n.owner_usr)
            return "type::" * nodes[top].qualname
        elseif n.kind == "function"
            bn = splitext(basename(n.file))[1]
            return "file_free::" * bn
        elseif n.kind in ("class", "struct")
            top = top_owner(usr)
            return "type::" * nodes[top].qualname
        elseif n.kind in ("enum", "typedef")
            if !isempty(n.owner_usr)
                top = top_owner(n.owner_usr)
                return "type::" * nodes[top].qualname
            elseif n.kind == "typedef" && haskey(struct_by_qn, n.qualname)
                return "type::" * n.qualname
            else
                return n.kind * "::" * n.qualname
            end
        end
        return nothing
    end

    return module_id, struct_by_qn
end

const EXCLUDE_PREFIXES = [
    # Barrier / Interior / Cholesky / PDCO / LSQR
    "ClpInterior", "ClpPredictorCorrector", "ClpCholesky", "ClpPdco", "ClpLsqr",
    # Quadratic / Nonlinear
    "ClpQuadraticObjective", "ClpConstraintQuadratic", "ClpSimplexNonlinear", "ClpPrimalQuadratic",
    # Non-V1 Matrices / Network
    "ClpGub", "ClpNetworkMatrix", "ClpNetworkBasis", "ClpDynamic", "ClpPlusMinusOneMatrix", "ClpDummyMatrix",
    # Drivers / CLI / Racing / C-API / Params
    "ClpRacingSolver", "Clp_C_Interface", "ClpSolver", "ClpMain", "MyEventHandler", "MyMessageHandler",
    "CbcOrClpParam", "ClpParam", "ClpNode", "unitTest",
    # Non-V1 CoinUtils builders / IO / extras
    "CoinLpIO", "CoinModel", "CoinStructuredModel", "CoinSnapshot", "CoinParam", "CoinSearchTree",
    "CoinOslFactorization", "CoinSimpFactorization"
]

function is_excluded_node(usr::String, nodes::Dict{String, AstNode})::Bool
    n = nodes[usr]
    qn = n.qualname
    owner_qn = isempty(n.owner_usr) ? "" : nodes[n.owner_usr].qualname
    fbase = basename(n.file)
    dbase = basename(n.decl_file)
    for p in EXCLUDE_PREFIXES
        startswith(qn, p) && return true
        startswith(owner_qn, p) && return true
        startswith(fbase, p) && return true
        startswith(dbase, p) && return true
    end
    return false
end

function inspect_factorization_connections(nodes::Dict{String, AstNode}, edges::Vector{AstEdge})
    targets = ["CoinOslFactorization", "CoinSimpFactorization", "CoinDenseFactorization", "CoinOtherFactorization"]
    report = Dict{String, Vector{String}}(t => String[] for t in targets)
    for e in edges
        sn = get(nodes, e.src, nothing)
        dn = get(nodes, e.dst, nothing)
        (sn === nothing || dn === nothing) && continue
        src_is_clpfact = startswith(sn.qualname, "ClpFactorization") ||
                         (!isempty(sn.owner_usr) && startswith(nodes[sn.owner_usr].qualname, "ClpFactorization"))
        src_is_clpfact || continue
        for t in targets
            if startswith(dn.qualname, t) || (!isempty(dn.owner_usr) && startswith(nodes[dn.owner_usr].qualname, t))
                push!(report[t], "$(sn.qualname) --[$(e.kind):$(e.count)]--> $(dn.qualname)")
            end
        end
    end
    for t in targets
        sort!(unique!(report[t]))
    end
    return report
end

function run_reachability(
    nodes::Dict{String, AstNode},
    edges::Vector{AstEdge},
    seeds::Vector{String};
    call_only::Bool=false
)
    full_kinds = Set(["calls", "inherits", "field_type", "param_or_ret_type", "body_type_ref", "member_field_access"])
    adj = Dict{String, Vector{String}}()
    rev_ov = Dict{String, Vector{String}}()
    for e in edges
        if (call_only && e.kind == "calls") || (!call_only && e.kind in full_kinds)
            push!(get!(adj, e.src, String[]), e.dst)
        end
        if e.kind == "overrides"
            # e.src (derived) overrides e.dst (base) -> reverse edge e.dst -> e.src
            push!(get!(rev_ov, e.dst, String[]), e.src)
        end
    end

    vis = Set{String}()
    q = String[]
    for s in seeds
        if !is_excluded_node(s, nodes) && !(s in vis)
            push!(vis, s)
            push!(q, s)
        end
    end

    while !isempty(q)
        u = popfirst!(q)
        n = nodes[u]
        if !call_only && !isempty(n.owner_usr) && !is_excluded_node(n.owner_usr, nodes) && !(n.owner_usr in vis)
            push!(vis, n.owner_usr)
            push!(q, n.owner_usr)
        end
        for v in get(adj, u, String[])
            if !is_excluded_node(v, nodes) && !(v in vis)
                push!(vis, v)
                push!(q, v)
            end
        end
        for v in get(rev_ov, u, String[])
            if !is_excluded_node(v, nodes) && !(v in vis)
                push!(vis, v)
                push!(q, v)
            end
        end
    end
    return vis
end

function tarjan_scc(vlist::Vector{String}, adj::Dict{String, Set{String}})
    idx = 0
    indices = Dict{String, Int}()
    lowlink = Dict{String, Int}()
    onstack = Set{String}()
    st = String[]
    sccs = Vector{Vector{String}}()

    function strongconnect(v::String)
        idx += 1
        indices[v] = idx
        lowlink[v] = idx
        push!(st, v)
        push!(onstack, v)
        for w in sort!(collect(get(adj, v, Set{String}())))
            if !haskey(indices, w)
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elseif w in onstack
                lowlink[v] = min(lowlink[v], indices[w])
            end
        end
        if lowlink[v] == indices[v]
            comp = String[]
            while true
                w = pop!(st)
                delete!(onstack, w)
                push!(comp, w)
                w == v && break
            end
            push!(sccs, sort!(comp))
        end
    end

    for v in sort(vlist)
        haskey(indices, v) || strongconnect(v)
    end
    return sccs
end

function compute_scc_dag_layers(sccs::Vector{Vector{String}}, adj::Dict{String, Set{String}})
    node_to_scc = Dict{String, Int}()
    for (i, comp) in enumerate(sccs)
        for v in comp
            node_to_scc[v] = i
        end
    end

    n_scc = length(sccs)
    dag_adj = [Set{Int}() for _ in 1:n_scc]
    for (i, comp) in enumerate(sccs)
        for u in comp
            for w in get(adj, u, Set{String}())
                j = get(node_to_scc, w, 0)
                if j != 0 && j != i
                    push!(dag_adj[i], j)
                end
            end
        end
    end

    layers = fill(-1, n_scc)
    function compute_layer(i::Int)::Int
        layers[i] != -1 && return layers[i]
        if isempty(dag_adj[i])
            layers[i] = 0
        else
            layers[i] = 1 + maximum(compute_layer(j) for j in dag_adj[i])
        end
        return layers[i]
    end
    for i in 1:n_scc
        compute_layer(i)
    end

    return node_to_scc, dag_adj, layers
end

function assign_cluster(mid::String)::String
    startswith(mid, "file_free::") && begin
        bn = mid[length("file_free::")+1:end]
        if bn in ("CoinHelperFunctions", "CoinSort", "CoinError", "CoinFinite", "CoinTime", "CoinFileIO")
            return "CoinUtils helpers/sort/error/message"
        elseif bn == "CoinPackedMatrix"
            return "CoinPackedMatrix"
        elseif bn == "CoinMpsIO"
            return "CoinMpsIO"
        elseif startswith(bn, "CoinFactorization")
            return "CoinFactorization"
        elseif startswith(bn, "CoinPresolve") || bn == "CoinPrePostsolveMatrix"
            return "CoinPresolve*"
        elseif bn == "ClpHelperFunctions"
            return "ClpHelperFunctions"
        elseif bn == "ClpPackedMatrix"
            return "ClpPackedMatrix (+ ClpMatrixBase)"
        elseif bn == "ClpNonLinearCost"
            return "ClpNonLinearCost"
        elseif bn == "ClpPrimalColumnSteepest"
            return "ClpPrimalColumnSteepest/Dantzig"
        elseif bn == "ClpPresolve"
            return "ClpPresolve"
        elseif bn in ("Idiot", "IdiSolve")
            return "Idiot"
        elseif bn in ("ClpModel", "ClpModelParameters")
            return "ClpModel"
        elseif bn == "ClpSimplex"
            return "ClpSimplex"
        elseif bn == "ClpSolve"
            return "ClpSolve"
        elseif bn == "ClpOutput"
            return "ClpOutput/Events"
        elseif bn == "CoinTable"
            return "CoinTable"
        else
            return "Other(" * bn * ")"
        end
    end

    name = replace(mid, r"^(type|enum|typedef)::" => "")
    if name in ("CoinError", "CoinAbsFltEq", "CoinRelFltEq", "CoinPair", "CoinFirstLess_2",
                "CoinMessage", "CoinMessages", "CoinMessageHandler", "CoinOneMessage",
                "Coin_message", "CoinMessageMarker", "CoinSighandler_t", "CoinThreadRandom",
                "CoinFileIOBase", "CoinFileInput", "CoinGetslessFileInput", "CoinPlainFileInput",
                "CoinGzipFileInput", "CoinBzip2FileInput", "CoinFileOutput", "CoinPlainFileOutput",
                "CoinGzipFileOutput", "CoinBzip2FileOutput", "CoinArrayWithLength",
                "CoinDoubleArrayWithLength", "CoinFactorizationDoubleArrayWithLength",
                "CoinIntArrayWithLength", "CoinUnsignedIntArrayWithLength",
                "CoinBigIndex", "CoinByteArray", "CoinFactorizationDouble", "CoinFactorizationDouble2",
                "COIN_Message", "CoinTable")
        return "CoinUtils helpers/sort/error/message"
    elseif name == "CoinIndexedVector"
        return "CoinIndexedVector"
    elseif name in ("CoinPackedVector", "CoinPackedVectorBase", "CoinShallowPackedVector")
        return "CoinPackedVector/CoinShallowPackedVector"
    elseif name == "CoinPackedMatrix"
        return "CoinPackedMatrix"
    elseif name in ("CoinDenseFactorization", "CoinOtherFactorization")
        return "CoinDenseFactorization"
    elseif name in ("CoinFactorization", "CoinCheckZero", "cipfint", "ipfint")
        return "CoinFactorization"
    elseif startswith(name, "CoinPresolve") || name in ("CoinPrePostsolveMatrix", "CoinPostsolveMatrix", "presolvehlink", "dropped_zero") || endswith(name, "_action")
        return "CoinPresolve*"
    elseif name in ("CoinMpsIO", "CoinMpsCardReader", "COINMpsType", "COINSectionType", "COINColumnIndex", "COINRowIndex")
        return "CoinMpsIO"
    elseif name in ("ClpPackedMatrix", "ClpPackedMatrix2", "ClpPackedMatrix3", "ClpMatrixBase", "blockStruct")
        return "ClpPackedMatrix (+ ClpMatrixBase)"
    elseif name == "ClpFactorization"
        return "ClpFactorization"
    elseif name == "ClpNonLinearCost"
        return "ClpNonLinearCost"
    elseif name in ("ClpDualRowSteepest", "ClpDualRowDantzig", "ClpDualRowPivot", "ClpPEDualRowSteepest", "ClpPEDualRowDantzig")
        return "ClpDualRowSteepest/Dantzig"
    elseif name in ("ClpPrimalColumnSteepest", "ClpPrimalColumnDantzig", "ClpPrimalColumnPivot", "ClpPEPrimalColumnSteepest", "ClpPEPrimalColumnDantzig")
        return "ClpPrimalColumnSteepest/Dantzig"
    elseif name == "ClpPresolve"
        return "ClpPresolve"
    elseif name in ("Idiot", "IdiotResult")
        return "Idiot"
    elseif name in ("ClpModel", "ClpDataSave", "ClpTrustedData", "ClpObjective", "ClpLinearObjective", "ClpMessage", "CLP_Message", "Clp_message", "ClpDblParam", "ClpStrParam")
        return "ClpModel"
    elseif name in ("ClpSimplex", "ClpSimplexOther", "Clp_scalars", "clpTempInfo", "ClpPESimplex")
        return "ClpSimplex"
    elseif name == "ClpSimplexDual"
        return "ClpSimplexDual"
    elseif name == "ClpSimplexPrimal"
        return "ClpSimplexPrimal"
    elseif name in ("ClpSolve", "ClpSimplexProgress")
        return "ClpSolve"
    elseif name in ("ClpEventHandler", "ClpDisasterHandler", "ClpLpEventHandler", "ClpLpMsgHandler", "ClpProgressEventHandler", "ClpOutput", "ClpLpPhaseState")
        return "ClpOutput/Events"
    else
        return "Other(" * name * ")"
    end
end

function main()
    nodes, edges = load_graph(NODES_TSV, EDGES_TSV)
    module_id, _struct_by_qn = build_module_mapper(nodes)

    fact_conn = inspect_factorization_connections(nodes, edges)

    entry_qualnames = Set([
        "ClpModel::readMps",
        "ClpSimplex::initialSolve",
        "ClpSimplex::dual",
        "ClpSimplex::primal",
        "ClpSimplexDual::dual",
        "ClpSimplexPrimal::primal",
        "ClpPresolve::presolvedModel",
        "ClpPresolve::postsolve",
        "Idiot::solve",
    ])
    seeds = sort([u for (u, n) in nodes if n.qualname in entry_qualnames && !is_excluded_node(u, nodes)])

    vis_full = run_reachability(nodes, edges, seeds; call_only=false)
    vis_call = run_reachability(nodes, edges, seeds; call_only=true)

    # Precompute reverse overrides within vis_full
    rev_ov_map = Dict{String, Vector{String}}()
    for e in edges
        if e.kind == "overrides" && e.src in vis_full && e.dst in vis_full
            push!(get!(rev_ov_map, e.dst, String[]), e.src)
        end
    end
    function all_v1_overrides(m::String)::Vector{String}
        res = String[]
        seen = Set{String}([m])
        st = String[m]
        while !isempty(st)
            x = pop!(st)
            for y in get(rev_ov_map, x, String[])
                if !(y in seen)
                    push!(seen, y)
                    push!(res, y)
                    push!(st, y)
                end
            end
        end
        return res
    end

    # =========================================================================
    # GRAPH 1: Type / Module Dependency Graph
    # =========================================================================
    g1_nodes = Set{String}()
    for u in vis_full
        mid = module_id(u)
        mid === nothing || push!(g1_nodes, mid)
    end
    g1_vlist = sort!(collect(g1_nodes))
    g1_adj = Dict{String, Set{String}}(m => Set{String}() for m in g1_vlist)
    g1_adj_static = Dict{String, Set{String}}(m => Set{String}() for m in g1_vlist)

    # Track exact cross-module edge breakdowns for cycle analysis
    # edge_detail[(ms, md, kind)] => total count
    edge_detail = Dict{Tuple{String, String, String}, Int}()
    # edge_examples[(ms, md)] => Set of "src_qn --[kind]--> dst_qn"
    edge_examples = Dict{Tuple{String, String}, Dict{String, Int}}()

    g1_direct_kinds = Set(["calls", "inherits", "field_type", "member_field_access"])
    for e in edges
        (e.src in vis_full && e.dst in vis_full) || continue
        ms = module_id(e.src)
        md = module_id(e.dst)
        (ms === nothing || md === nothing) && continue
        sn = nodes[e.src]
        dn = nodes[e.dst]

        if (e.kind in g1_direct_kinds || e.kind == "overrides") && ms != md
            push!(g1_adj[ms], md)
            push!(g1_adj_static[ms], md)
            k = (ms, md, e.kind)
            edge_detail[k] = get(edge_detail, k, 0) + e.count
            exmap = get!(edge_examples, (ms, md), Dict{String, Int}())
            label = "$(sn.qualname) --[$(e.kind)]--> $(dn.qualname)"
            exmap[label] = get(exmap, label, 0) + e.count
        end

        if e.kind == "calls"
            for ov in all_v1_overrides(e.dst)
                mo = module_id(ov)
                if mo !== nothing && mo != ms
                    push!(g1_adj[ms], mo)
                    k = (ms, mo, "virtual_dispatch")
                    edge_detail[k] = get(edge_detail, k, 0) + e.count
                    exmap = get!(edge_examples, (ms, mo), Dict{String, Int}())
                    on = nodes[ov]
                    label = "$(sn.qualname) --[vcall via $(dn.qualname)]--> $(on.qualname)"
                    exmap[label] = get(exmap, label, 0) + e.count
                end
            end
        end
    end

    sccs1 = tarjan_scc(g1_vlist, g1_adj)
    node_to_scc1, dag_adj1, layers1 = compute_scc_dag_layers(sccs1, g1_adj)

    sccs1_static = tarjan_scc(g1_vlist, g1_adj_static)
    _, _, layers1_static = compute_scc_dag_layers(sccs1_static, g1_adj_static)

    # Module stats
    mod_mf_def = Dict{String, Int}(m => 0 for m in g1_vlist)
    mod_mf_all = Dict{String, Int}(m => 0 for m in g1_vlist)
    mod_loc = Dict{String, Int}(m => 0 for m in g1_vlist)
    mod_rep = Dict{String, AstNode}()
    for u in vis_full
        n = nodes[u]
        mid = module_id(u)
        mid === nothing && continue
        if n.kind in ("class", "struct", "enum", "typedef") && isempty(n.owner_usr)
            if !haskey(mod_rep, mid) || n.kind in ("class", "struct")
                mod_rep[mid] = n
            end
        elseif !haskey(mod_rep, mid)
            mod_rep[mid] = n
        end
        if n.kind in ("method", "ctor", "dtor", "function")
            mod_mf_all[mid] += 1
            if n.is_def
                mod_mf_def[mid] += 1
                mod_loc[mid] += n.loc
            end
        end
    end

    # =========================================================================
    # GRAPH 2: Fine-Grained MF Call Graph
    # =========================================================================
    mf_nodes = sort!([u for u in vis_full if nodes[u].kind in ("method", "ctor", "dtor", "function") && nodes[u].is_def])
    mf_set = Set(mf_nodes)
    g2_adj = Dict{String, Set{String}}(u => Set{String}() for u in mf_nodes)
    for e in edges
        if e.kind == "calls" && e.src in mf_set && e.dst in vis_full
            if e.dst in mf_set && e.dst != e.src
                push!(g2_adj[e.src], e.dst)
            end
            for ov in all_v1_overrides(e.dst)
                if ov in mf_set && ov != e.src
                    push!(g2_adj[e.src], ov)
                end
            end
        end
    end

    sccs2 = tarjan_scc(mf_nodes, g2_adj)
    node_to_scc2, dag_adj2, layers2 = compute_scc_dag_layers(sccs2, g2_adj)

    g2_indeg = Dict{String, Int}(u => 0 for u in mf_nodes)
    for (u, nbrs) in g2_adj
        for v in nbrs
            g2_indeg[v] = get(g2_indeg, v, 0) + 1
        end
    end

    # =========================================================================
    # Write v1_inventory_types.tsv
    # =========================================================================
    g1_indeg = Dict{String, Int}(m => 0 for m in g1_vlist)
    for (m, nbrs) in g1_adj
        for w in nbrs
            g1_indeg[w] = get(g1_indeg, w, 0) + 1
        end
    end

    open(joinpath(OUT_DIR, "v1_inventory_types.tsv"), "w") do io
        println(io, "module_id\tkind\trepo\tqualname\tfile\tcluster\tscc_id\tscc_size\tlayer\tmf_def_count\tmf_all_count\ttotal_loc\tin_degree\tout_degree\tdeps")
        order = sort(g1_vlist; by=m -> (layers1[node_to_scc1[m]], node_to_scc1[m], m))
        for m in order
            rep = mod_rep[m]
            sid = node_to_scc1[m]
            ssz = length(sccs1[sid])
            lay = layers1[sid]
            kind_lbl = startswith(m, "file_free::") ? "file_free" : rep.kind
            file_lbl = startswith(m, "file_free::") ? rep.file : (isempty(rep.decl_file) ? rep.file : rep.decl_file)
            deps = join(sort!(collect(g1_adj[m])), ",")
            println(io, join([
                m, kind_lbl, rep.repo, rep.qualname, file_lbl, assign_cluster(m),
                string(sid), string(ssz), string(lay),
                string(mod_mf_def[m]), string(mod_mf_all[m]), string(mod_loc[m]),
                string(g1_indeg[m]), string(length(g1_adj[m])), deps
            ], '\t'))
        end
    end

    # =========================================================================
    # Write v1_inventory_mf.tsv
    # =========================================================================
    open(joinpath(OUT_DIR, "v1_inventory_mf.tsv"), "w") do io
        println(io, "usr\tkind\trepo\tqualname\tmodule_id\tcluster\tfile\tline_start\tline_end\tloc\tis_virtual\tscc_id\tscc_size\tlayer\tin_calls\tout_calls\treach_call_only")
        order = sort(mf_nodes; by=u -> (layers2[node_to_scc2[u]], node_to_scc2[u], nodes[u].qualname, u))
        for u in order
            n = nodes[u]
            mid = something(module_id(u), "")
            sid = node_to_scc2[u]
            ssz = length(sccs2[sid])
            lay = layers2[sid]
            println(io, join([
                u, n.kind, n.repo, n.qualname, mid, assign_cluster(mid),
                n.file, string(n.line_start), string(n.line_end), string(n.loc),
                n.is_virtual ? "1" : "0",
                string(sid), string(ssz), string(lay),
                string(g2_indeg[u]), string(length(g2_adj[u])),
                (u in vis_call) ? "1" : "0"
            ], '\t'))
        end
    end

    # =========================================================================
    # Write v1_scc_report.txt
    # =========================================================================
    open(joinpath(OUT_DIR, "v1_scc_report.txt"), "w") do io
        println(io, "================================================================================")
        println(io, "JClp.jl V1 LP + CoinUtils AST Graph Analysis Report")
        println(io, "================================================================================\n")

        println(io, "1. RAW AST INVENTORY")
        println(io, "   Total nodes loaded: ", length(nodes))
        println(io, "   Total edges loaded: ", length(edges))
        println(io, "   Entry seeds matched: ", length(seeds))
        for s in seeds
            n = nodes[s]
            println(io, "     - $(n.qualname) [$(n.file):$(n.line_start)]")
        end
        println(io)

        println(io, "2. CLPFACTORIZATION CONNECTIONS TO COIN*FACTORIZATION")
        for t in ["CoinFactorization", "CoinOtherFactorization", "CoinDenseFactorization", "CoinOslFactorization", "CoinSimpFactorization"]
            if t == "CoinFactorization"
                cnt = count(e -> (get(nodes, e.src, nothing) !== nothing && get(nodes, e.dst, nothing) !== nothing) &&
                                 (startswith(nodes[e.src].qualname, "ClpFactorization") || (!isempty(nodes[e.src].owner_usr) && startswith(nodes[nodes[e.src].owner_usr].qualname, "ClpFactorization"))) &&
                                 (startswith(nodes[e.dst].qualname, "CoinFactorization") || (!isempty(nodes[e.dst].owner_usr) && startswith(nodes[nodes[e.dst].owner_usr].qualname, "CoinFactorization"))), edges)
                println(io, "   ClpFactorization -> CoinFactorization: $cnt AST edge records (default active sparse LU backend)")
            else
                conns = fact_conn[t]
                println(io, "   ClpFactorization -> $t ($(length(conns)) unique AST edge pairs):")
                for c in conns[1:min(15, length(conns))]
                    println(io, "     * $c")
                end
                length(conns) > 15 && println(io, "     * ... ($(length(conns) - 15) more)")
            end
        end
        println(io, "   VERDICT: ClpFactorization holds field coinFactorizationB_ :: CoinOtherFactorization* and")
        println(io, "   instantiates CoinDenseFactorization / CoinSimpFactorization / CoinOslFactorization in")
        println(io, "   copy-ctor/operator=/goDenseOrSmall/forceOtherFactorization. However, default thresholds")
        println(io, "   are goOslThreshold_ = -1, goSmallThreshold_ = -1, goDenseThreshold_ = -1 (default ctor")
        println(io, "   uses CoinFactorization only). CoinOslFactorization & CoinSimpFactorization are optional")
        println(io, "   and safely EXCLUDED for V1; CoinDenseFactorization is KEPT as the small-dense override.\n")

        println(io, "3. REACHABILITY COMPARISON: FULL vs STRICT CALL-ONLY")
        call_mf_def = count(u -> nodes[u].kind in ("method", "ctor", "dtor", "function") && nodes[u].is_def, vis_call)
        call_mf_all = count(u -> nodes[u].kind in ("method", "ctor", "dtor", "function"), vis_call)
        call_loc = sum(nodes[u].loc for u in vis_call if nodes[u].kind in ("method", "ctor", "dtor", "function") && nodes[u].is_def; init=0)
        call_mods = Set(m for u in vis_call for m in (module_id(u),) if m !== nothing)
        full_mf_def = length(mf_nodes)
        full_mf_all = count(u -> nodes[u].kind in ("method", "ctor", "dtor", "function"), vis_full)
        full_loc = sum(nodes[u].loc for u in mf_nodes; init=0)
        @printf(io, "   %-28s | %10s | %10s | %10s | %10s | %10s\n",
                "Mode", "TotalNodes", "MF_Defined", "MF_All", "Modules", "Def_LOC")
        println(io, "   " * "-"^90)
        @printf(io, "   %-28s | %10d | %10d | %10d | %10d | %10d\n",
                "Full (all edge kinds + revOv)", length(vis_full), full_mf_def, full_mf_all, length(g1_vlist), full_loc)
        @printf(io, "   %-28s | %10d | %10d | %10d | %10d | %10d\n",
                "Strict call-only (calls+revOv)", length(vis_call), call_mf_def, call_mf_all, length(call_mods), call_loc)
        extra_mods = sort!(collect(setdiff(Set(g1_vlist), call_mods)))
        println(io, "   Extra modules in Full vs Call-only ($(length(extra_mods))): ", join(extra_mods, ", "))
        println(io)

        println(io, "4. GRAPH 1: V1 TYPE/MODULE DEPENDENCY GRAPH (WITH VIRTUAL DISPATCH)")
        multi1 = [i for i in 1:length(sccs1) if length(sccs1[i]) > 1]
        multi1_static = [i for i in 1:length(sccs1_static) if length(sccs1_static[i]) > 1]
        println(io, "   Nodes: $(length(g1_vlist)) (types=$(count(x->startswith(x,"type::"), g1_vlist)), file_free=$(count(x->startswith(x,"file_free::"), g1_vlist)), enum/typedef=$(count(x->!startswith(x,"type::")&&!startswith(x,"file_free::"), g1_vlist)))")
        println(io, "   Directed edges: $(sum(length, values(g1_adj)))")
        println(io, "   SCCs: $(length(sccs1)) (singletons=$(length(sccs1)-length(multi1)), multi-node=$(length(multi1)), max_layer=$(maximum(layers1)))")
        println(io, "   [Comparison without virtual dispatch edges in Graph 1: $(length(sccs1_static)) SCCs, $(length(multi1_static)) multi-node, max_layer=$(maximum(layers1_static))]")
        println(io)

        println(io, "   --- All Graph 1 SCCs by Bottom-Up Layer (0 = leaves) ---")
        scc_order = sort(1:length(sccs1); by=i -> (layers1[i], -length(sccs1[i]), sccs1[i][1]))
        for i in scc_order
            comp = sccs1[i]
            mf_cnt = sum(mod_mf_def[m] for m in comp)
            loc_cnt = sum(mod_loc[m] for m in comp)
            dep_layers = sort!(unique([layers1[j] for j in dag_adj1[i]]))
            dep_sccs = sort!(collect(dag_adj1[i]))
            if length(comp) == 1
                m = comp[1]
                deps_str = isempty(dep_sccs) ? "-" : join(["L$(layers1[j]):$(sccs1[j][1])" * (length(sccs1[j])>1 ? "(SCC$(j))" : "") for j in dep_sccs], ", ")
                @printf(io, "   [L%02d | SCC#%03d | size=1] %-42s | mf=%4d | loc=%6d | deps: %s\n",
                        layers1[i], i, m, mf_cnt, loc_cnt, deps_str)
            else
                println(io, "   [L$(lpad(layers1[i],2,'0')) | SCC#$(lpad(i,3,'0')) | size=$(length(comp))] MULTI-NODE SCC | mf=$(mf_cnt) | loc=$(loc_cnt)")
                for m in comp
                    @printf(io, "       - %-46s | mf=%4d | loc=%6d\n", m, mod_mf_def[m], mod_loc[m])
                end
                if !isempty(dep_sccs)
                    deps_str = join(["L$(layers1[j]):SCC#$(j)($(sccs1[j][1]))" for j in dep_sccs], ", ")
                    println(io, "       -> External SCC deps: ", deps_str)
                end
            end
        end
        println(io)

        println(io, "   --- Cross-Class Cycle Analysis for Multi-Node SCCs in Graph 1 ---")
        for i in sort(multi1; by=x -> (layers1[x], length(sccs1[x])))
            comp = sccs1[i]
            comp_set = Set(comp)
            println(io, "\n   === SCC #$(i) (Layer $(layers1[i]), size=$(length(comp))) ===")
            # Summarize edge kinds inside SCC
            kind_totals = Dict{String, Int}()
            pair_rows = Tuple{Int, Int, Int, Int, Int, Int, String, String}[]
            for a in comp, b in comp
                a == b && continue
                c_calls = get(edge_detail, (a, b, "calls"), 0)
                c_mfa   = get(edge_detail, (a, b, "member_field_access"), 0)
                c_ft    = get(edge_detail, (a, b, "field_type"), 0)
                c_inh   = get(edge_detail, (a, b, "inherits"), 0)
                c_ov    = get(edge_detail, (a, b, "overrides"), 0)
                c_vcall = get(edge_detail, (a, b, "virtual_dispatch"), 0)
                tot = c_calls + c_mfa + c_ft + c_inh + c_ov + c_vcall
                tot == 0 && continue
                kind_totals["calls"] = get(kind_totals, "calls", 0) + c_calls
                kind_totals["member_field_access"] = get(kind_totals, "member_field_access", 0) + c_mfa
                kind_totals["field_type"] = get(kind_totals, "field_type", 0) + c_ft
                kind_totals["inherits"] = get(kind_totals, "inherits", 0) + c_inh
                kind_totals["overrides"] = get(kind_totals, "overrides", 0) + c_ov
                kind_totals["virtual_dispatch"] = get(kind_totals, "virtual_dispatch", 0) + c_vcall
                push!(pair_rows, (tot, c_calls, c_mfa, c_ft, c_inh + c_ov, c_vcall, a, b))
            end
            println(io, "   Internal edge weight totals: ", kind_totals)
            sort!(pair_rows; by=x -> (-x[1], x[7], x[8]))
            println(io, "   Internal module pairs ($(length(pair_rows)) directed pairs):")
            limit = length(comp) <= 6 ? length(pair_rows) : min(45, length(pair_rows))
            for idx_p in 1:limit
                tot, c_calls, c_mfa, c_ft, c_io, c_vcall, a, b = pair_rows[idx_p]
                @printf(io, "     %-34s -> %-34s | tot=%4d (calls=%3d, mfa=%3d, ft=%2d, inh/ov=%2d, vcall=%3d)\n",
                        a, b, tot, c_calls, c_mfa, c_ft, c_io, c_vcall)
                # Print top 2 concrete edge examples for small SCCs or top 15 pairs
                if length(comp) <= 6 || idx_p <= 20
                    exs = sort!(collect(get(edge_examples, (a, b), Dict{String,Int}())); by=x -> -x[2])
                    for (lbl, cnt) in exs[1:min(3, length(exs))]
                        println(io, "         e.g. [$cnt] $lbl")
                    end
                end
            end
            if length(pair_rows) > limit
                println(io, "     ... ($(length(pair_rows) - limit) more directed pairs)")
            end
        end
        println(io)

        println(io, "5. GRAPH 2: V1 FINE-GRAINED METHOD/FUNCTION (MF) CALL GRAPH")
        multi2 = sort!([i for i in 1:length(sccs2) if length(sccs2[i]) > 1]; by=i -> (-length(sccs2[i]), layers2[i]))
        in_cycles2 = sum(length(sccs2[i]) for i in multi2; init=0)
        println(io, "   Defined V1 MF nodes: $(length(mf_nodes))")
        println(io, "   Directed call+vdispatch edges: $(sum(length, values(g2_adj)))")
        println(io, "   Total SCCs: $(length(sccs2))")
        println(io, "   Pure singletons: $(length(sccs2) - length(multi2))")
        println(io, "   Multi-node MF SCCs: $(length(multi2)) (covering $(in_cycles2) MF nodes)")
        println(io, "   Max topological layer: $(maximum(layers2))")
        println(io)
        println(io, "   --- All Multi-Node MF SCCs ---")
        for i in multi2
            comp = sccs2[i]
            loc_c = sum(nodes[u].loc for u in comp)
            println(io, "   MF-SCC #$(i) (Layer $(layers2[i]), size=$(length(comp)), loc=$(loc_c)):")
            for u in comp
                n = nodes[u]
                println(io, "     - $(n.qualname) [$(n.file):$(n.line_start), loc=$(n.loc)]")
            end
        end
        println(io)

        println(io, "   --- MF Node Distribution Across Layers (0..$(maximum(layers2))) ---")
        for lay in 0:maximum(layers2)
            us = [u for u in mf_nodes if layers2[node_to_scc2[u]] == lay]
            loc_l = sum(nodes[u].loc for u in us; init=0)
            # top 3 clusters in this layer
            ccounts = Dict{String, Int}()
            for u in us
                cl = assign_cluster(something(module_id(u), ""))
                ccounts[cl] = get(ccounts, cl, 0) + 1
            end
            topc = sort!(collect(ccounts); by=x -> -x[2])
            top_str = join(["$(c)($(k))" for (c, k) in topc[1:min(4, length(topc))]], ", ")
            @printf(io, "   Layer %2d: %4d MF nodes | %6d LOC | top: %s\n", lay, length(us), loc_l, top_str)
        end
        println(io)

        println(io, "6. KEY V1 CLUSTER DEPENDENCY PROFILES (INTERNAL vs EXTERNAL)")
        cluster_order = [
            "CoinUtils helpers/sort/error/message",
            "CoinIndexedVector",
            "CoinPackedVector/CoinShallowPackedVector",
            "CoinPackedMatrix",
            "CoinDenseFactorization",
            "CoinFactorization",
            "CoinPresolve*",
            "CoinMpsIO",
            "CoinTable",
            "ClpHelperFunctions",
            "ClpPackedMatrix (+ ClpMatrixBase)",
            "ClpFactorization",
            "ClpNonLinearCost",
            "ClpDualRowSteepest/Dantzig",
            "ClpPrimalColumnSteepest/Dantzig",
            "ClpOutput/Events",
            "ClpPresolve",
            "Idiot",
            "ClpModel",
            "ClpSimplex",
            "ClpSimplexDual",
            "ClpSimplexPrimal",
            "ClpSolve",
        ]
        # include any remaining clusters
        all_clusters = unique([assign_cluster(m) for m in g1_vlist])
        for c in sort(all_clusters)
            c in cluster_order || push!(cluster_order, c)
        end

        for cl in cluster_order
            mods = sort!([m for m in g1_vlist if assign_cluster(m) == cl])
            isempty(mods) && continue
            mod_set = Set(mods)
            mf_cnt = sum(mod_mf_def[m] for m in mods)
            loc_cnt = sum(mod_loc[m] for m in mods)
            lays = sort!(unique([layers1[node_to_scc1[m]] for m in mods]))

            # Compute internal vs external outgoing & incoming edges across Graph 1 edge kinds
            int_calls = 0
            ext_out = Dict{String, Int}()
            ext_in  = Dict{String, Int}()
            for e in edges
                (e.src in vis_full && e.dst in vis_full) || continue
                (e.kind in g1_direct_kinds || e.kind == "overrides") || continue
                ms = module_id(e.src); md = module_id(e.dst)
                (ms === nothing || md === nothing) && continue
                cs = assign_cluster(ms); cd = assign_cluster(md)
                if cs == cl && cd == cl
                    e.kind == "calls" && (int_calls += e.count)
                elseif cs == cl && cd != cl
                    ext_out[cd] = get(ext_out, cd, 0) + e.count
                elseif cs != cl && cd == cl
                    ext_in[cs] = get(ext_in, cs, 0) + e.count
                end
            end
            # also virtual dispatch outgoing from cl
            for e in edges
                (e.src in vis_full && e.dst in vis_full && e.kind == "calls") || continue
                ms = module_id(e.src)
                ms === nothing && continue
                assign_cluster(ms) == cl || continue
                for ov in all_v1_overrides(e.dst)
                    mo = module_id(ov)
                    mo === nothing && continue
                    cd = assign_cluster(mo)
                    cd != cl && (ext_out[cd] = get(ext_out, cd, 0) + e.count)
                end
            end

            println(io, "\n   [Cluster] $cl")
            println(io, "     Modules ($(length(mods))): ", join(mods, ", "))
            println(io, "     Defined MFs: $mf_cnt | LOC: $loc_cnt | Graph1 SCC layers: $lays | Internal calls: $int_calls")
            out_sorted = sort!(collect(ext_out); by=x -> -x[2])
            in_sorted  = sort!(collect(ext_in); by=x -> -x[2])
            println(io, "     Outgoing deps ($(length(out_sorted))): ",
                    isempty(out_sorted) ? "NONE" : join(["$(k)($(v))" for (k, v) in out_sorted], ", "))
            println(io, "     Incoming deps ($(length(in_sorted))): ",
                    isempty(in_sorted) ? "NONE" : join(["$(k)($(v))" for (k, v) in in_sorted], ", "))
        end
    end

    println("Analysis complete:")
    println("  v1_inventory_types.tsv: $(length(g1_vlist)) modules")
    println("  v1_inventory_mf.tsv:    $(length(mf_nodes)) defined MF nodes")
    println("  v1_scc_report.txt:      written")
end

main()
