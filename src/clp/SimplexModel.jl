# Pure idiomatic 1-based Julia standalone ClpSimplexModel (Phase B — B12c-1).
# Zero Ptr, zero unsafe_load/unsafe_store!, zero PodView, --trim=safe clean.
# Composes all validated B8..B12b pure-Julia bricks with zero C++ Clp dependency.

"""
    ClpSimplexModel

Standalone pure-Julia LP simplex solver state matching Clp's `ClpSimplex` layout.
All arrays and indices are 1-based: columns `1:ncols`, rows/slacks `(ncols+1):(ncols+nrows)`.
"""
mutable struct ClpSimplexModel
    name::String
    nrows::Int
    ncols::Int
    obj_sense::Float64        # 1.0 = minimize, -1.0 = maximize
    obj_offset::Float64

    # Original unscaled problem data (1-based CSC)
    mat_starts::Vector{Int32}
    mat_lengths::Vector{Int32}
    mat_indices::Vector{Int32}
    mat_elements::Vector{Float64}
    col_lower::Vector{Float64}
    col_upper::Vector{Float64}
    objective::Vector{Float64}
    row_lower::Vector{Float64}
    row_upper::Vector{Float64}

    # Scaling & row copy (1-based CSR of active scaled/unscaled matrix)
    scaling_flag::Int         # 0=off, 1=equilibrium, 3=geometric (default 1)
    has_scale::Bool
    row_scale::Vector{Float64}
    col_scale::Vector{Float64}
    inv_row_scale::Vector{Float64}
    inv_col_scale::Vector{Float64}
    scaled_elements::Vector{Float64}
    obj_scale::Float64
    rhs_scale::Float64
    has_row_copy::Bool
    rc_starts::Vector{Int32}
    rc_cols::Vector{Int32}
    rc_elements::Vector{Float64}

    # Working rim arrays (length ntotal = nrows + ncols)
    rim_active::Bool
    solution::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
    cost::Vector{Float64}
    dj::Vector{Float64}
    status::Vector{UInt8}
    pivot_var::Vector{Int32}

    # Unscaled user-facing primal/dual solution
    col_activity::Vector{Float64}
    row_activity::Vector{Float64}
    dual_row_sol::Vector{Float64}
    reduced_cost::Vector{Float64}
    dual_work::Vector{Float64}

    # NonLinearCost (method 2)
    nlc_status::Vector{UInt8}
    nlc_bound::Vector{Float64}
    nlc_cost2::Vector{Float64}
    nlc_infeas_weight::Float64
    nlc_num_infeas::Int
    nlc_sum_infeas::Float64
    nlc_change_cost::Float64
    nlc_avg_theta::Float64

    # Factorization & indexed workspaces
    cf::ClpFactorization
    r0_val::Vector{Float64}; r0_idx::Vector{Int32}
    r1_val::Vector{Float64}; r1_idx::Vector{Int32}
    r2_val::Vector{Float64}; r2_idx::Vector{Int32}
    r3_val::Vector{Float64}; r3_idx::Vector{Int32}
    c0_val::Vector{Float64}; c0_idx::Vector{Int32}
    c1_val::Vector{Float64}; c1_idx::Vector{Int32}
    marked_buf::Vector{UInt8}
    work_col::Vector{Float64}
    work_row::Vector{Float64}
    pivot_temp::Vector{Int32}

    # Dual Row Steepest state (B12a / B12c-2)
    drs_weights::Vector{Float64}
    drs_infeas_val::Vector{Float64}
    drs_infeas_idx::Vector{Int32}
    drs_infeas_nnz::Int
    drs_mode::Int

    # Primal Column Steepest state (B12b / B12c-2)
    pcs_weights::Vector{Float64}
    pcs_ref::Vector{UInt32}
    pcs_infeas_val::Vector{Float64}
    pcs_infeas_idx::Vector{Int32}
    pcs_infeas_nnz::Int
    pcs_alt_val::Vector{Float64}
    pcs_alt_idx::Vector{Int32}
    pcs_alt_nnz::Int
    pcs_mode::Int
    pcs_num_switched::Int
    pcs_pivot_seq::Int
    pcs_devex::Float64

    # Progress & tolerances & solver status
    prog_in::Vector{Int32}
    prog_out::Vector{Int32}
    prog_way::Vector{Int8}
    prog_bad_times::Int

    primal_tol::Float64
    dual_tol::Float64
    zero_tol::Float64
    infeas_cost::Float64
    large_value::Float64
    max_iterations::Int
    special_options::Int
    more_special_options::Int

    problem_status::Int
    secondary_status::Int
    number_iterations::Int
    objective_value::Float64
    sum_primal_infeas::Float64
    num_primal_infeas::Int
    largest_primal_err::Float64
    sum_dual_infeas::Float64
    num_dual_infeas::Int
    largest_dual_err::Float64

    force_factorization::Int
    last_bad_iter::Int
    last_good_iter::Int
    change_made::Int
    progress_flag::Int
    alpha_accuracy::Float64
    acceptable_pivot::Float64
    rng_seed::UInt32
end

function ClpSimplexModel()
    return ClpSimplexModel(
        "", 0, 0, 1.0, 0.0,
        Int32[1], Int32[], Int32[], Float64[],
        Float64[], Float64[], Float64[], Float64[], Float64[],
        1, false, Float64[], Float64[], Float64[], Float64[], Float64[], 1.0, 1.0,
        false, Int32[1], Int32[], Float64[],
        false, Float64[], Float64[], Float64[], Float64[], Float64[], UInt8[], Int32[],
        Float64[], Float64[], Float64[], Float64[], Float64[],
        UInt8[], Float64[], Float64[], 1.0e10, 0, 0.0, 0.0, 1.0,
        ClpFactorization(),
        Float64[], Int32[], Float64[], Int32[], Float64[], Int32[], Float64[], Int32[],
        Float64[], Int32[], Float64[], Int32[],
        UInt8[], Float64[], Float64[], Int32[],
        Float64[], Float64[], Int32[], 0, 3,
        Float64[], UInt32[], Float64[], Int32[], 0, Float64[], Int32[], 0, 3, 0, 0, 1.0,
        fill(Int32(-1), CLP_CYCLE), fill(Int32(-1), CLP_CYCLE), zeros(Int8, CLP_CYCLE), 0,
        1.0e-7, 1.0e-7, 1.0e-13, 1.0e10, 1.0e15, typemax(Int32), 0, 0,
        -1, 0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0,
        -1, -1, 0, 0, 0, -1.0, 1.0e-9, UInt32(1234567),
    )
end

@inline function _sm_active_elements(m::ClpSimplexModel)::Vector{Float64}
    return m.has_scale ? m.scaled_elements : m.mat_elements
end

"""
    load_problem!(m::ClpSimplexModel, nrows, ncols, col_starts, col_indices, col_elements,
                  col_lower, col_upper, objective, row_lower, row_upper;
                  obj_sense=1.0, obj_offset=0.0, name="")

Load a 1-based CSC LP into `m` and allocate all working buffers.
"""
function load_problem!(
    m::ClpSimplexModel,
    nrows::Int,
    ncols::Int,
    col_starts::AbstractVector{<:Integer},
    col_indices::AbstractVector{<:Integer},
    col_elements::AbstractVector{<:Real},
    col_lower::AbstractVector{<:Real},
    col_upper::AbstractVector{<:Real},
    objective::AbstractVector{<:Real},
    row_lower::AbstractVector{<:Real},
    row_upper::AbstractVector{<:Real};
    obj_sense::Float64 = 1.0,
    obj_offset::Float64 = 0.0,
    name::AbstractString = "",
)::ClpSimplexModel
    ntotal = nrows + ncols
    nnz = length(col_elements)

    m.name = String(name)
    m.nrows = nrows
    m.ncols = ncols
    m.obj_sense = obj_sense
    m.obj_offset = obj_offset

    m.mat_starts = Int32[Int32(col_starts[j]) for j in 1:(ncols + 1)]
    m.mat_lengths = Vector{Int32}(undef, ncols)
    @inbounds for j in 1:ncols
        m.mat_lengths[j] = m.mat_starts[j + 1] - m.mat_starts[j]
    end
    m.mat_indices = Int32[Int32(col_indices[k]) for k in 1:nnz]
    m.mat_elements = Float64[Float64(col_elements[k]) for k in 1:nnz]

    m.col_lower = Float64[Float64(col_lower[j]) for j in 1:ncols]
    m.col_upper = Float64[Float64(col_upper[j]) for j in 1:ncols]
    m.objective = Float64[Float64(objective[j]) for j in 1:ncols]
    m.row_lower = Float64[Float64(row_lower[i]) for i in 1:nrows]
    m.row_upper = Float64[Float64(row_upper[i]) for i in 1:nrows]

    # Reset scaling & row copy
    m.has_scale = false
    m.row_scale = Float64[]
    m.col_scale = Float64[]
    m.inv_row_scale = Float64[]
    m.inv_col_scale = Float64[]
    m.scaled_elements = Float64[]
    m.obj_scale = 1.0
    m.rhs_scale = 1.0
    m.has_row_copy = false
    m.rc_starts = Int32[1]
    m.rc_cols = Int32[]
    m.rc_elements = Float64[]

    # Working rim & solution arrays
    m.rim_active = false
    m.solution = zeros(Float64, ntotal)
    m.lower = zeros(Float64, ntotal)
    m.upper = zeros(Float64, ntotal)
    m.cost = zeros(Float64, ntotal)
    m.dj = zeros(Float64, ntotal)
    m.status = zeros(UInt8, ntotal)
    m.pivot_var = Vector{Int32}(undef, nrows)

    m.col_activity = zeros(Float64, ncols)
    m.row_activity = zeros(Float64, nrows)
    m.dual_row_sol = zeros(Float64, nrows)
    m.reduced_cost = zeros(Float64, ncols)
    m.dual_work = zeros(Float64, nrows)

    # NonLinearCost
    m.nlc_status = fill(UInt8((CLP_NLC_SAME << 4) | CLP_NLC_FEASIBLE), ntotal)
    m.nlc_bound = zeros(Float64, ntotal)
    m.nlc_cost2 = zeros(Float64, ntotal)
    m.nlc_infeas_weight = m.infeas_cost
    m.nlc_num_infeas = 0
    m.nlc_sum_infeas = 0.0
    m.nlc_change_cost = 0.0
    m.nlc_avg_theta = 1.0

    # Factorization & workspaces (rowArray capacity >= max(nrows + max_pivots, ntotal)
    # for Forrest-Tomlin updates and dual ratio-test candidate lists on wide models)
    m.cf = ClpFactorization()
    rcap = max(nrows + max(m.cf.lu.max_pivots, 500), ntotal) + 16
    ccap = ntotal + 16
    m.r0_val = zeros(Float64, rcap); m.r0_idx = zeros(Int32, rcap)
    m.r1_val = zeros(Float64, rcap); m.r1_idx = zeros(Int32, rcap)
    m.r2_val = zeros(Float64, rcap); m.r2_idx = zeros(Int32, rcap)
    m.r3_val = zeros(Float64, rcap); m.r3_idx = zeros(Int32, rcap)
    m.c0_val = zeros(Float64, ccap); m.c0_idx = zeros(Int32, ccap)
    m.c1_val = zeros(Float64, ccap); m.c1_idx = zeros(Int32, ccap)
    m.marked_buf = zeros(UInt8, max(nrows, ncols))
    m.work_col = zeros(Float64, ncols)
    m.work_row = zeros(Float64, nrows)
    m.pivot_temp = zeros(Int32, max(ntotal, 1))

    # Dual & Primal steepest arrays
    m.drs_weights = ones(Float64, nrows)
    m.drs_infeas_val = zeros(Float64, nrows)
    m.drs_infeas_idx = zeros(Int32, nrows)
    m.drs_infeas_nnz = 0

    ref_words = (ntotal + 31) >> 5
    m.pcs_weights = ones(Float64, ntotal)
    m.pcs_ref = zeros(UInt32, ref_words)
    m.pcs_infeas_val = zeros(Float64, ntotal)
    m.pcs_infeas_idx = zeros(Int32, ntotal)
    m.pcs_infeas_nnz = 0
    m.pcs_alt_val = zeros(Float64, rcap)
    m.pcs_alt_idx = zeros(Int32, rcap)
    m.pcs_alt_nnz = 0
    m.pcs_num_switched = 0
    m.pcs_pivot_seq = 0
    m.pcs_devex = 1.0

    # Progress & status
    fill!(m.prog_in, Int32(-1))
    fill!(m.prog_out, Int32(-1))
    fill!(m.prog_way, Int8(0))
    m.prog_bad_times = 0

    m.problem_status = -1
    m.secondary_status = 0
    m.number_iterations = 0
    m.objective_value = 0.0
    m.sum_primal_infeas = 0.0
    m.num_primal_infeas = 0
    m.largest_primal_err = 0.0
    m.sum_dual_infeas = 0.0
    m.num_dual_infeas = 0
    m.largest_dual_err = 0.0
    m.force_factorization = -1
    m.last_bad_iter = -1
    m.last_good_iter = 0
    m.change_made = 0
    m.progress_flag = 0
    m.alpha_accuracy = -1.0
    m.acceptable_pivot = 1.0e-9

    all_slack_basis!(m)
    return m
end

"""
    read_mps!(m::ClpSimplexModel, path::AbstractString) -> ClpSimplexModel

Read an MPS file and load it into `m`.
"""
function read_mps!(m::ClpSimplexModel, path::AbstractString)::ClpSimplexModel
    data = clp_read_mps(path)
    return load_problem!(
        m, data.nrows, data.ncols,
        data.col_starts, data.col_indices, data.col_elements,
        data.col_lower, data.col_upper, data.objective,
        data.row_lower, data.row_upper;
        obj_sense = data.obj_sense,
        obj_offset = data.obj_offset,
        name = data.name,
    )
end

"""
    build_row_copy!(m::ClpSimplexModel)

Build 1-based CSR row copy (`rc_starts`, `rc_cols`, `rc_elements`) of the active matrix
(scaled if `m.has_scale`, otherwise unscaled).
"""
function build_row_copy!(m::ClpSimplexModel)::Nothing
    nrows = m.nrows
    ncols = m.ncols
    starts = m.mat_starts
    indices = m.mat_indices
    elems = _sm_active_elements(m)
    nnz = length(elems)

    rc_starts = zeros(Int32, nrows + 1)
    @inbounds for k in 1:nnz
        rc_starts[Int(indices[k]) + 1] += Int32(1)
    end
    pos = Int32(1)
    @inbounds for i in 1:nrows
        cnt = rc_starts[i + 1]
        rc_starts[i] = pos
        pos += cnt
    end
    rc_starts[nrows + 1] = pos

    rc_cols = Vector{Int32}(undef, nnz)
    rc_elems = Vector{Float64}(undef, nnz)
    cur = copy(rc_starts)
    @inbounds for j in 1:ncols
        s = Int(starts[j])
        e = Int(starts[j + 1]) - 1
        for k in s:e
            ir = Int(indices[k])
            p = Int(cur[ir])
            rc_cols[p] = Int32(j)
            rc_elems[p] = elems[k]
            cur[ir] = Int32(p + 1)
        end
    end

    m.rc_starts = rc_starts
    m.rc_cols = rc_cols
    m.rc_elements = rc_elems
    m.has_row_copy = true
    return nothing
end

"""
    scale_model!(m::ClpSimplexModel; mode::Int=m.scaling_flag) -> Bool

Scale the constraint matrix using Clp's equilibrium (`mode == 1`) or geometric (`mode == 3`)
scaling, populating `row_scale`, `col_scale`, `inv_row_scale`, `inv_col_scale`, and
`scaled_elements`. Returns `true` if scaling was applied.
"""
function scale_model!(m::ClpSimplexModel; mode::Int = m.scaling_flag)::Bool
    nrows = m.nrows
    ncols = m.ncols
    if mode <= 0 || nrows == 0 || ncols == 0
        m.has_scale = false
        m.row_scale = Float64[]
        m.col_scale = Float64[]
        m.inv_row_scale = Float64[]
        m.inv_col_scale = Float64[]
        m.scaled_elements = Float64[]
        return false
    end

    sn, ln, sp, lp = clp_mat_range_of_elements(m.mat_starts, m.mat_lengths, m.mat_elements, ncols)
    smallest = min(sp, -ln)
    largest = max(lp, -sn)
    if largest <= 0.0 || smallest >= floatmax(Float64) * 0.5
        m.has_scale = false
        return false
    end
    # Skip scaling if already well-conditioned (Clp heuristic: ratio <= 2.0 and near 1)
    if smallest >= 0.5 && largest <= 2.0
        m.has_scale = false
        return false
    end

    rs = ones(Float64, nrows)
    cs = ones(Float64, ncols)
    row_max = zeros(Float64, nrows)
    row_min = fill(floatmax(Float64), nrows)

    starts = m.mat_starts
    indices = m.mat_indices
    elems = m.mat_elements

    if mode == 1
        # Equilibrium scaling (Clp default): normalize rows by max abs, then columns by max abs
        @inbounds for j in 1:ncols
            for k in Int(starts[j]):(Int(starts[j + 1]) - 1)
                ir = Int(indices[k])
                v = abs(elems[k])
                if v > row_max[ir]
                    row_max[ir] = v
                end
            end
        end
        @inbounds for i in 1:nrows
            rs[i] = row_max[i] > 1.0e-12 ? (1.0 / row_max[i]) : 1.0
        end
        @inbounds for j in 1:ncols
            cmax = 0.0
            for k in Int(starts[j]):(Int(starts[j + 1]) - 1)
                v = abs(elems[k]) * rs[Int(indices[k])]
                if v > cmax
                    cmax = v
                end
            end
            cs[j] = cmax > 1.0e-12 ? (1.0 / cmax) : 1.0
        end
    else
        # Geometric scaling (3 passes of geometric mean + 1 final equilibrium pass, matching Clp)
        for _ in 1:3
            fill!(row_max, 0.0)
            fill!(row_min, floatmax(Float64))
            @inbounds for j in 1:ncols
                cj = cs[j]
                for k in Int(starts[j]):(Int(starts[j + 1]) - 1)
                    ir = Int(indices[k])
                    v = abs(elems[k]) * rs[ir] * cj
                    if v > 1.0e-12
                        if v > row_max[ir]; row_max[ir] = v; end
                        if v < row_min[ir]; row_min[ir] = v; end
                    end
                end
            end
            @inbounds for i in 1:nrows
                if row_max[i] > 0.0 && row_min[i] < floatmax(Float64) * 0.5
                    g = sqrt(row_max[i] * row_min[i])
                    if g > 1.0e-12
                        rs[i] /= g
                    end
                end
            end
            @inbounds for j in 1:ncols
                cmax = 0.0
                cmin = floatmax(Float64)
                cj = cs[j]
                for k in Int(starts[j]):(Int(starts[j + 1]) - 1)
                    v = abs(elems[k]) * rs[Int(indices[k])] * cj
                    if v > 1.0e-12
                        if v > cmax; cmax = v; end
                        if v < cmin; cmin = v; end
                    end
                end
                if cmax > 0.0 && cmin < floatmax(Float64) * 0.5
                    g = sqrt(cmax * cmin)
                    if g > 1.0e-12
                        cs[j] /= g
                    end
                end
            end
        end
        # Final equilibrium pass on rows & cols
        fill!(row_max, 0.0)
        @inbounds for j in 1:ncols
            cj = cs[j]
            for k in Int(starts[j]):(Int(starts[j + 1]) - 1)
                ir = Int(indices[k])
                v = abs(elems[k]) * rs[ir] * cj
                if v > row_max[ir]; row_max[ir] = v; end
            end
        end
        @inbounds for i in 1:nrows
            if row_max[i] > 1.0e-12
                rs[i] /= row_max[i]
            end
        end
        @inbounds for j in 1:ncols
            cmax = 0.0
            cj = cs[j]
            for k in Int(starts[j]):(Int(starts[j + 1]) - 1)
                v = abs(elems[k]) * rs[Int(indices[k])] * cj
                if v > cmax; cmax = v; end
            end
            if cmax > 1.0e-12
                cs[j] /= cmax
            end
        end
    end

    # Clamp extreme scale factors (matching Clp's 1e-20 .. 1e20 safety)
    inv_rs = Vector{Float64}(undef, nrows)
    inv_cs = Vector{Float64}(undef, ncols)
    @inbounds for i in 1:nrows
        s = clamp(rs[i], 1.0e-20, 1.0e20)
        rs[i] = s
        inv_rs[i] = 1.0 / s
    end
    @inbounds for j in 1:ncols
        s = clamp(cs[j], 1.0e-20, 1.0e20)
        cs[j] = s
        inv_cs[j] = 1.0 / s
    end

    scaled_el = copy(elems)
    clp_mat_really_scale!(starts, m.mat_lengths, indices, scaled_el, ncols, rs, cs)

    m.row_scale = rs
    m.col_scale = cs
    m.inv_row_scale = inv_rs
    m.inv_col_scale = inv_cs
    m.scaled_elements = scaled_el
    m.has_scale = true
    return true
end

"""
    all_slack_basis!(m::ClpSimplexModel)

Initialize an all-slack basis (all rows basic, all columns at nearest bound/0).
"""
function all_slack_basis!(m::ClpSimplexModel)::Nothing
    nrows = m.nrows
    ncols = m.ncols
    clp_all_slack_basis!(
        m.col_activity, m.status, m.col_lower, m.col_upper, ncols,
        nothing, nothing, 1.0,
    )
    @inbounds for i in 1:nrows
        m.status[ncols + i] = _sb_set_status(UInt8(0), CLP_STATUS_BASIC)
        m.pivot_var[i] = Int32(ncols + i)
        m.row_activity[i] = 0.0
    end
    return nothing
end

"""
    create_rim!(m::ClpSimplexModel)

Build scaled working rim arrays (`lower`, `upper`, `cost`, `solution`, `pivot_var`,
`nlc_*`) from the model's unscaled bounds, costs, and current basis `status`.
Preserves an existing warm-start basis if `status` already has `nrows` basic variables.
"""
function create_rim!(m::ClpSimplexModel)::Nothing
    nrows = m.nrows
    ncols = m.ncols
    ntotal = nrows + ncols

    rs = m.has_scale ? m.row_scale : nothing
    cs = m.has_scale ? m.col_scale : nothing
    ics = m.has_scale ? m.inv_col_scale : nothing

    clp_create_rim1!(
        m.lower, m.upper, m.col_lower, m.col_upper, m.row_lower, m.row_upper,
        rs, ics, nrows, 0, ncols, nrows, ncols, m.special_options, false,
        m.rhs_scale, m.primal_tol,
    )
    clp_create_rim4!(
        m.cost, m.objective, nothing, rs, cs,
        nrows, 0, ncols, nrows, ncols, m.special_options, false,
        m.obj_sense * m.obj_scale,
    )

    # Populate working solution from unscaled activities (clamped to bounds for nonbasic)
    @inbounds for j in 1:ncols
        st = _sb_get_status(m.status[j])
        if st == CLP_STATUS_AT_LOWER || st == CLP_STATUS_IS_FIXED
            m.solution[j] = m.lower[j]
        elseif st == CLP_STATUS_AT_UPPER
            m.solution[j] = m.upper[j]
        elseif st == CLP_STATUS_IS_FREE
            v = m.has_scale ? (m.col_activity[j] * (m.rhs_scale * m.inv_col_scale[j])) : (m.col_activity[j] * m.rhs_scale)
            m.solution[j] = isfinite(v) ? v : 0.0
        else
            m.solution[j] = m.has_scale ? (m.col_activity[j] * (m.rhs_scale * m.inv_col_scale[j])) : (m.col_activity[j] * m.rhs_scale)
        end
    end
    @inbounds for i in 1:nrows
        seq = ncols + i
        st = _sb_get_status(m.status[seq])
        if st == CLP_STATUS_AT_LOWER || st == CLP_STATUS_IS_FIXED
            m.solution[seq] = m.lower[seq]
        elseif st == CLP_STATUS_AT_UPPER
            m.solution[seq] = m.upper[seq]
        elseif st == CLP_STATUS_IS_FREE
            v = m.has_scale ? (m.row_activity[i] * (m.rhs_scale * m.row_scale[i])) : (m.row_activity[i] * m.rhs_scale)
            m.solution[seq] = isfinite(v) ? v : 0.0
        else
            m.solution[seq] = m.has_scale ? (m.row_activity[i] * (m.rhs_scale * m.row_scale[i])) : (m.row_activity[i] * m.rhs_scale)
        end
    end

    # Clean status & rebuild pivot_var from basic variables
    nbasic = 0
    @inbounds for seq in 1:ntotal
        st = _sb_get_status(m.status[seq])
        if st == CLP_STATUS_BASIC
            if nbasic < nrows
                nbasic += 1
                m.pivot_var[nbasic] = Int32(seq)
            else
                # Extra basic -> push to bound
                lo = m.lower[seq]
                hi = m.upper[seq]
                if lo == hi
                    m.status[seq] = _sb_set_status(m.status[seq], CLP_STATUS_IS_FIXED)
                    m.solution[seq] = lo
                elseif lo > -1.0e20
                    m.status[seq] = _sb_set_status(m.status[seq], CLP_STATUS_AT_LOWER)
                    m.solution[seq] = lo
                elseif hi < 1.0e20
                    m.status[seq] = _sb_set_status(m.status[seq], CLP_STATUS_AT_UPPER)
                    m.solution[seq] = hi
                else
                    m.status[seq] = _sb_set_status(m.status[seq], CLP_STATUS_IS_FREE)
                    m.solution[seq] = 0.0
                end
            end
        elseif st == CLP_STATUS_AT_LOWER && m.lower[seq] == m.upper[seq]
            m.status[seq] = _sb_set_status(m.status[seq], CLP_STATUS_IS_FIXED)
        end
    end
    if nbasic < nrows
        # Fill missing basis slots with nonbasic slacks
        @inbounds for i in 1:nrows
            seq = ncols + i
            if _sb_get_status(m.status[seq]) != CLP_STATUS_BASIC
                nbasic += 1
                m.status[seq] = _sb_set_status(m.status[seq], CLP_STATUS_BASIC)
                m.pivot_var[nbasic] = Int32(seq)
                nbasic == nrows && break
            end
        end
    end

    # Initialize NonLinearCost arrays
    @inbounds for seq in 1:ntotal
        m.nlc_status[seq] = UInt8((CLP_NLC_SAME << 4) | CLP_NLC_FEASIBLE)
        m.nlc_bound[seq] = 0.0
        m.nlc_cost2[seq] = m.cost[seq]
    end
    m.nlc_num_infeas = 0
    m.nlc_sum_infeas = 0.0
    m.nlc_change_cost = 0.0

    # Ensure row copy exists for fast transposeTimes
    if !m.has_row_copy
        build_row_copy!(m)
    end

    m.rim_active = true
    return nothing
end

"""
    delete_rim!(m::ClpSimplexModel)

Unscale working `solution` and `dj` back into `col_activity`, `row_activity`,
`dual_row_sol`, `reduced_cost`, and compute the true unscaled `objective_value`.
"""
function delete_rim!(m::ClpSimplexModel)::Nothing
    nrows = m.nrows
    ncols = m.ncols
    ntotal = nrows + ncols
    col_act_w = @view m.solution[1:ncols]
    row_act_w = @view m.solution[(ncols + 1):ntotal]
    col_lo_w = @view m.lower[1:ncols]
    col_hi_w = @view m.upper[1:ncols]
    row_lo_w = @view m.lower[(ncols + 1):ntotal]
    row_hi_w = @view m.upper[(ncols + 1):ntotal]
    col_dj_w = @view m.dj[1:ncols]

    rs = m.has_scale ? m.row_scale : nothing
    irs = m.has_scale ? m.inv_row_scale : nothing
    cs = m.has_scale ? m.col_scale : nothing
    ics = m.has_scale ? m.inv_col_scale : nothing

    @inbounds for i in 1:nrows
        m.dual_row_sol[i] = m.dual_work[i]
    end
    clp_delete_rim_unscale!(
        m.col_activity, m.reduced_cost, m.row_activity, m.dual_row_sol,
        col_act_w, col_lo_w, col_hi_w, col_dj_w,
        row_act_w, row_lo_w, row_hi_w,
        m.col_lower, m.col_upper, m.row_lower, m.row_upper,
        nothing, rs, irs, cs, ics, nothing,
        nrows, ncols, true,
        m.obj_scale, m.rhs_scale, m.primal_tol, m.dual_tol,
        m.problem_status, m.secondary_status,
    )

    # True unscaled objective value (in user's original sense + offset)
    obj = -m.obj_offset
    @inbounds for j in 1:ncols
        obj += m.objective[j] * m.col_activity[j]
    end
    m.objective_value = obj
    m.rim_active = false
    return nothing
end

"""
    internal_factorize!(m::ClpSimplexModel) -> Int

Factorize the current basis stored in `m.status` / `m.pivot_var` using `ClpFactorization`.
Automatically repairs singular bases by replacing singular basic variables with slacks.
Returns `0` on success, `< 0` on unrecoverable failure.
"""
function internal_factorize!(m::ClpSimplexModel)::Int
    nrows = m.nrows
    ncols = m.ncols
    ntotal = nrows + ncols
    elems = _sm_active_elements(m)

    for attempt in 1:5
        # Collect basic slacks first (1-based row index), then basic structurals (1-based col index)
        n_row_basic = 0
        @inbounds for i in 1:nrows
            seq = Int(m.pivot_var[i])
            if seq > ncols
                n_row_basic += 1
                m.pivot_temp[n_row_basic] = Int32(seq - ncols)
            end
        end
        n_basic = n_row_basic
        @inbounds for i in 1:nrows
            seq = Int(m.pivot_var[i])
            if seq <= ncols
                n_basic += 1
                m.pivot_temp[n_basic] = Int32(seq)
            end
        end

        clp_fact_factorize_packed!(
            m.cf, nrows, ncols, n_row_basic, n_basic, m.pivot_temp,
            m.mat_starts, m.mat_lengths, m.mat_indices, elems,
            false, nothing, nothing, m.number_iterations, attempt, m.pivot_var, nrows,
        )
        status = m.cf.lu.status
        if status == 0
            return 0
        elseif status == -1
            # Singular basis: replace unpivoted positions with slacks and retry
            # Mark which slacks are already basic
            @inbounds for i in 1:nrows
                m.marked_buf[i] = UInt8(0)
            end
            @inbounds for i in 1:nrows
                seq = Int(m.pivot_var[i])
                if seq > ncols
                    m.marked_buf[seq - ncols] = UInt8(1)
                end
            end
            # Fall back to all-slack if repeated singularity, else replace unpivoted slots
            if attempt >= 4
                all_slack_basis!(m)
            else
                next_slack = 1
                @inbounds for i in 1:nrows
                    if m.cf.lu.permute[i] <= Int32(0)
                        old_seq = Int(m.pivot_var[i])
                        # Demote old_seq to bound
                        lo = m.lower[old_seq]
                        hi = m.upper[old_seq]
                        if lo == hi
                            m.status[old_seq] = _sb_set_status(m.status[old_seq], CLP_STATUS_IS_FIXED)
                            m.solution[old_seq] = lo
                        elseif abs(m.solution[old_seq] - lo) <= abs(hi - m.solution[old_seq]) && lo > -1.0e20
                            m.status[old_seq] = _sb_set_status(m.status[old_seq], CLP_STATUS_AT_LOWER)
                            m.solution[old_seq] = lo
                        elseif hi < 1.0e20
                            m.status[old_seq] = _sb_set_status(m.status[old_seq], CLP_STATUS_AT_UPPER)
                            m.solution[old_seq] = hi
                        else
                            m.status[old_seq] = _sb_set_status(m.status[old_seq], CLP_STATUS_IS_FREE)
                            m.solution[old_seq] = 0.0
                        end
                        while next_slack <= nrows && m.marked_buf[next_slack] != UInt8(0)
                            next_slack += 1
                        end
                        if next_slack <= nrows
                            new_seq = ncols + next_slack
                            m.marked_buf[next_slack] = UInt8(1)
                            m.status[new_seq] = _sb_set_status(m.status[new_seq], CLP_STATUS_BASIC)
                            m.pivot_var[i] = Int32(new_seq)
                        end
                    end
                end
            end
        elseif status == -99
            m.cf.lu.area_factor *= 2.0
        else
            return status
        end
    end
    return m.cf.lu.status
end

"""
    compute_primals!(m::ClpSimplexModel) -> Float64

Recompute basic primal values `x_B = B^{-1} (s_N - A x_N)` from scratch with
iterative refinement, updating `m.solution` and `m.largest_primal_err`.
"""
function compute_primals!(m::ClpSimplexModel)::Float64
    nrows = m.nrows
    ncols = m.ncols
    ntotal = nrows + ncols
    elems = _sm_active_elements(m)
    col_sol = @view m.solution[1:ncols]
    row_sol = @view m.solution[(ncols + 1):ntotal]

    clp_zero_basic_solution!(m.solution, m.pivot_var, nrows)
    fill!(m.work_row, 0.0)
    clp_mat_times_unscaled!(m.mat_starts, nothing, false, m.mat_indices, elems, ncols, -1.0, col_sol, m.work_row)

    nnz = clp_pack_sum_nonzero!(m.r1_val, m.r1_idx, row_sol, m.work_row, nrows)
    nnz = clp_fact_update_column!(m.cf, m.r2_val, m.r2_idx, 0, m.r1_val, m.r1_idx, nnz, false, false)
    clp_scatter_basic_solution!(m.solution, m.pivot_var, m.r1_val, m.r1_idx, nnz, nrows, false)

    # Compute primal residual: work_row = s - A x
    fill!(m.work_row, 0.0)
    clp_mat_times_unscaled!(m.mat_starts, nothing, false, m.mat_indices, elems, ncols, -1.0, col_sol, m.work_row)
    max_err = clp_primal_residual_scale!(m.work_row, row_sol, nrows, 1.0)

    # One iterative refinement pass if residual is noticeable
    if max_err > 1.0e-12 && max_err < 1.0e6
        # Save current r1_val into r3_val
        @inbounds for i in 1:nrows
            m.r3_val[i] = m.r1_val[i]
            m.r1_val[i] = 0.0
        end
        mult = max_err < 1.0e-5 ? 1.0e8 : 1.0
        nnz_r = clp_refine_prepare!(m.r1_val, m.r1_idx, m.work_row, nrows, -mult)
        if nnz_r > 0
            clp_fact_update_column!(m.cf, m.r2_val, m.r2_idx, 0, m.r1_val, m.r1_idx, nnz_r, false, false)
            nnz = clp_refine_combine!(m.r1_val, m.r1_idx, m.r3_val, nrows, 1.0 / mult)
            clp_scatter_basic_solution!(m.solution, m.pivot_var, m.r1_val, m.r1_idx, nnz, nrows, true)
            fill!(m.work_row, 0.0)
            clp_mat_times_unscaled!(m.mat_starts, nothing, false, m.mat_indices, elems, ncols, -1.0, col_sol, m.work_row)
            max_err = clp_primal_residual_scale!(m.work_row, row_sol, nrows, 1.0)
        end
        @inbounds for i in 1:nrows
            m.r3_val[i] = 0.0
        end
    end

    # Clear r1_val
    _dl_clear_unpacked!(m.r1_val, m.r1_idx, nnz)
    @inbounds for i in 1:nrows
        m.r1_val[i] = 0.0
    end
    m.largest_primal_err = max_err
    return max_err
end

"""
    compute_duals!(m::ClpSimplexModel) -> Float64

Recompute dual variables `y = B^{-T} c_B` and reduced costs `dj = c - A^T y` from scratch,
updating `m.dual_work`, `m.dj`, and `m.largest_dual_err`.
"""
function compute_duals!(m::ClpSimplexModel)::Float64
    nrows = m.nrows
    ncols = m.ncols
    ntotal = nrows + ncols
    elems = _sm_active_elements(m)
    col_cost = @view m.cost[1:ncols]
    row_cost = @view m.cost[(ncols + 1):ntotal]
    col_dj = @view m.dj[1:ncols]
    row_dj = @view m.dj[(ncols + 1):ntotal]

    @inbounds for i in 1:nrows
        m.r0_val[i] = 0.0
    end
    nnz = clp_compute_duals_setup!(m.r0_val, m.r0_idx, m.cost, m.pivot_var, nothing, m.status, nrows)
    if nnz > 0
        nnz = clp_fact_update_column_transpose!(m.cf, m.r2_val, m.r2_idx, 0, false, m.r0_val, m.r0_idx, nnz, false)
    end

    # Compute dual residual on basic variables
    ns = clp_compute_duals_collect_structurals!(m.c0_idx, m.pivot_var, nrows, ncols)
    if ns > 0
        clp_mat_subset_tt!(
            m.mat_starts, nothing, false, m.mat_indices, elems,
            m.c0_idx, ns, m.r0_val, nothing, nothing, m.c0_val,
        )
    end
    max_err = clp_compute_duals_residual!(
        m.work_row, m.pivot_var, m.r0_val, row_cost, col_cost, m.c0_val, nothing, nrows, ncols,
    )

    # Finalize duals & reduced costs: col_dj = col_cost - A^T y, row_dj = row_cost + y
    clp_compute_duals_finalize_prep!(
        m.dual_work, row_dj, col_dj, nothing, m.r0_val, row_cost, col_cost, m.status, nrows, ncols,
    )
    clp_mat_transpose_times_unscaled!(
        m.mat_starts, nothing, false, m.mat_indices, elems, ncols, -1.0, m.dual_work, col_dj,
    )
    # Zero basic reduced costs
    @inbounds for i in 1:nrows
        m.dj[Int(m.pivot_var[i])] = 0.0
    end

    # Clear workspaces
    @inbounds for i in 1:nrows
        m.r0_val[i] = 0.0
    end
    @inbounds for j in 1:ns
        m.c0_val[j] = 0.0
    end
    m.largest_dual_err = max_err
    return max_err
end

"""
    guts_of_solution!(m::ClpSimplexModel; do_primals=true, do_duals=true, check_infeas=true, algorithm=1)

Recompute primal/dual solutions from the current factorization and evaluate infeasibilities
and objective value using `clp_check_both_solutions`.
"""
function guts_of_solution!(
    m::ClpSimplexModel;
    do_primals::Bool = true,
    do_duals::Bool = true,
    check_infeas::Bool = true,
    algorithm::Int = 1,
)::Nothing
    if do_primals
        compute_primals!(m)
    end
    if do_duals
        compute_duals!(m)
    end
    if check_infeas
        ntotal = m.nrows + m.ncols
        obj_v, sum_p, _, num_p, sum_d, _, num_d, _, _, _, _ = clp_check_both_solutions(
            m.solution, m.cost, m.lower, m.upper, m.dj, m.status,
            ntotal, m.primal_tol, m.dual_tol,
            m.largest_primal_err, m.largest_dual_err,
            algorithm, m.more_special_options, 0, m.number_iterations,
            0.0, m.obj_scale, m.rhs_scale,
        )
        m.objective_value = obj_v
        m.sum_primal_infeas = sum_p
        m.num_primal_infeas = num_p
        m.sum_dual_infeas = sum_d
        m.num_dual_infeas = num_d
    end
    return nothing
end
