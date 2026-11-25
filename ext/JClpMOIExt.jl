module JClpMOIExt

using JClp
import MathOptInterface as MOI

const _MOI_INF = 1.0e30

const _SCALAR_SETS = Union{
    MOI.LessThan{Float64},
    MOI.GreaterThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64},
}

const _BOUND_LT = UInt8(1)
const _BOUND_GT = UInt8(2)
const _BOUND_EQ = UInt8(4)
const _BOUND_IV = UInt8(8)

@inline _set_flag(::Type{MOI.LessThan{Float64}}) = _BOUND_LT
@inline _set_flag(::Type{MOI.GreaterThan{Float64}}) = _BOUND_GT
@inline _set_flag(::Type{MOI.EqualTo{Float64}}) = _BOUND_EQ
@inline _set_flag(::Type{MOI.Interval{Float64}}) = _BOUND_IV

@inline _row_tag(::Type{MOI.LessThan{Float64}}) = :lt
@inline _row_tag(::Type{MOI.GreaterThan{Float64}}) = :gt
@inline _row_tag(::Type{MOI.EqualTo{Float64}}) = :eq
@inline _row_tag(::Type{MOI.Interval{Float64}}) = :iv

@inline _to_clp_lo(v::Float64)::Float64 = v <= -_MOI_INF ? -_MOI_INF : v
@inline _to_clp_hi(v::Float64)::Float64 = v >= _MOI_INF ? _MOI_INF : v

"""
    MOIOptimizer <: MOI.AbstractOptimizer

`MathOptInterface` optimizer wrapping `JClp.ClpSimplexModel`.
Construct via `JClp.Optimizer(; kwargs...)`.
"""
mutable struct MOIOptimizer <: MOI.AbstractOptimizer
    inner::JClp.ClpSimplexModel
    silent::Bool
    presolve::Bool
    max_iterations::Int
    algorithm::String
    sense::MOI.OptimizationSense
    obj_constant::Float64
    has_objective::Bool
    has_solved::Bool
    solve_time::Float64
    name::String
    col_bound_flags::Vector{UInt8}
    row_types::Vector{Symbol}
    row_constants::Vector{Float64}
end

function MOIOptimizer(; kwargs...)
    opt = MOIOptimizer(
        JClp.ClpSimplexModel(),
        false,
        true,
        Int(typemax(Int32)),
        "initial",
        MOI.FEASIBILITY_SENSE,
        0.0,
        false,
        false,
        0.0,
        "",
        UInt8[],
        Symbol[],
        Float64[],
    )
    for (k, v) in kwargs
        if k === :silent
            MOI.set(opt, MOI.Silent(), v)
        else
            MOI.set(opt, MOI.RawOptimizerAttribute(string(k)), v)
        end
    end
    return opt
end

JClp.Optimizer(; kwargs...) = MOIOptimizer(; kwargs...)

function MOI.is_empty(opt::MOIOptimizer)::Bool
    return opt.inner.nrows == 0 &&
           opt.inner.ncols == 0 &&
           opt.sense == MOI.FEASIBILITY_SENSE &&
           !opt.has_objective &&
           !opt.has_solved &&
           isempty(opt.name) &&
           isempty(opt.col_bound_flags) &&
           isempty(opt.row_types)
end

function MOI.empty!(opt::MOIOptimizer)::Nothing
    opt.inner = JClp.ClpSimplexModel()
    opt.inner.max_iterations = opt.max_iterations
    opt.sense = MOI.FEASIBILITY_SENSE
    opt.obj_constant = 0.0
    opt.has_objective = false
    opt.has_solved = false
    opt.solve_time = 0.0
    opt.name = ""
    empty!(opt.col_bound_flags)
    empty!(opt.row_types)
    empty!(opt.row_constants)
    return nothing
end

MOI.supports_incremental_interface(::MOIOptimizer)::Bool = false

# ==============================================================================
# Supported constraints & attributes
# ==============================================================================

function MOI.supports_constraint(
    ::MOIOptimizer,
    ::Type{MOI.VariableIndex},
    ::Type{<:_SCALAR_SETS},
)::Bool
    return true
end

function MOI.supports_constraint(
    ::MOIOptimizer,
    ::Type{MOI.ScalarAffineFunction{Float64}},
    ::Type{<:_SCALAR_SETS},
)::Bool
    return true
end

MOI.supports(::MOIOptimizer, ::MOI.SolverName)::Bool = true
MOI.supports(::MOIOptimizer, ::MOI.SolverVersion)::Bool = true
MOI.supports(::MOIOptimizer, ::MOI.Silent)::Bool = true
MOI.supports(::MOIOptimizer, ::MOI.Name)::Bool = true
MOI.supports(::MOIOptimizer, ::MOI.ObjectiveSense)::Bool = true
MOI.supports(::MOIOptimizer, ::MOI.ObjectiveFunction{MOI.VariableIndex})::Bool = true
MOI.supports(::MOIOptimizer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}})::Bool = true

function MOI.supports(
    ::MOIOptimizer,
    attr::MOI.RawOptimizerAttribute,
)::Bool
    return attr.name == "presolve" ||
           attr.name == "max_iterations" ||
           attr.name == "algorithm"
end

# ==============================================================================
# Optimizer & model attribute getters / setters
# ==============================================================================

MOI.get(::MOIOptimizer, ::MOI.SolverName)::String = "JClp"
MOI.get(::MOIOptimizer, ::MOI.SolverVersion)::String = string(pkgversion(JClp))

MOI.get(opt::MOIOptimizer, ::MOI.Silent)::Bool = opt.silent
function MOI.set(opt::MOIOptimizer, ::MOI.Silent, val::Bool)::Nothing
    opt.silent = val
    return nothing
end

MOI.get(opt::MOIOptimizer, ::MOI.Name)::String = opt.name
function MOI.set(opt::MOIOptimizer, ::MOI.Name, val::String)::Nothing
    opt.name = val
    opt.inner.name = val
    return nothing
end

MOI.get(opt::MOIOptimizer, ::MOI.ObjectiveSense)::MOI.OptimizationSense = opt.sense
MOI.get(opt::MOIOptimizer, ::MOI.NumberOfVariables)::Int64 = Int64(opt.inner.ncols)

function MOI.get(opt::MOIOptimizer, attr::MOI.RawOptimizerAttribute)
    if attr.name == "presolve"
        return opt.presolve
    elseif attr.name == "max_iterations"
        return opt.max_iterations
    elseif attr.name == "algorithm"
        return opt.algorithm
    end
    throw(MOI.UnsupportedAttribute(attr))
end

function MOI.set(opt::MOIOptimizer, attr::MOI.RawOptimizerAttribute, val)::Nothing
    if attr.name == "presolve"
        val isa Bool || throw(MOI.SetAttributeNotAllowed(attr, "Expected Bool for \"presolve\"."))
        opt.presolve = val
        return nothing
    elseif attr.name == "max_iterations"
        (val isa Integer && val >= 0) || throw(MOI.SetAttributeNotAllowed(attr, "Expected non-negative Integer for \"max_iterations\"."))
        opt.max_iterations = Int(val)
        opt.inner.max_iterations = opt.max_iterations
        return nothing
    elseif attr.name == "algorithm"
        if !(val isa AbstractString)
            throw(MOI.SetAttributeNotAllowed(attr, "Expected String for \"algorithm\"."))
        end
        alg = String(val)
        if alg != "initial" && alg != "dual" && alg != "primal"
            throw(MOI.SetAttributeNotAllowed(attr, "Expected algorithm in (\"initial\", \"dual\", \"primal\"), got \"$alg\"."))
        end
        opt.algorithm = alg
        return nothing
    end
    throw(MOI.UnsupportedAttribute(attr))
end

# ==============================================================================
# Index validity
# ==============================================================================

function MOI.is_valid(opt::MOIOptimizer, vi::MOI.VariableIndex)::Bool
    return 1 <= vi.value <= opt.inner.ncols
end

function MOI.is_valid(
    opt::MOIOptimizer,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,S},
)::Bool where {S<:_SCALAR_SETS}
    col = ci.value
    return 1 <= col <= opt.inner.ncols &&
           (opt.col_bound_flags[col] & _set_flag(S)) != UInt8(0)
end

function MOI.is_valid(
    opt::MOIOptimizer,
    ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},S},
)::Bool where {S<:_SCALAR_SETS}
    row = ci.value
    return 1 <= row <= opt.inner.nrows && opt.row_types[row] === _row_tag(S)
end

MOI.is_valid(::MOIOptimizer, ::MOI.ConstraintIndex)::Bool = false

# ==============================================================================
# Copy-based model construction
# ==============================================================================

function _validate_copy_attributes(dest::MOIOptimizer, src::MOI.ModelLike, con_types)
    for attr in MOI.get(src, MOI.ListOfModelAttributesSet())
        if attr isa MOI.ObjectiveSense ||
           attr isa MOI.ObjectiveFunction{MOI.VariableIndex} ||
           attr isa MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}} ||
           attr isa MOI.Name
            continue
        elseif !MOI.supports(dest, attr)
            throw(MOI.UnsupportedAttribute(attr))
        end
    end
    for attr in MOI.get(src, MOI.ListOfVariableAttributesSet())
        if !MOI.supports(dest, attr, MOI.VariableIndex)
            if attr == MOI.VariableName() || attr == MOI.VariablePrimalStart()
                continue
            end
            throw(MOI.UnsupportedAttribute(attr))
        end
    end
    for (F, S) in con_types
        for attr in MOI.get(src, MOI.ListOfConstraintAttributesSet{F,S}())
            if !MOI.supports(dest, attr, MOI.ConstraintIndex{F,S})
                if attr == MOI.ConstraintName() ||
                   attr == MOI.ConstraintPrimalStart() ||
                   attr == MOI.ConstraintDualStart()
                    continue
                end
                throw(MOI.UnsupportedAttribute(attr))
            end
        end
    end
    return nothing
end

function _copy_var_bounds!(
    col_lower::Vector{Float64},
    col_upper::Vector{Float64},
    col_bound_flags::Vector{UInt8},
    index_map::MOI.Utilities.IndexMap,
    src::MOI.ModelLike,
    ::Type{S},
) where {S<:_SCALAR_SETS}
    flag = _set_flag(S)
    for ci_src in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VariableIndex,S}())
        v_src = MOI.get(src, MOI.ConstraintFunction(), ci_src)::MOI.VariableIndex
        set = MOI.get(src, MOI.ConstraintSet(), ci_src)::S
        col = index_map[v_src].value
        index_map[ci_src] = MOI.ConstraintIndex{MOI.VariableIndex,S}(col)
        col_bound_flags[col] |= flag
        if S === MOI.LessThan{Float64}
            col_upper[col] = min(col_upper[col], _to_clp_hi(Float64(set.upper)))
        elseif S === MOI.GreaterThan{Float64}
            col_lower[col] = max(col_lower[col], _to_clp_lo(Float64(set.lower)))
        elseif S === MOI.EqualTo{Float64}
            val = Float64(set.value)
            col_lower[col] = max(col_lower[col], _to_clp_lo(val))
            col_upper[col] = min(col_upper[col], _to_clp_hi(val))
        else # MOI.Interval{Float64}
            col_lower[col] = max(col_lower[col], _to_clp_lo(Float64(set.lower)))
            col_upper[col] = min(col_upper[col], _to_clp_hi(Float64(set.upper)))
        end
    end
    return nothing
end

function _copy_affine_rows!(
    row_idx_ref::Ref{Int},
    row_lower::Vector{Float64},
    row_upper::Vector{Float64},
    row_types::Vector{Symbol},
    row_constants::Vector{Float64},
    trip_col::Vector{Int32},
    trip_row::Vector{Int32},
    trip_val::Vector{Float64},
    col_counts::Vector{Int32},
    index_map::MOI.Utilities.IndexMap,
    src::MOI.ModelLike,
    ::Type{S},
) where {S<:_SCALAR_SETS}
    tag = _row_tag(S)
    F = MOI.ScalarAffineFunction{Float64}
    for ci_src in MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
        row = row_idx_ref[] + 1
        row_idx_ref[] = row
        index_map[ci_src] = MOI.ConstraintIndex{F,S}(row)

        f = MOI.get(src, MOI.ConstraintFunction(), ci_src)::F
        set = MOI.get(src, MOI.ConstraintSet(), ci_src)::S
        c_f = Float64(f.constant)
        row_types[row] = tag
        row_constants[row] = c_f

        if S === MOI.LessThan{Float64}
            row_lower[row] = -_MOI_INF
            row_upper[row] = _to_clp_hi(Float64(set.upper) - c_f)
        elseif S === MOI.GreaterThan{Float64}
            row_lower[row] = _to_clp_lo(Float64(set.lower) - c_f)
            row_upper[row] = _MOI_INF
        elseif S === MOI.EqualTo{Float64}
            rhs = Float64(set.value) - c_f
            row_lower[row] = rhs
            row_upper[row] = rhs
        else # MOI.Interval{Float64}
            row_lower[row] = _to_clp_lo(Float64(set.lower) - c_f)
            row_upper[row] = _to_clp_hi(Float64(set.upper) - c_f)
        end

        row32 = Int32(row)
        for term in f.terms
            v = Float64(term.coefficient)
            v == 0.0 && continue
            col = index_map[term.variable].value
            push!(trip_col, Int32(col))
            push!(trip_row, row32)
            push!(trip_val, v)
            col_counts[col] += Int32(1)
        end
    end
    return nothing
end

function _assemble_csc(
    ncols::Int,
    col_counts::Vector{Int32},
    trip_col::Vector{Int32},
    trip_row::Vector{Int32},
    trip_val::Vector{Float64},
)
    col_starts = Vector{Int32}(undef, ncols + 1)
    pos = Int32(1)
    @inbounds for j in 1:ncols
        col_starts[j] = pos
        pos += col_counts[j]
        col_counts[j] = col_starts[j]
    end
    col_starts[ncols + 1] = pos

    nnz_raw = length(trip_col)
    raw_indices = Vector{Int32}(undef, nnz_raw)
    raw_elements = Vector{Float64}(undef, nnz_raw)
    @inbounds for t in 1:nnz_raw
        c = Int(trip_col[t])
        p = Int(col_counts[c])
        col_counts[c] = Int32(p + 1)
        raw_indices[p] = trip_row[t]
        raw_elements[p] = trip_val[t]
    end

    # Since rows were visited in strictly increasing order 1:nrows, entries
    # within each column bucket are already non-decreasing in row index, so
    # duplicate terms in the same row are adjacent and merge in O(nnz).
    col_indices = Vector{Int32}(undef, nnz_raw)
    col_elements = Vector{Float64}(undef, nnz_raw)
    out_pos = 1
    next_raw_start = Int(col_starts[1])
    @inbounds for j in 1:ncols
        raw_start = next_raw_start
        next_raw_start = Int(col_starts[j + 1])
        col_starts[j] = Int32(out_pos)
        col_out_start = out_pos
        last_row = Int32(0)
        for k in raw_start:(next_raw_start - 1)
            r = raw_indices[k]
            v = raw_elements[k]
            if r == last_row && out_pos > col_out_start
                col_elements[out_pos - 1] += v
            else
                col_indices[out_pos] = r
                col_elements[out_pos] = v
                last_row = r
                out_pos += 1
            end
        end
        # Drop exact zeros resulting from cancelling duplicate terms
        w = col_out_start
        for p in col_out_start:(out_pos - 1)
            if col_elements[p] != 0.0
                if w != p
                    col_indices[w] = col_indices[p]
                    col_elements[w] = col_elements[p]
                end
                w += 1
            end
        end
        out_pos = w
    end
    col_starts[ncols + 1] = Int32(out_pos)
    resize!(col_indices, out_pos - 1)
    resize!(col_elements, out_pos - 1)
    return col_starts, col_indices, col_elements
end

function MOI.copy_to(dest::MOIOptimizer, src::MOI.ModelLike)::MOI.Utilities.IndexMap
    MOI.empty!(dest)
    index_map = MOI.Utilities.IndexMap()

    con_types = MOI.get(src, MOI.ListOfConstraintTypesPresent())
    for (F, S) in con_types
        if !MOI.supports_constraint(dest, F, S)
            throw(MOI.UnsupportedConstraint{F,S}())
        end
    end
    _validate_copy_attributes(dest, src, con_types)

    vis_src = MOI.get(src, MOI.ListOfVariableIndices())
    ncols = length(vis_src)
    for (col, v_src) in enumerate(vis_src)
        index_map[v_src] = MOI.VariableIndex(col)
    end

    col_lower = fill(-_MOI_INF, ncols)
    col_upper = fill(_MOI_INF, ncols)
    col_bound_flags = zeros(UInt8, ncols)

    nrows = 0
    for (F, S) in con_types
        if F === MOI.VariableIndex
            _copy_var_bounds!(col_lower, col_upper, col_bound_flags, index_map, src, S)
        else
            nrows += MOI.get(src, MOI.NumberOfConstraints{F,S}())
        end
    end

    row_lower = Vector{Float64}(undef, nrows)
    row_upper = Vector{Float64}(undef, nrows)
    row_types = Vector{Symbol}(undef, nrows)
    row_constants = Vector{Float64}(undef, nrows)
    trip_col = Int32[]
    trip_row = Int32[]
    trip_val = Float64[]
    col_counts = zeros(Int32, ncols)

    row_idx_ref = Ref(0)
    for (F, S) in con_types
        if F === MOI.ScalarAffineFunction{Float64}
            _copy_affine_rows!(
                row_idx_ref,
                row_lower,
                row_upper,
                row_types,
                row_constants,
                trip_col,
                trip_row,
                trip_val,
                col_counts,
                index_map,
                src,
                S,
            )
        end
    end

    col_starts, col_indices, col_elements = _assemble_csc(
        ncols, col_counts, trip_col, trip_row, trip_val,
    )

    model_attrs = MOI.get(src, MOI.ListOfModelAttributesSet())
    sense = MOI.FEASIBILITY_SENSE
    if MOI.ObjectiveSense() in model_attrs
        sense = MOI.get(src, MOI.ObjectiveSense())
    end
    dest.sense = sense

    if MOI.Name() in model_attrs
        dest.name = String(MOI.get(src, MOI.Name()))
    end

    objective = zeros(Float64, ncols)
    obj_constant = 0.0
    has_obj = false
    clp_sense = sense == MOI.MAX_SENSE ? -1.0 : 1.0

    if sense != MOI.FEASIBILITY_SENSE
        for attr in model_attrs
            if attr isa MOI.ObjectiveFunction{MOI.VariableIndex}
                v_src = MOI.get(src, attr)::MOI.VariableIndex
                objective[index_map[v_src].value] = 1.0
                has_obj = true
            elseif attr isa MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}
                f = MOI.get(src, attr)::MOI.ScalarAffineFunction{Float64}
                obj_constant = Float64(f.constant)
                for term in f.terms
                    col = index_map[term.variable].value
                    objective[col] += Float64(term.coefficient)
                end
                has_obj = true
            end
        end
    end

    dest.obj_constant = obj_constant
    dest.has_objective = has_obj
    dest.col_bound_flags = col_bound_flags
    dest.row_types = row_types
    dest.row_constants = row_constants

    JClp.load_problem!(
        dest.inner,
        nrows,
        ncols,
        col_starts,
        col_indices,
        col_elements,
        col_lower,
        col_upper,
        objective,
        row_lower,
        row_upper;
        obj_sense = clp_sense,
        obj_offset = -obj_constant,
        name = dest.name,
    )
    dest.inner.max_iterations = dest.max_iterations

    return index_map
end

# ==============================================================================
# Solve
# ==============================================================================

function MOI.optimize!(opt::MOIOptimizer)::Nothing
    t0 = time_ns()
    m = opt.inner
    m.max_iterations = opt.max_iterations
    if opt.algorithm == "initial"
        JClp.initial_solve!(m; do_presolve = opt.presolve, max_iterations = opt.max_iterations)
    elseif opt.algorithm == "dual"
        if opt.presolve && m.nrows > 0 && m.ncols > 0
            JClp.initial_solve!(m; do_presolve = true, max_iterations = opt.max_iterations)
        else
            JClp.dual!(m; max_iterations = opt.max_iterations)
        end
    elseif opt.algorithm == "primal"
        if opt.presolve && m.nrows > 0 && m.ncols > 0
            ps = JClp.presolve(m)
            if ps.status != 0
                m.problem_status = ps.status
                m.number_iterations = 0
            else
                JClp.primal!(ps.reduced; max_iterations = opt.max_iterations)
                JClp.postsolve!(m, ps)
            end
        else
            JClp.primal!(m; max_iterations = opt.max_iterations)
        end
    else
        throw(ArgumentError("Unsupported algorithm \"$(opt.algorithm)\"."))
    end
    opt.solve_time = (time_ns() - t0) * 1.0e-9
    opt.has_solved = true
    return nothing
end

# ==============================================================================
# Solution status & attributes
# ==============================================================================

function MOI.get(opt::MOIOptimizer, ::MOI.TerminationStatus)::MOI.TerminationStatusCode
    opt.has_solved || return MOI.OPTIMIZE_NOT_CALLED
    st = opt.inner.problem_status
    if st == JClp.CLP_STATUS_OPTIMAL
        return MOI.OPTIMAL
    elseif st == JClp.CLP_STATUS_PRIMAL_INFEAS
        return MOI.INFEASIBLE
    elseif st == JClp.CLP_STATUS_DUAL_INFEAS
        return MOI.DUAL_INFEASIBLE
    elseif st == JClp.CLP_STATUS_MAX_ITERATIONS
        return MOI.ITERATION_LIMIT
    elseif st == JClp.CLP_STATUS_ERROR
        return MOI.NUMERICAL_ERROR
    end
    return MOI.OTHER_ERROR
end

function MOI.get(opt::MOIOptimizer, ::MOI.RawStatusString)::String
    opt.has_solved || return "Optimize not called"
    st = opt.inner.problem_status
    if st == JClp.CLP_STATUS_OPTIMAL
        return "Optimal"
    elseif st == JClp.CLP_STATUS_PRIMAL_INFEAS
        return "Primal infeasible"
    elseif st == JClp.CLP_STATUS_DUAL_INFEAS
        return "Dual infeasible"
    elseif st == JClp.CLP_STATUS_MAX_ITERATIONS
        return "Iteration limit reached"
    elseif st == JClp.CLP_STATUS_ERROR
        return "Numerical error"
    end
    return "Unknown status ($st)"
end

function MOI.get(opt::MOIOptimizer, ::MOI.ResultCount)::Int64
    return (opt.has_solved && opt.inner.problem_status == JClp.CLP_STATUS_OPTIMAL) ? Int64(1) : Int64(0)
end

function MOI.get(opt::MOIOptimizer, attr::MOI.PrimalStatus)::MOI.ResultStatusCode
    if attr.result_index == 1 && opt.has_solved && opt.inner.problem_status == JClp.CLP_STATUS_OPTIMAL
        return MOI.FEASIBLE_POINT
    end
    return MOI.NO_SOLUTION
end

function MOI.get(opt::MOIOptimizer, attr::MOI.DualStatus)::MOI.ResultStatusCode
    if attr.result_index == 1 && opt.has_solved && opt.inner.problem_status == JClp.CLP_STATUS_OPTIMAL
        return MOI.FEASIBLE_POINT
    end
    return MOI.NO_SOLUTION
end

MOI.get(opt::MOIOptimizer, ::MOI.SimplexIterations)::Int64 = Int64(opt.inner.number_iterations)
MOI.get(opt::MOIOptimizer, ::MOI.SolveTimeSec)::Float64 = opt.solve_time

function MOI.get(opt::MOIOptimizer, attr::MOI.ObjectiveValue)::Float64
    MOI.check_result_index_bounds(opt, attr)
    opt.sense == MOI.FEASIBILITY_SENSE && return 0.0
    return opt.inner.objective_value
end

function _compute_dual_objective(m::JClp.ClpSimplexModel, obj_constant::Float64)::Float64
    s = m.obj_sense
    val = obj_constant
    @inbounds for i in 1:m.nrows
        yi = m.dual_row_sol[i]
        y_min = s * yi
        rlo = m.row_lower[i]
        rhi = m.row_upper[i]
        st = m.status[m.ncols + i] & JClp.CLP_STATUS_MASK
        if st == JClp.CLP_STATUS_AT_LOWER || st == JClp.CLP_STATUS_IS_FIXED
            if rlo > -1.0e20
                val += yi * rlo
            end
        elseif st == JClp.CLP_STATUS_AT_UPPER
            if rhi < 1.0e20
                val += yi * rhi
            end
        elseif abs(y_min) > m.dual_tol
            if y_min > 0.0 && rlo > -1.0e20
                val += yi * rlo
            elseif y_min < 0.0 && rhi < 1.0e20
                val += yi * rhi
            end
        end
    end
    @inbounds for j in 1:m.ncols
        rcj = m.reduced_cost[j]
        rc_min = s * rcj
        clo = m.col_lower[j]
        chi = m.col_upper[j]
        st = m.status[j] & JClp.CLP_STATUS_MASK
        if st == JClp.CLP_STATUS_AT_LOWER || st == JClp.CLP_STATUS_IS_FIXED
            if clo > -1.0e20
                val += rcj * clo
            end
        elseif st == JClp.CLP_STATUS_AT_UPPER
            if chi < 1.0e20
                val += rcj * chi
            end
        elseif abs(rc_min) > m.dual_tol
            if rc_min > 0.0 && clo > -1.0e20
                val += rcj * clo
            elseif rc_min < 0.0 && chi < 1.0e20
                val += rcj * chi
            end
        end
    end
    return val
end

function MOI.get(opt::MOIOptimizer, attr::MOI.DualObjectiveValue)::Float64
    MOI.check_result_index_bounds(opt, attr)
    opt.sense == MOI.FEASIBILITY_SENSE && return 0.0
    return _compute_dual_objective(opt.inner, opt.obj_constant)
end

function MOI.get(opt::MOIOptimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)::Float64
    MOI.check_result_index_bounds(opt, attr)
    MOI.throw_if_not_valid(opt, vi)
    return opt.inner.col_activity[vi.value]
end

function MOI.get(
    opt::MOIOptimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,S},
)::Float64 where {S<:_SCALAR_SETS}
    MOI.check_result_index_bounds(opt, attr)
    MOI.throw_if_not_valid(opt, ci)
    return opt.inner.col_activity[ci.value]
end

function MOI.get(
    opt::MOIOptimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},S},
)::Float64 where {S<:_SCALAR_SETS}
    MOI.check_result_index_bounds(opt, attr)
    MOI.throw_if_not_valid(opt, ci)
    row = ci.value
    return opt.inner.row_activity[row] + opt.row_constants[row]
end

@inline function _signed_set_dual(d_moi::Float64, ::Type{MOI.LessThan{Float64}})::Float64
    return min(0.0, d_moi)
end
@inline function _signed_set_dual(d_moi::Float64, ::Type{MOI.GreaterThan{Float64}})::Float64
    return max(0.0, d_moi)
end
@inline function _signed_set_dual(
    d_moi::Float64,
    ::Type{<:Union{MOI.EqualTo{Float64},MOI.Interval{Float64}}},
)::Float64
    return d_moi
end

function MOI.get(
    opt::MOIOptimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,S},
)::Float64 where {S<:_SCALAR_SETS}
    MOI.check_result_index_bounds(opt, attr)
    MOI.throw_if_not_valid(opt, ci)
    opt.sense == MOI.FEASIBILITY_SENSE && return 0.0
    m = opt.inner
    rc_moi = m.obj_sense * m.reduced_cost[ci.value]
    return _signed_set_dual(rc_moi, S)
end

function MOI.get(
    opt::MOIOptimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},S},
)::Float64 where {S<:_SCALAR_SETS}
    MOI.check_result_index_bounds(opt, attr)
    MOI.throw_if_not_valid(opt, ci)
    opt.sense == MOI.FEASIBILITY_SENSE && return 0.0
    m = opt.inner
    y_moi = m.obj_sense * m.dual_row_sol[ci.value]
    return _signed_set_dual(y_moi, S)
end

function MOI.get(
    opt::MOIOptimizer,
    attr::MOI.VariableBasisStatus,
    vi::MOI.VariableIndex,
)::MOI.BasisStatusCode
    MOI.check_result_index_bounds(opt, attr)
    MOI.throw_if_not_valid(opt, vi)
    st = opt.inner.status[vi.value] & JClp.CLP_STATUS_MASK
    if st == JClp.CLP_STATUS_BASIC
        return MOI.BASIC
    elseif st == JClp.CLP_STATUS_AT_LOWER || st == JClp.CLP_STATUS_IS_FIXED
        return MOI.NONBASIC_AT_LOWER
    elseif st == JClp.CLP_STATUS_AT_UPPER
        return MOI.NONBASIC_AT_UPPER
    end
    return MOI.SUPER_BASIC
end

function MOI.get(
    opt::MOIOptimizer,
    attr::MOI.ConstraintBasisStatus,
    ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},S},
)::MOI.BasisStatusCode where {S<:_SCALAR_SETS}
    MOI.check_result_index_bounds(opt, attr)
    MOI.throw_if_not_valid(opt, ci)
    m = opt.inner
    row = ci.value
    st = m.status[m.ncols + row] & JClp.CLP_STATUS_MASK
    if st == JClp.CLP_STATUS_BASIC
        return MOI.BASIC
    elseif st == JClp.CLP_STATUS_IS_FREE || st == JClp.CLP_STATUS_SUPER_BASIC
        return MOI.SUPER_BASIC
    elseif S === MOI.Interval{Float64}
        if st == JClp.CLP_STATUS_AT_LOWER
            return MOI.NONBASIC_AT_LOWER
        elseif st == JClp.CLP_STATUS_AT_UPPER
            return MOI.NONBASIC_AT_UPPER
        else
            y_min = m.obj_sense * m.dual_row_sol[row]
            return y_min >= 0.0 ? MOI.NONBASIC_AT_LOWER : MOI.NONBASIC_AT_UPPER
        end
    end
    return MOI.NONBASIC
end

end # module JClpMOIExt
