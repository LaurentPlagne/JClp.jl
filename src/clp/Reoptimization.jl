struct _BasisSnapshot
    status::Vector{UInt8}
    columns::Vector{Float64}
    rows::Vector{Float64}
end

"""
    BasisCache()

Keep the last four optimal solutions and basis statuses for bound changes.
Pass the same cache to `reoptimize!` on every solve of one model.
"""
mutable struct BasisCache
    matrix::Union{Nothing,Vector{Float64}}
    signature::UInt64
    history::Vector{_BasisSnapshot}
end

BasisCache() = BasisCache(nothing, UInt64(0), _BasisSnapshot[])

function _basis_cache_signature(m::ClpSimplexModel)
    h = _sm_hash_mix(_sm_matrix_hash(m), reinterpret(UInt64, m.obj_sense))
    for c in m.objective
        h = _sm_hash_mix(h, reinterpret(UInt64, c))
    end
    return h
end

function _bound_violation(activity, lower, upper, tol)
    violation = 0.0
    @inbounds for i in eachindex(activity, lower, upper)
        violation += max(0.0, lower[i] - activity[i] - tol, activity[i] - upper[i] - tol)
    end
    return violation
end

function _basis_score(m::ClpSimplexModel, columns, rows)
    return _bound_violation(columns, m.col_lower, m.col_upper, m.primal_tol) +
           _bound_violation(rows, m.row_lower, m.row_upper, m.primal_tol)
end

function _select_cached_basis!(m::ClpSimplexModel, cache::BasisCache)
    # Score the saved primal points against the new bounds. Only the chosen
    # basis incurs a refactorization. A tie keeps the most recent basis.
    best_score = _basis_score(m, m.col_activity, m.row_activity)
    best = nothing
    for snapshot in Iterators.reverse(cache.history)
        best_score <= m.primal_tol && break
        snapshot.status == m.status && continue
        score = _basis_score(m, snapshot.columns, snapshot.rows)
        if score + m.primal_tol < best_score
            best_score = score
            best = snapshot
        end
    end
    if best !== nothing
        copyto!(m.status, best.status)
        copyto!(m.col_activity, best.columns)
        copyto!(m.row_activity, best.rows)
        m.fact_hash = UInt64(0)
        m.drs_weights_stale = true
    end
    return nothing
end

function _remember_basis!(cache::BasisCache, m::ClpSimplexModel)
    if length(cache.history) == 4
        slot = popfirst!(cache.history)
        copyto!(slot.status, m.status)
        copyto!(slot.columns, m.col_activity)
        copyto!(slot.rows, m.row_activity)
    else
        slot = _BasisSnapshot(copy(m.status), copy(m.col_activity), copy(m.row_activity))
    end
    push!(cache.history, slot)
    return nothing
end

"""
    reoptimize!(m; warm_start, cold_presolve=false, defer_warm_perturbation=false,
                basis_cache=nothing, max_iterations=m.max_iterations) -> Int

Solve one LP in a sequence. Pass `warm_start=true` to reuse the previous basis.
`cold_presolve=true` calls `initial_solve!` on a cold start.
`defer_warm_perturbation=true` delays automatic cost perturbation on warm starts.
With a `BasisCache()`, warm starts can reuse an older basis when its saved primal
point violates the new bounds less. A cold start, unsuccessful solve, model
reload, or change of matrix or objective starts a new history.
"""
function reoptimize!(
    m::ClpSimplexModel;
    warm_start::Bool,
    cold_presolve::Bool = false,
    defer_warm_perturbation::Bool = false,
    basis_cache::Union{Nothing,BasisCache} = nothing,
    max_iterations::Int = m.max_iterations,
)::Int
    if basis_cache !== nothing
        signature = _basis_cache_signature(m)
        if warm_start && m.problem_status == CLP_STATUS_OPTIMAL &&
           basis_cache.matrix === m.mat_elements && basis_cache.signature == signature
            _select_cached_basis!(m, basis_cache)
        else
            empty!(basis_cache.history)
        end
        basis_cache.matrix = m.mat_elements
        basis_cache.signature = signature
    end

    # Primal cleanup in dual! can set max_iterations to its remaining budget.
    # The next point in the sequence gets the model's configured limit again.
    configured_limit = m.max_iterations
    status = try
        if !warm_start && cold_presolve && _sm_bounds_feasible(m)
            initial_solve!(m; max_iterations)
        else
            dual!(m; warm_start, max_iterations, defer_warm_perturbation)
        end
    finally
        m.max_iterations = configured_limit
    end

    if basis_cache !== nothing
        if status == CLP_STATUS_OPTIMAL
            # Cold simplex may have changed the matrix scaling.
            if !warm_start
                basis_cache.signature = _basis_cache_signature(m)
            end
            _remember_basis!(basis_cache, m)
        else
            empty!(basis_cache.history)
        end
    end
    return status
end
