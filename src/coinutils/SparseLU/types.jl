# Pure idiomatic 1-based Julia data structures for Markowitz Sparse LU factorization
# with Forrest-Tomlin updates and hypersparse BTRAN/FTRAN (equivalent to CoinFactorization).
#
# Design principles:
#   - 100% 1-based Julia indexing (`1:nrows`, `1:nnz`), zero Ptr / unsafe_load / unsafe_store!.
#   - All buffers are concrete `Vector{T}` owned by `SparseLU` and pre-allocated/reused across
#     factorizations for zero-allocation hot paths (`@allocated == 0` on BTRAN/FTRAN/replace).
#   - Compatible with `JuliaC --trim=safe` (all fields concrete, no Union/Any dispatch).

const LU_TINY_ELEMENT = 1.0e-50
const LU_NO_LINK      = Int32(0)
const LU_REMOVED      = Int32(-1)

# Hardware FMA matching Apple arm64 clang++ -O3 contraction of `c += a * b` / `c -= a * b`
@inline _lu_fmadd(a::Float64, b::Float64, c::Float64)::Float64 = fma(a, b, c)
@inline _lu_fmsub(a::Float64, b::Float64, c::Float64)::Float64 = fma(-a, b, c)

# 8-row bitmask helpers (1-based row `r` -> byte `((r-1) >> 3) + 1`, bit `1 << ((r-1) & 7)`)
@inline _mark_byte(r::Int)::Int = ((r - 1) >> 3) + 1
@inline _mark_bit(r::Int)::UInt8 = UInt8(1) << ((r - 1) & 7)

@inline function _set_mark!(mark::Vector{UInt8}, r::Int)::Nothing
    @inbounds mark[_mark_byte(r)] |= _mark_bit(r)
    return nothing
end

@inline function _is_marked(mark::Vector{UInt8}, r::Int)::Bool
    @inbounds return (mark[_mark_byte(r)] & _mark_bit(r)) != UInt8(0)
end

@inline function _clear_mark!(mark::Vector{UInt8}, r::Int)::Nothing
    @inbounds mark[_mark_byte(r)] = UInt8(0)
    return nothing
end

"""
    IndexedVector{VT, VI}

Hypersparse accumulator vector pairing a dense value array `values` with a list of
active 1-based non-zero indices `indices[1:nnz]`. Supports O(nnz) scatter/gather/clear.
"""
mutable struct IndexedVector{VT<:AbstractVector{Float64}, VI<:AbstractVector{Int32}}
    values::VT
    indices::VI
    nnz::Int
    packed::Bool
end

function IndexedVector(n::Int)
    return IndexedVector(zeros(Float64, n), Vector{Int32}(undef, n), 0, false)
end

@inline function clear!(v::IndexedVector)::Nothing
    vals = v.values
    idx = v.indices
    if v.packed
        @inbounds for i in 1:v.nnz
            vals[i] = 0.0
        end
    else
        @inbounds for i in 1:v.nnz
            vals[Int(idx[i])] = 0.0
        end
    end
    v.nnz = 0
    v.packed = false
    return nothing
end

"""
    SparseLU

Pure 1-based Julia Markowitz sparse LU factorization with:
- Markowitz pivot selection + dense LAPACK Accelerate fallback (`dgetrf_`/`dgetrs_`)
- Forrest-Tomlin rank-1 eta updates (`R` stored by column and optionally by row)
- PFI fallback updates when `do_forrest_tomlin == false`
- Hypersparse DFS / bitmask / dense adaptive BTRAN & FTRAN
"""
mutable struct SparseLU
    # --- Dimensions & counters ---
    nrows::Int
    ncols::Int
    bigger_dim::Int
    nrows_extra::Int
    max_rows_extra::Int
    ncols_extra::Int
    max_cols_extra::Int
    ngood_u::Int
    ngood_l::Int
    nu::Int
    nl::Int
    base_l::Int
    nr::Int
    len_u::Int
    len_l::Int
    len_r::Int
    len_area_u::Int
    len_area_l::Int
    len_area_r::Int
    max_u::Int
    total_elements::Int
    factor_elements::Int
    ndense::Int
    nslacks::Int
    npivots::Int
    max_pivots::Int
    ncompressions::Int
    status::Int

    # --- Control parameters ---
    zero_tol::Float64
    pivot_tol::Float64
    relax_check::Float64
    slack_value::Float64
    area_factor::Float64
    sparse_threshold::Int
    sparse_threshold2::Int
    dense_threshold::Int
    bias_lu::Int
    num_trials::Int
    time_limit::Float64
    do_forrest_tomlin::Bool
    collect_statistics::Bool
    has_row_l::Bool
    has_row_r::Bool
    has_sparse::Bool
    got_rcopy::Bool
    got_ucopy::Bool
    got_lcopy::Bool

    # --- Adaptive hypersparse statistics ---
    nbtran_counts::Int
    btran_count_input::Float64
    btran_count_after_u::Float64
    btran_count_after_r::Float64
    btran_count_after_l::Float64
    btran_avg_after_u::Float64
    btran_avg_after_r::Float64
    btran_avg_after_l::Float64

    nftran_counts::Int
    ftran_count_input::Float64
    ftran_count_after_l::Float64
    ftran_count_after_r::Float64
    ftran_count_after_u::Float64
    ftran_avg_after_l::Float64
    ftran_avg_after_r::Float64
    ftran_avg_after_u::Float64

    # --- U factor (1-based column-wise & row-wise linked storage) ---
    element_u::Vector{Float64}
    index_row_u::Vector{Int32}
    start_col_u::Vector{Int32}       # length >= max_cols_extra + 2 (sentinel at max_cols_extra + 1)
    len_col_u::Vector{Int32}         # length >= max_cols_extra
    len_col_plus::Vector{Int32}      # extra entries below start_col_u[j] during factor/cleanup
    start_row_u::Vector{Int32}       # length >= max_rows_extra + 2
    len_row_u::Vector{Int32}         # length >= max_rows_extra
    index_col_u::Vector{Int32}       # length >= len_area_u
    row_to_col_u::Vector{Int32}      # 1-based offset into element_u

    # --- L factor (1-based column-wise & row-wise for BTRAN) ---
    element_l::Vector{Float64}
    index_row_l::Vector{Int32}
    start_col_l::Vector{Int32}       # length >= nrows + 2
    element_row_l::Vector{Float64}
    index_col_l::Vector{Int32}
    start_row_l::Vector{Int32}       # length >= nrows + 2

    # --- R factor (Forrest-Tomlin etas: column-wise + optional row-wise) ---
    element_r::Vector{Float64}
    index_row_r::Vector{Int32}
    start_col_r::Vector{Int32}       # length >= max_pivots + 2
    element_row_r::Vector{Float64}
    index_col_r::Vector{Int32}
    start_row_r::Vector{Int32}       # length >= max_rows_extra + 2
    len_row_r::Vector{Int32}         # length >= max_rows_extra

    # --- Pivot & permutation arrays (1-based!) ---
    pivot_region::Vector{Float64}    # length >= max_rows_extra + 2
    pivot_col::Vector{Int32}         # length >= max_cols_extra + 2
    pivot_col_back::Vector{Int32}    # length >= max_rows_extra + 2
    permute::Vector{Int32}           # length >= max_rows_extra + 2
    permute_back::Vector{Int32}      # length >= max_rows_extra + 2

    # --- Doubly-linked physical layout & Markowitz count lists (1-based!) ---
    next_col::Vector{Int32}          # length >= max_cols_extra + 2
    last_col::Vector{Int32}          # length >= max_cols_extra + 2
    next_row::Vector{Int32}          # length >= max_rows_extra + 2
    last_row::Vector{Int32}          # length >= max_rows_extra + 2
    first_count::Vector{Int32}       # 1:(nrows+1) row heads, (nrows+2):(2*nrows+2) col heads
    next_count::Vector{Int32}        # 1:nrows rows, (nrows+1):(nrows+ncols) cols
    last_count::Vector{Int32}        # 1:nrows rows, (nrows+1):(nrows+ncols) cols
    save_col::Vector{Int32}
    mark_row::Vector{Int32}
    mark_row16::Vector{UInt16}

    # --- Pre-allocated workspaces (zeroed at rest) ---
    work_area::Vector{Float64}
    work_flags::Vector{UInt32}
    sparse_list::Vector{Int32}
    sparse_stack::Vector{Int32}
    sparse_aux::Vector{Int32}
    sparse_mark::Vector{UInt8}
    dense_area::Vector{Float64}
    dense_permute::Vector{Int32}
    pivot_linked_back::Vector{Int32}
    pivot_linked_forw::Vector{Int32}
    pivot_l_order::Vector{Int32}
    last_slack::Int
end

function SparseLU()
    return SparseLU(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 0, 0,
        1.0e-13, 0.1, 1.0, -1.0, 0.0, 0, 0, 71, 2, 4, -1.0, true, true, false, false, false, false, false, false,
        0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        Float64[], Int32[], Int32[], Int32[], Int32[], Int32[], Int32[], Int32[], Int32[],
        Float64[], Int32[], Int32[], Float64[], Int32[], Int32[],
        Float64[], Int32[], Int32[], Float64[], Int32[], Int32[], Int32[],
        Float64[], Int32[], Int32[], Int32[], Int32[],
        Int32[], Int32[], Int32[], Int32[], Int32[], Int32[], Int32[], Int32[], Int32[], UInt16[],
        Float64[], UInt32[], Int32[], Int32[], Int32[], UInt8[], Float64[], Int32[],
        Int32[], Int32[], Int32[], 0,
    )
end

@inline function _ensure_len!(v::Vector{T}, n::Int)::Nothing where {T}
    old_n = length(v)
    if old_n < n
        resize!(v, n)
        @inbounds for i in (old_n + 1):n
            v[i] = zero(T)
        end
    end
    return nothing
end

@inline function _ensure_zero!(v::Vector{T}, n::Int)::Nothing where {T}
    if length(v) < n
        resize!(v, n)
    end
    @inbounds for i in 1:n
        v[i] = zero(T)
    end
    return nothing
end

"""
    allocate_areas!(lu::SparseLU, nrows, ncols, max_l, max_u)

Ensure all internal buffers of `lu` are large enough for a `(nrows, ncols)` basis
with `max_l` L-entries and `max_u` U-entries. Reuses existing allocations when possible.
Matches `CoinFactorization::getAreas`.
"""
function allocate_areas!(
    lu::SparseLU,
    nrows::Int,
    ncols::Int,
    max_l_in::Int,
    max_u_in::Int;
    already_scaled::Bool = false,
)::Nothing
    lu.nrows = nrows
    lu.ncols = ncols
    lu.bigger_dim = max(nrows, ncols)
    lu.nrows_extra = nrows
    lu.ncols_extra = ncols
    lu.max_rows_extra = nrows + lu.max_pivots
    lu.max_cols_extra = ncols + lu.max_pivots

    mre = lu.max_rows_extra
    mce = lu.max_cols_extra

    # Scale areas by area_factor (matching CoinFactorization::getAreas)
    len_area_u = max_u_in
    len_area_l = max_l_in
    if !already_scaled
        if lu.area_factor == 0.0
            lu.area_factor = 1.0
        end
        if lu.area_factor != 1.0
            len_area_u = unsafe_trunc(Int, min(lu.area_factor * len_area_u, Float64(typemax(Int32))))
            len_area_l = unsafe_trunc(Int, min(lu.area_factor * len_area_l, Float64(typemax(Int32))))
        end
    end

    lu.len_area_u = len_area_u
    lu.len_area_l = len_area_l
    lu.max_u = len_area_u

    u_alloc = len_area_u + 16
    _ensure_len!(lu.element_u, u_alloc)
    _ensure_len!(lu.index_row_u, u_alloc)
    _ensure_len!(lu.index_col_u, u_alloc)
    _ensure_len!(lu.row_to_col_u, u_alloc)
    _ensure_len!(lu.start_col_u, mce + 2)
    _ensure_len!(lu.len_col_u, mce + 2)
    _ensure_len!(lu.len_col_plus, mce + 2)
    _ensure_len!(lu.start_row_u, mre + 2)
    _ensure_len!(lu.len_row_u, mre + 2)

    _ensure_len!(lu.element_l, len_area_l + 4)
    _ensure_len!(lu.index_row_l, len_area_l + 4)
    _ensure_len!(lu.start_col_l, nrows + 2)

    _ensure_len!(lu.start_col_r, lu.max_pivots + 2)

    _ensure_len!(lu.pivot_region, mre + 2)
    _ensure_len!(lu.pivot_col, mce + 2)
    _ensure_len!(lu.pivot_col_back, mre + 2)
    _ensure_len!(lu.permute, mre + 2)
    _ensure_len!(lu.permute_back, mre + 2)

    _ensure_len!(lu.next_col, mce + 2)
    _ensure_len!(lu.last_col, mce + 2)
    _ensure_len!(lu.next_row, mre + 2)
    _ensure_len!(lu.last_row, mre + 2)
    _ensure_len!(lu.first_count, max(lu.bigger_dim + 2, mre + 1) + 2)
    _ensure_len!(lu.next_count, nrows + ncols + 2)
    _ensure_len!(lu.last_count, nrows + ncols + 2)
    _ensure_len!(lu.save_col, max(nrows, ncols) + 2)
    _ensure_len!(lu.mark_row, nrows + 2)
    _ensure_len!(lu.mark_row16, nrows + 2)

    _ensure_zero!(lu.work_area, nrows + 2)
    _ensure_len!(lu.work_flags, max(length(lu.work_flags), nrows + 2))

    _ensure_len!(lu.sparse_list, mre + 2)
    _ensure_len!(lu.sparse_stack, mre + 2)
    _ensure_len!(lu.sparse_aux, mre + 2)
    _ensure_zero!(lu.sparse_mark, mre + 16)

    _ensure_len!(lu.pivot_linked_back, mre + 2)
    _ensure_len!(lu.pivot_linked_forw, mre + 2)
    _ensure_len!(lu.pivot_l_order, nrows + 2)

    lu.ngood_u = 0
    lu.ngood_l = 0
    lu.nu = 0
    lu.nl = 0
    lu.base_l = 0
    lu.nr = 0
    lu.len_l = 0
    lu.len_r = 0
    lu.npivots = 0
    lu.ncompressions = 0
    lu.ndense = 0
    lu.nslacks = 0
    lu.has_row_l = false
    lu.has_row_r = false
    lu.has_sparse = false
    lu.status = 0

    @inbounds begin
        lu.start_col_l[1] = Int32(1)
        lu.start_col_r[1] = Int32(1)
        lu.start_row_u[mre + 1] = Int32(1)
    end
    return nothing
end

function copy_lu(src::SparseLU)::SparseLU
    return SparseLU(
        src.nrows, src.ncols, src.bigger_dim, src.nrows_extra, src.max_rows_extra,
        src.ncols_extra, src.max_cols_extra, src.ngood_u, src.ngood_l,
        src.nu, src.nl, src.base_l, src.nr, src.len_u, src.len_l, src.len_r,
        src.len_area_u, src.len_area_l, src.len_area_r, src.max_u,
        src.total_elements, src.factor_elements, src.ndense, src.nslacks,
        src.npivots, src.max_pivots, src.ncompressions, src.status,
        src.zero_tol, src.pivot_tol, src.relax_check, src.slack_value,
        src.area_factor, src.sparse_threshold, src.sparse_threshold2,
        src.dense_threshold, src.bias_lu, src.num_trials, src.time_limit, src.do_forrest_tomlin,
        src.collect_statistics, src.has_row_l, src.has_row_r, src.has_sparse,
        src.got_rcopy, src.got_ucopy, src.got_lcopy,
        src.nbtran_counts, src.btran_count_input, src.btran_count_after_u,
        src.btran_count_after_r, src.btran_count_after_l,
        src.btran_avg_after_u, src.btran_avg_after_r, src.btran_avg_after_l,
        src.nftran_counts, src.ftran_count_input, src.ftran_count_after_l,
        src.ftran_count_after_r, src.ftran_count_after_u,
        src.ftran_avg_after_l, src.ftran_avg_after_r, src.ftran_avg_after_u,
        copy(src.element_u), copy(src.index_row_u), copy(src.start_col_u),
        copy(src.len_col_u), copy(src.len_col_plus), copy(src.start_row_u),
        copy(src.len_row_u), copy(src.index_col_u), copy(src.row_to_col_u),
        copy(src.element_l), copy(src.index_row_l), copy(src.start_col_l),
        copy(src.element_row_l), copy(src.index_col_l), copy(src.start_row_l),
        copy(src.element_r), copy(src.index_row_r), copy(src.start_col_r),
        copy(src.element_row_r), copy(src.index_col_r), copy(src.start_row_r),
        copy(src.len_row_r),
        copy(src.pivot_region), copy(src.pivot_col), copy(src.pivot_col_back),
        copy(src.permute), copy(src.permute_back),
        copy(src.next_col), copy(src.last_col), copy(src.next_row), copy(src.last_row),
        copy(src.first_count), copy(src.next_count), copy(src.last_count),
        copy(src.save_col), copy(src.mark_row), copy(src.mark_row16),
        copy(src.work_area), copy(src.work_flags),
        copy(src.sparse_list), copy(src.sparse_stack), copy(src.sparse_aux),
        copy(src.sparse_mark), copy(src.dense_area), copy(src.dense_permute),
        copy(src.pivot_linked_back), copy(src.pivot_linked_forw), copy(src.pivot_l_order),
        src.last_slack,
    )
end

# --- Zero-allocation portable pure-Julia LU (LAPACK dgetrf/dgetrs layout) for dense trailing block ---
# Stores L (unit diagonal) and U in-place in column-major `a` (n x n) with 1-based pivot indices `ipiv`.
# 100% portable (macOS/Linux/Windows), zero external library dependency, thread-safe under 48+ threads,
# and zero heap allocations.
@inline function lu_lapack_dgetrf!(a::Vector{Float64}, n::Int, ipiv::Vector{Int32})::Int
    info = 0
    @inbounds for j in 1:n
        col_j = (j - 1) * n
        # Find pivot row p in j:n maximizing |a[p, j]|
        p = j
        max_v = abs(a[col_j + j])
        for i in (j + 1):n
            v = abs(a[col_j + i])
            if v > max_v
                max_v = v
                p = i
            end
        end
        ipiv[j] = Int32(p)
        piv = a[col_j + p]
        if piv == 0.0
            if info == 0
                info = j
            end
            continue
        end
        # Swap rows j and p across all columns 1:n (matches LAPACK dgetrf row swaps)
        if p != j
            for k in 1:n
                ck = (k - 1) * n
                tmp = a[ck + j]
                a[ck + j] = a[ck + p]
                a[ck + p] = tmp
            end
        end
        # Scale column j below diagonal by 1 / piv
        inv_piv = 1.0 / piv
        @simd for i in (j + 1):n
            a[col_j + i] *= inv_piv
        end
        # Rank-1 trailing submatrix update: A[j+1:n, k] -= A[j+1:n, j] * A[j, k]
        for k in (j + 1):n
            col_k = (k - 1) * n
            ajk = a[col_k + j]
            if ajk != 0.0
                @simd for i in (j + 1):n
                    a[col_k + i] = _lu_fmsub(a[col_j + i], ajk, a[col_k + i])
                end
            end
        end
    end
    return info
end

@inline function lu_lapack_dgetrs!(
    trans::UInt8,
    a::Vector{Float64},
    n::Int,
    ipiv::Vector{Int32},
    rhs::AbstractVector{Float64},
    offset::Int,
)::Nothing
    base = offset - 1
    if trans == UInt8('N') || trans == UInt8('n')
        # Solve P * L * U * x = b
        # 1. Apply row interchanges P' * b
        @inbounds for i in 1:n
            p = Int(ipiv[i])
            if p != i
                tmp = rhs[base + i]
                rhs[base + i] = rhs[base + p]
                rhs[base + p] = tmp
            end
        end
        # 2. Forward substitution: L * y = P' * b (L has unit diagonal, column-oriented)
        @inbounds for j in 1:n
            xj = rhs[base + j]
            if xj != 0.0
                col_j = (j - 1) * n
                @simd for i in (j + 1):n
                    rhs[base + i] = _lu_fmsub(a[col_j + i], xj, rhs[base + i])
                end
            end
        end
        # 3. Back substitution: U * x = y (column-oriented)
        @inbounds for j in n:-1:1
            xj = rhs[base + j]
            if xj != 0.0
                col_j = (j - 1) * n
                xj /= a[col_j + j]
                rhs[base + j] = xj
                @simd for i in 1:(j - 1)
                    rhs[base + i] = _lu_fmsub(a[col_j + i], xj, rhs[base + i])
                end
            end
        end
    else
        # Solve (P * L * U)' * x = b  =>  U' * L' * P' * x = b
        # 1. Forward substitution: U' * y = b
        @inbounds for j in 1:n
            col_j = (j - 1) * n
            s = rhs[base + j]
            @simd for i in 1:(j - 1)
                s = _lu_fmsub(a[col_j + i], rhs[base + i], s)
            end
            rhs[base + j] = s / a[col_j + j]
        end
        # 2. Back substitution: L' * z = y (L has unit diagonal)
        @inbounds for j in n:-1:1
            col_j = (j - 1) * n
            s = rhs[base + j]
            @simd for i in (j + 1):n
                s = _lu_fmsub(a[col_j + i], rhs[base + i], s)
            end
            rhs[base + j] = s
        end
        # 3. Reverse row interchanges: x = P * z
        @inbounds for i in n:-1:1
            p = Int(ipiv[i])
            if p != i
                tmp = rhs[base + i]
                rhs[base + i] = rhs[base + p]
                rhs[base + p] = tmp
            end
        end
    end
    return nothing
end
