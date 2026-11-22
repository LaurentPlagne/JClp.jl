# Faithful translation of CoinUtils/src/CoinIndexedVector.{hpp,cpp}
# Provides:
#   1. Zero-allocation raw-pointer kernels (0-based indices) for C++ bridge (`JCLP_BRICK_B2`)
#   2. Standalone Julia `CoinIndexedVector` struct for pure-Julia bricks B3..B17.

const COIN_INDEXED_TINY_ELEMENT = 1.0e-50
const COIN_INDEXED_REALLY_TINY_ELEMENT = 1.0e-100
const COIN_DBL_MIN_VAL = floatmin(Float64)

# =============================================================================
# 1. Raw-pointer kernels (0-based index convention matching CoinIndexedVector)
# =============================================================================

@inline function civ_clear_ptr!(
    elements::Ptr{Float64},
    indices::Ptr{Int32},
    n_elements::Int32,
    capacity::Int32,
    packed_mode::Bool,
)::Nothing
    n = Int(n_elements)
    cap = Int(capacity)
    if !packed_mode
        if 3 * n < cap
            i = 1
            if (n & 1) != 0
                idx0 = Int(unsafe_load(indices, 1))
                unsafe_store!(elements, 0.0, idx0 + 1)
                i = 2
            end
            while i <= n
                i0 = Int(unsafe_load(indices, i))
                i1 = Int(unsafe_load(indices, i + 1))
                unsafe_store!(elements, 0.0, i0 + 1)
                unsafe_store!(elements, 0.0, i1 + 1)
                i += 2
            end
        else
            coin_zero_n!(elements, cap)
        end
    else
        coin_zero_n!(elements, n)
    end
    return nothing
end

@inline function civ_really_clear_ptr!(elements::Ptr{Float64}, capacity::Int32)::Nothing
    coin_zero_n!(elements, Int(capacity))
    return nothing
end

@inline function civ_clean_ptr!(
    elements::Ptr{Float64},
    indices::Ptr{Int32},
    n_elements::Int32,
    tolerance::Float64,
)::Int32
    number = Int(n_elements)
    out_n = 0
    i = 1
    while i <= number
        idx = Int(unsafe_load(indices, i))
        val = unsafe_load(elements, idx + 1)
        if abs(val) >= tolerance
            out_n += 1
            unsafe_store!(indices, Int32(idx), out_n)
        else
            unsafe_store!(elements, 0.0, idx + 1)
        end
        i += 1
    end
    return Int32(out_n)
end

@inline function civ_scan_ptr!(
    elements::Ptr{Float64},
    indices::Ptr{Int32},
    n_elements::Int32,
    capacity::Int32,
    start_idx::Int32,
    end_idx::Int32,
)::Int32
    s = Int(start_idx < Int32(0) ? Int32(0) : start_idx)
    e = Int(end_idx > capacity ? capacity : end_idx)
    base = Int(n_elements)
    number = 0
    i = s
    while i < e
        if unsafe_load(elements, i + 1) != 0.0
            number += 1
            unsafe_store!(indices, Int32(i), base + number)
        end
        i += 1
    end
    return Int32(number)
end

@inline function civ_scan_tol_ptr!(
    elements::Ptr{Float64},
    indices::Ptr{Int32},
    n_elements::Int32,
    capacity::Int32,
    start_idx::Int32,
    end_idx::Int32,
    tolerance::Float64,
)::Int32
    s = Int(start_idx < Int32(0) ? Int32(0) : start_idx)
    e = Int(end_idx > capacity ? capacity : end_idx)
    base = Int(n_elements)
    number = 0
    i = s
    while i < e
        val = unsafe_load(elements, i + 1)
        if val != 0.0
            if abs(val) >= tolerance
                number += 1
                unsafe_store!(indices, Int32(i), base + number)
            else
                unsafe_store!(elements, 0.0, i + 1)
            end
        end
        i += 1
    end
    return Int32(number)
end

@inline function civ_clean_and_pack_ptr!(
    elements::Ptr{Float64},
    indices::Ptr{Int32},
    n_elements::Int32,
    tolerance::Float64,
)::Int32
    number = Int(n_elements)
    out_n = 0
    i = 1
    while i <= number
        idx = Int(unsafe_load(indices, i))
        val = unsafe_load(elements, idx + 1)
        unsafe_store!(elements, 0.0, idx + 1)
        if abs(val) >= tolerance
            out_n += 1
            unsafe_store!(elements, val, out_n)
            unsafe_store!(indices, Int32(idx), out_n)
        end
        i += 1
    end
    return Int32(out_n)
end

@inline function civ_scan_and_pack_ptr!(
    elements::Ptr{Float64},
    indices::Ptr{Int32},
    n_elements::Int32,
    capacity::Int32,
    start_idx::Int32,
    end_idx::Int32,
)::Int32
    s = Int(start_idx < Int32(0) ? Int32(0) : start_idx)
    e = Int(end_idx > capacity ? capacity : end_idx)
    base = Int(n_elements)
    number = 0
    i = s
    while i < e
        val = unsafe_load(elements, i + 1)
        unsafe_store!(elements, 0.0, i + 1)
        if val != 0.0
            number += 1
            unsafe_store!(elements, val, number)
            unsafe_store!(indices, Int32(i), base + number)
        end
        i += 1
    end
    return Int32(number)
end

@inline function civ_scan_and_pack_tol_ptr!(
    elements::Ptr{Float64},
    indices::Ptr{Int32},
    n_elements::Int32,
    capacity::Int32,
    start_idx::Int32,
    end_idx::Int32,
    tolerance::Float64,
)::Int32
    s = Int(start_idx < Int32(0) ? Int32(0) : start_idx)
    e = Int(end_idx > capacity ? capacity : end_idx)
    base = Int(n_elements)
    number = 0
    i = s
    while i < e
        val = unsafe_load(elements, i + 1)
        unsafe_store!(elements, 0.0, i + 1)
        if abs(val) >= tolerance
            number += 1
            unsafe_store!(elements, val, number)
            unsafe_store!(indices, Int32(i), base + number)
        end
        i += 1
    end
    return Int32(number)
end

@inline function civ_create_packed_ptr!(
    elements::Ptr{Float64},
    indices::Ptr{Int32},
    number::Int32,
    src_indices::Ptr{Int32},
    src_elements::Ptr{Float64},
)::Nothing
    n = Int(number)
    coin_memcpy_n!(src_indices, n, indices)
    coin_memcpy_n!(src_elements, n, elements)
    return nothing
end

@inline function civ_create_unpacked_ptr!(
    elements::Ptr{Float64},
    indices::Ptr{Int32},
    number::Int32,
    src_indices::Ptr{Int32},
    src_elements::Ptr{Float64},
)::Nothing
    n = Int(number)
    i = 1
    while i <= n
        row = Int(unsafe_load(src_indices, i))
        unsafe_store!(indices, Int32(row), i)
        unsafe_store!(elements, unsafe_load(src_elements, i), row + 1)
        i += 1
    end
    return nothing
end

@inline function civ_copy_fast_ptr!(
    dst_elements::Ptr{Float64},
    dst_indices::Ptr{Int32},
    dst_n::Int32,
    capacity::Int32,
    dst_packed::Bool,
    src_elements::Ptr{Float64},
    src_indices::Ptr{Int32},
    src_n::Int32,
    src_packed::Bool,
    multiplier::Float64,
)::Int32
    civ_clear_ptr!(dst_elements, dst_indices, dst_n, capacity, dst_packed)
    sn = Int(src_n)
    out_n = 0
    if !src_packed
        i = 1
        while i <= sn
            idx = Int(unsafe_load(src_indices, i))
            val = unsafe_load(src_elements, idx + 1) * multiplier
            if abs(val) < COIN_INDEXED_TINY_ELEMENT
                val = COIN_INDEXED_REALLY_TINY_ELEMENT
            end
            unsafe_store!(dst_elements, val, idx + 1)
            out_n += 1
            unsafe_store!(dst_indices, Int32(idx), out_n)
            i += 1
        end
    else
        i = 1
        while i <= sn
            idx = Int(unsafe_load(src_indices, i))
            val = unsafe_load(src_elements, i) * multiplier
            if abs(val) < COIN_INDEXED_TINY_ELEMENT
                val = COIN_INDEXED_REALLY_TINY_ELEMENT
            end
            out_n += 1
            unsafe_store!(dst_elements, val, out_n)
            unsafe_store!(dst_indices, Int32(idx), out_n)
            i += 1
        end
    end
    return Int32(out_n)
end

@inline function civ_add_ptr!(
    elements::Ptr{Float64},
    indices::Ptr{Int32},
    n_elements::Int32,
    index::Int32,
    element::Float64,
)::Int32
    idx1 = Int(index) + 1
    cur = unsafe_load(elements, idx1)
    if cur != 0.0
        nv = cur + element
        if abs(nv) >= COIN_INDEXED_TINY_ELEMENT
            unsafe_store!(elements, nv, idx1)
        else
            unsafe_store!(elements, COIN_INDEXED_REALLY_TINY_ELEMENT, idx1)
        end
        return n_elements
    elseif abs(element) >= COIN_INDEXED_TINY_ELEMENT
        n_new = n_elements + Int32(1)
        unsafe_store!(indices, index, Int(n_new))
        unsafe_store!(elements, element, idx1)
        return n_new
    end
    return n_elements
end

# =============================================================================
# 2. Standalone Julia `CoinIndexedVector` (0-based indices stored in `indices`)
# =============================================================================

mutable struct CoinIndexedVector
    indices::Vector{Int32}
    elements::Vector{Float64}
    n_elements::Int32
    capacity::Int32
    packed_mode::Bool
end

function CoinIndexedVector(cap::Integer = 0)
    c = Int(cap)
    return CoinIndexedVector(
        zeros(Int32, c),
        zeros(Float64, c),
        Int32(0),
        Int32(c),
        false,
    )
end

@inline get_num_elements(v::CoinIndexedVector)::Int32 = v.n_elements
@inline capacity(v::CoinIndexedVector)::Int32 = v.capacity
@inline packed_mode(v::CoinIndexedVector)::Bool = v.packed_mode
@inline function set_packed_mode!(v::CoinIndexedVector, yes::Bool)::Nothing
    v.packed_mode = yes
    return nothing
end
@inline function set_num_elements!(v::CoinIndexedVector, n::Int32)::Nothing
    v.n_elements = n
    if n == Int32(0)
        v.packed_mode = false
    end
    return nothing
end

function reserve!(v::CoinIndexedVector, n::Integer)::Nothing
    n_int = Int(n)
    if n_int > Int(v.capacity)
        old_cap = Int(v.capacity)
        resize!(v.indices, n_int)
        resize!(v.elements, n_int)
        for i in (old_cap + 1):n_int
            @inbounds v.indices[i] = Int32(0)
            @inbounds v.elements[i] = 0.0
        end
        v.capacity = Int32(n_int)
    end
    return nothing
end

function clear!(v::CoinIndexedVector)::Nothing
    GC.@preserve v civ_clear_ptr!(
        pointer(v.elements), pointer(v.indices), v.n_elements, v.capacity, v.packed_mode,
    )
    v.n_elements = Int32(0)
    v.packed_mode = false
    return nothing
end

function really_clear!(v::CoinIndexedVector)::Nothing
    GC.@preserve v civ_really_clear_ptr!(pointer(v.elements), v.capacity)
    v.n_elements = Int32(0)
    v.packed_mode = false
    return nothing
end

function empty!(v::CoinIndexedVector)::Nothing
    Base.empty!(v.indices)
    Base.empty!(v.elements)
    v.n_elements = Int32(0)
    v.capacity = Int32(0)
    v.packed_mode = false
    return nothing
end

@inline function quick_insert!(v::CoinIndexedVector, index0::Int32, val::Float64)::Nothing
    v.n_elements += Int32(1)
    @inbounds v.indices[Int(v.n_elements)] = index0
    @inbounds v.elements[Int(index0) + 1] = val
    return nothing
end

function insert!(v::CoinIndexedVector, index0::Int32, val::Float64)::Nothing
    if index0 >= v.capacity
        reserve!(v, Int(index0) + 1)
    end
    quick_insert!(v, index0, val)
    return nothing
end

@inline function quick_add!(v::CoinIndexedVector, index0::Int32, val::Float64)::Nothing
    v.n_elements = GC.@preserve v civ_add_ptr!(
        pointer(v.elements), pointer(v.indices), v.n_elements, index0, val,
    )
    return nothing
end

function add!(v::CoinIndexedVector, index0::Int32, val::Float64)::Nothing
    if index0 >= v.capacity
        reserve!(v, Int(index0) + 1)
    end
    quick_add!(v, index0, val)
    return nothing
end

@inline function zero_entry!(v::CoinIndexedVector, index0::Int32)::Nothing
    i1 = Int(index0) + 1
    @inbounds if v.elements[i1] != 0.0
        v.elements[i1] = COIN_INDEXED_REALLY_TINY_ELEMENT
    end
    return nothing
end

function clean!(v::CoinIndexedVector, tolerance::Float64)::Int32
    v.n_elements = GC.@preserve v civ_clean_ptr!(
        pointer(v.elements), pointer(v.indices), v.n_elements, tolerance,
    )
    return v.n_elements
end

function scan!(v::CoinIndexedVector)::Int32
    v.n_elements = Int32(0)
    return scan!(v, Int32(0), v.capacity)
end

function scan!(v::CoinIndexedVector, start0::Int32, end0::Int32)::Int32
    added = GC.@preserve v civ_scan_ptr!(
        pointer(v.elements), pointer(v.indices), v.n_elements, v.capacity, start0, end0,
    )
    v.n_elements += added
    return added
end

function scan!(v::CoinIndexedVector, tolerance::Float64)::Int32
    v.n_elements = Int32(0)
    return scan!(v, Int32(0), v.capacity, tolerance)
end

function scan!(v::CoinIndexedVector, start0::Int32, end0::Int32, tolerance::Float64)::Int32
    added = GC.@preserve v civ_scan_tol_ptr!(
        pointer(v.elements), pointer(v.indices), v.n_elements, v.capacity, start0, end0, tolerance,
    )
    v.n_elements += added
    return added
end

function clean_and_pack!(v::CoinIndexedVector, tolerance::Float64)::Int32
    if !v.packed_mode
        v.n_elements = GC.@preserve v civ_clean_and_pack_ptr!(
            pointer(v.elements), pointer(v.indices), v.n_elements, tolerance,
        )
        v.packed_mode = true
    end
    return v.n_elements
end

function scan_and_pack!(v::CoinIndexedVector, start0::Int32 = Int32(0), end0::Int32 = v.capacity)::Int32
    if start0 == Int32(0) && end0 == v.capacity
        v.n_elements = Int32(0)
    end
    added = GC.@preserve v civ_scan_and_pack_ptr!(
        pointer(v.elements), pointer(v.indices), v.n_elements, v.capacity, start0, end0,
    )
    v.n_elements += added
    v.packed_mode = true
    return added
end

function scan_and_pack!(v::CoinIndexedVector, tolerance::Float64)::Int32
    v.n_elements = Int32(0)
    added = GC.@preserve v civ_scan_and_pack_tol_ptr!(
        pointer(v.elements), pointer(v.indices), v.n_elements, v.capacity, Int32(0), v.capacity, tolerance,
    )
    v.n_elements += added
    v.packed_mode = true
    return added
end

function expand!(v::CoinIndexedVector)::Nothing
    if v.n_elements > Int32(0) && v.packed_mode
        n = Int(v.n_elements)
        temp = zeros(Float64, Int(v.capacity))
        for i in 1:n
            @inbounds temp[Int(v.indices[i]) + 1] = v.elements[i]
        end
        coin_zero_n!(v.elements, n)
        for i in 1:n
            @inbounds r = Int(v.indices[i]) + 1
            @inbounds v.elements[r] = temp[r]
        end
    end
    v.packed_mode = false
    return nothing
end

function create_packed!(v::CoinIndexedVector, n::Int32, src_ind::Vector{Int32}, src_el::Vector{Float64})::Nothing
    if n > v.capacity
        reserve!(v, n)
    end
    GC.@preserve v src_ind src_el civ_create_packed_ptr!(
        pointer(v.elements), pointer(v.indices), n, pointer(src_ind), pointer(src_el),
    )
    v.n_elements = n
    v.packed_mode = true
    return nothing
end

function create_unpacked!(v::CoinIndexedVector, n::Int32, src_ind::Vector{Int32}, src_el::Vector{Float64})::Nothing
    GC.@preserve v src_ind src_el civ_create_unpacked_ptr!(
        pointer(v.elements), pointer(v.indices), n, pointer(src_ind), pointer(src_el),
    )
    v.n_elements = n
    v.packed_mode = false
    return nothing
end

function copy_indexed!(dst::CoinIndexedVector, src::CoinIndexedVector, multiplier::Float64 = 1.0)::Nothing
    if dst.capacity != src.capacity
        empty!(dst)
        reserve!(dst, src.capacity)
    end
    dst.n_elements = GC.@preserve dst src civ_copy_fast_ptr!(
        pointer(dst.elements), pointer(dst.indices), dst.n_elements, dst.capacity, dst.packed_mode,
        pointer(src.elements), pointer(src.indices), src.n_elements, src.packed_mode, multiplier,
    )
    dst.packed_mode = src.packed_mode
    return nothing
end

function sort_packed!(v::CoinIndexedVector)::Nothing
    @assert v.packed_mode
    coin_sort_2!(v.indices, v.elements, Int(v.n_elements))
    return nothing
end

function check_clear(v::CoinIndexedVector)::Bool
    if v.n_elements != Int32(0) || v.packed_mode
        return false
    end
    for i in 1:Int(v.capacity)
        @inbounds if v.elements[i] != 0.0
            return false
        end
    end
    return true
end
