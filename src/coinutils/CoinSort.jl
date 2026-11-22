# Faithful translation of CoinUtils/src/CoinSort.hpp
# Zero-allocation raw-pointer implementations (1-based unsafe_load/unsafe_store! offsets 1..n).

@inline function _swap2!(k::Ptr{S}, v::Ptr{T}, i::Int, j::Int)::Nothing where {S,T}
    ki = unsafe_load(k, i)
    kj = unsafe_load(k, j)
    unsafe_store!(k, kj, i)
    unsafe_store!(k, ki, j)
    vi = unsafe_load(v, i)
    vj = unsafe_load(v, j)
    unsafe_store!(v, vj, i)
    unsafe_store!(v, vi, j)
    return nothing
end

@inline function _swap3!(k::Ptr{S}, v::Ptr{T}, w::Ptr{U}, i::Int, j::Int)::Nothing where {S,T,U}
    _swap2!(k, v, i, j)
    wi = unsafe_load(w, i)
    wj = unsafe_load(w, j)
    unsafe_store!(w, wj, i)
    unsafe_store!(w, wi, j)
    return nothing
end

# Heapsort fallback for guaranteed O(n log n) worst-case if quicksort depth is exceeded
function _sift_down_2!(k::Ptr{S}, v::Ptr{T}, lo::Int, start::Int, hi::Int)::Nothing where {S,T}
    root = start
    while true
        child = lo + 2 * (root - lo) + 1
        if child > hi
            break
        end
        swap_idx = root
        if unsafe_load(k, swap_idx) < unsafe_load(k, child)
            swap_idx = child
        end
        if child + 1 <= hi && unsafe_load(k, swap_idx) < unsafe_load(k, child + 1)
            swap_idx = child + 1
        end
        if swap_idx == root
            break
        end
        _swap2!(k, v, root, swap_idx)
        root = swap_idx
    end
    return nothing
end

function _heapsort_2!(k::Ptr{S}, v::Ptr{T}, lo::Int, hi::Int)::Nothing where {S,T}
    len = hi - lo + 1
    if len <= 1
        return nothing
    end
    start = lo + (len >> 1) - 1
    while start >= lo
        _sift_down_2!(k, v, lo, start, hi)
        start -= 1
    end
    end_idx = hi
    while end_idx > lo
        _swap2!(k, v, lo, end_idx)
        end_idx -= 1
        _sift_down_2!(k, v, lo, lo, end_idx)
    end
    return nothing
end

# Exact branch-for-branch EKK quicksort partition (CoinSort.hpp:305-360) with depth guard.
# Processes the exact same sub-interval order as the explicit stack in CoinSort.hpp.
function _ekk_qsort_2!(k::Ptr{S}, v::Ptr{T}, lo::Int, hi::Int, depth_limit::Int)::Nothing where {S,T}
    while hi - lo > 10
        if depth_limit <= 0
            _heapsort_2!(k, v, lo, hi)
            return nothing
        end
        depth_limit -= 1
        l = lo
        r = hi
        m = l + ((r - l) >> 1)
        if unsafe_load(k, l) > unsafe_load(k, m)
            _swap2!(k, v, l, m)
        end
        if unsafe_load(k, m) > unsafe_load(k, r)
            _swap2!(k, v, m, r)
            if unsafe_load(k, l) > unsafe_load(k, m)
                _swap2!(k, v, l, m)
            end
        end
        c = unsafe_load(k, m)
        while r - l > 1
            l += 1
            while unsafe_load(k, l) < c
                l += 1
            end
            r -= 1
            while unsafe_load(k, r) > c
                r -= 1
            end
            _swap2!(k, v, l, r)
        end
        l = r - 1
        if l < m
            _ekk_qsort_2!(k, v, lo, l, depth_limit)
            lo = r
        else
            _ekk_qsort_2!(k, v, r, hi, depth_limit)
            hi = l
        end
    end
    return nothing
end

# Faithful CoinSort_2 (EKK path: CoinSort.hpp:263-388; delegates n > 10000 to std sort)
function coin_sort_2!(k::Ptr{S}, v::Ptr{T}, n::Int)::Nothing where {S,T}
    if n <= 1
        return nothing
    elseif n > 10000
        coin_sort_2_std!(k, v, n)
        return nothing
    end
    # Check already sorted (CoinSort.hpp:291-301)
    last = unsafe_load(k, 1)
    j = 2
    while j <= n
        kj = unsafe_load(k, j)
        if kj >= last
            last = kj
            j += 1
        else
            break
        end
    end
    if j > n
        return nothing
    end

    # Depth limit 2 * floor(log2(n)) + 4 for guaranteed O(n log n)
    depth_limit = 2 * (64 - leading_zeros(UInt64(n))) + 4
    _ekk_qsort_2!(k, v, 1, n, depth_limit)

    # Final insertion sort pass (CoinSort.hpp:361-372)
    l = 1
    while l < n
        kl = unsafe_load(k, l)
        kn = unsafe_load(k, l + 1)
        if kl > kn
            c = kn
            it = unsafe_load(v, l + 1)
            r = l
            while r >= 1
                kr = unsafe_load(k, r)
                if kr > c
                    unsafe_store!(k, kr, r + 1)
                    unsafe_store!(v, unsafe_load(v, r), r + 1)
                    r -= 1
                else
                    break
                end
            end
            unsafe_store!(k, c, r + 1)
            unsafe_store!(v, it, r + 1)
        end
        l += 1
    end
    return nothing
end

# Descending variant (CoinFirstGreater_2)
function coin_sort_2_desc!(k::Ptr{S}, v::Ptr{T}, n::Int)::Nothing where {S,T}
    if n <= 1
        return nothing
    end
    # Insertion sort for small arrays; negating/reversing or direct comparison
    # Quick exact introsort for descending order
    _qsort_2_desc!(k, v, 1, n, 2 * (64 - leading_zeros(UInt64(n))) + 4)
    l = 1
    while l < n
        kl = unsafe_load(k, l)
        kn = unsafe_load(k, l + 1)
        if kl < kn
            c = kn
            it = unsafe_load(v, l + 1)
            r = l
            while r >= 1
                kr = unsafe_load(k, r)
                if kr < c
                    unsafe_store!(k, kr, r + 1)
                    unsafe_store!(v, unsafe_load(v, r), r + 1)
                    r -= 1
                else
                    break
                end
            end
            unsafe_store!(k, c, r + 1)
            unsafe_store!(v, it, r + 1)
        end
        l += 1
    end
    return nothing
end

function _qsort_2_desc!(k::Ptr{S}, v::Ptr{T}, lo::Int, hi::Int, depth_limit::Int)::Nothing where {S,T}
    while hi - lo > 10
        if depth_limit <= 0
            # Simple insertion fallback if needed; rarely triggered
            return nothing
        end
        depth_limit -= 1
        l = lo
        r = hi
        m = l + ((r - l) >> 1)
        if unsafe_load(k, l) < unsafe_load(k, m)
            _swap2!(k, v, l, m)
        end
        if unsafe_load(k, m) < unsafe_load(k, r)
            _swap2!(k, v, m, r)
            if unsafe_load(k, l) < unsafe_load(k, m)
                _swap2!(k, v, l, m)
            end
        end
        c = unsafe_load(k, m)
        while r - l > 1
            l += 1
            while unsafe_load(k, l) > c
                l += 1
            end
            r -= 1
            while unsafe_load(k, r) < c
                r -= 1
            end
            _swap2!(k, v, l, r)
        end
        l = r - 1
        if l < m
            _qsort_2_desc!(k, v, lo, l, depth_limit)
            lo = r
        else
            _qsort_2_desc!(k, v, r, hi, depth_limit)
            hi = l
        end
    end
    return nothing
end

# CoinSort_3 (ascending on key, carrying array2 and array3)
function _ekk_qsort_3!(k::Ptr{S}, v::Ptr{T}, w::Ptr{U}, lo::Int, hi::Int, depth_limit::Int)::Nothing where {S,T,U}
    while hi - lo > 10
        if depth_limit <= 0
            return nothing
        end
        depth_limit -= 1
        l = lo
        r = hi
        m = l + ((r - l) >> 1)
        if unsafe_load(k, l) > unsafe_load(k, m)
            _swap3!(k, v, w, l, m)
        end
        if unsafe_load(k, m) > unsafe_load(k, r)
            _swap3!(k, v, w, m, r)
            if unsafe_load(k, l) > unsafe_load(k, m)
                _swap3!(k, v, w, l, m)
            end
        end
        c = unsafe_load(k, m)
        while r - l > 1
            l += 1
            while unsafe_load(k, l) < c
                l += 1
            end
            r -= 1
            while unsafe_load(k, r) > c
                r -= 1
            end
            _swap3!(k, v, w, l, r)
        end
        l = r - 1
        if l < m
            _ekk_qsort_3!(k, v, w, lo, l, depth_limit)
            lo = r
        else
            _ekk_qsort_3!(k, v, w, r, hi, depth_limit)
            hi = l
        end
    end
    return nothing
end

function coin_sort_3!(k::Ptr{S}, v::Ptr{T}, w::Ptr{U}, n::Int)::Nothing where {S,T,U}
    if n <= 1
        return nothing
    end
    last = unsafe_load(k, 1)
    j = 2
    while j <= n
        kj = unsafe_load(k, j)
        if kj >= last
            last = kj
            j += 1
        else
            break
        end
    end
    if j > n
        return nothing
    end
    depth_limit = 2 * (64 - leading_zeros(UInt64(n))) + 4
    _ekk_qsort_3!(k, v, w, 1, n, depth_limit)
    l = 1
    while l < n
        kl = unsafe_load(k, l)
        kn = unsafe_load(k, l + 1)
        if kl > kn
            c = kn
            it = unsafe_load(v, l + 1)
            iu = unsafe_load(w, l + 1)
            r = l
            while r >= 1
                kr = unsafe_load(k, r)
                if kr > c
                    unsafe_store!(k, kr, r + 1)
                    unsafe_store!(v, unsafe_load(v, r), r + 1)
                    unsafe_store!(w, unsafe_load(w, r), r + 1)
                    r -= 1
                else
                    break
                end
            end
            unsafe_store!(k, c, r + 1)
            unsafe_store!(v, it, r + 1)
            unsafe_store!(w, iu, r + 1)
        end
        l += 1
    end
    return nothing
end

# Exact port of Apple libc++ std::sort (__introsort, non-branchless, _UseBitSetPartition=false)
# Guarantees bit-identical tie-breaking order with C++ std::sort on CoinPair<S, T>
# (needed for exact presolve parity on czprob, greenbea, greenbeb, fit1d, etc.).

@inline function _lc_sort3!(k::Ptr{S}, v::Ptr{T}, x::Int, y::Int, z::Int)::Bool where {S,T}
    if !(unsafe_load(k, y) < unsafe_load(k, x))
        if !(unsafe_load(k, z) < unsafe_load(k, y))
            return false
        end
        _swap2!(k, v, y, z)
        if unsafe_load(k, y) < unsafe_load(k, x)
            _swap2!(k, v, x, y)
        end
        return true
    end
    if unsafe_load(k, z) < unsafe_load(k, y)
        _swap2!(k, v, x, z)
        return true
    end
    _swap2!(k, v, x, y)
    if unsafe_load(k, z) < unsafe_load(k, y)
        _swap2!(k, v, y, z)
    end
    return true
end

@inline function _lc_sort4!(k::Ptr{S}, v::Ptr{T}, x1::Int, x2::Int, x3::Int, x4::Int)::Nothing where {S,T}
    _lc_sort3!(k, v, x1, x2, x3)
    if unsafe_load(k, x4) < unsafe_load(k, x3)
        _swap2!(k, v, x3, x4)
        if unsafe_load(k, x3) < unsafe_load(k, x2)
            _swap2!(k, v, x2, x3)
            if unsafe_load(k, x2) < unsafe_load(k, x1)
                _swap2!(k, v, x1, x2)
            end
        end
    end
    return nothing
end

@inline function _lc_sort5!(k::Ptr{S}, v::Ptr{T}, x1::Int, x2::Int, x3::Int, x4::Int, x5::Int)::Nothing where {S,T}
    _lc_sort4!(k, v, x1, x2, x3, x4)
    if unsafe_load(k, x5) < unsafe_load(k, x4)
        _swap2!(k, v, x4, x5)
        if unsafe_load(k, x4) < unsafe_load(k, x3)
            _swap2!(k, v, x3, x4)
            if unsafe_load(k, x3) < unsafe_load(k, x2)
                _swap2!(k, v, x2, x3)
                if unsafe_load(k, x2) < unsafe_load(k, x1)
                    _swap2!(k, v, x1, x2)
                end
            end
        end
    end
    return nothing
end

# Range [first, last) with 1-based indices (last is one-past-end)
function _lc_insertion_sort!(k::Ptr{S}, v::Ptr{T}, first::Int, last::Int)::Nothing where {S,T}
    if first == last
        return nothing
    end
    i = first + 1
    while i < last
        j = i - 1
        if unsafe_load(k, i) < unsafe_load(k, j)
            tk = unsafe_load(k, i)
            tv = unsafe_load(v, i)
            kk = j
            j = i
            while true
                unsafe_store!(k, unsafe_load(k, kk), j)
                unsafe_store!(v, unsafe_load(v, kk), j)
                j = kk
                if j == first
                    break
                end
                kk -= 1
                if !(tk < unsafe_load(k, kk))
                    break
                end
            end
            unsafe_store!(k, tk, j)
            unsafe_store!(v, tv, j)
        end
        i += 1
    end
    return nothing
end

function _lc_insertion_sort_unguarded!(k::Ptr{S}, v::Ptr{T}, first::Int, last::Int)::Nothing where {S,T}
    if first == last
        return nothing
    end
    i = first + 1
    while i < last
        j = i - 1
        if unsafe_load(k, i) < unsafe_load(k, j)
            tk = unsafe_load(k, i)
            tv = unsafe_load(v, i)
            kk = j
            j = i
            while true
                unsafe_store!(k, unsafe_load(k, kk), j)
                unsafe_store!(v, unsafe_load(v, kk), j)
                j = kk
                kk -= 1
                if !(tk < unsafe_load(k, kk))
                    break
                end
            end
            unsafe_store!(k, tk, j)
            unsafe_store!(v, tv, j)
        end
        i += 1
    end
    return nothing
end

function _lc_insertion_sort_incomplete!(k::Ptr{S}, v::Ptr{T}, first::Int, last::Int)::Bool where {S,T}
    len = last - first
    if len <= 1
        return true
    elseif len == 2
        if unsafe_load(k, last - 1) < unsafe_load(k, first)
            _swap2!(k, v, first, last - 1)
        end
        return true
    elseif len == 3
        _lc_sort3!(k, v, first, first + 1, last - 1)
        return true
    elseif len == 4
        _lc_sort4!(k, v, first, first + 1, first + 2, last - 1)
        return true
    elseif len == 5
        _lc_sort5!(k, v, first, first + 1, first + 2, first + 3, last - 1)
        return true
    end
    j = first + 2
    _lc_sort3!(k, v, first, first + 1, j)
    count = 0
    i = j + 1
    while i < last
        if unsafe_load(k, i) < unsafe_load(k, j)
            tk = unsafe_load(k, i)
            tv = unsafe_load(v, i)
            kk = j
            j = i
            while true
                unsafe_store!(k, unsafe_load(k, kk), j)
                unsafe_store!(v, unsafe_load(v, kk), j)
                j = kk
                if j == first
                    break
                end
                kk -= 1
                if !(tk < unsafe_load(k, kk))
                    break
                end
            end
            unsafe_store!(k, tk, j)
            unsafe_store!(v, tv, j)
            count += 1
            if count == 8
                return (i + 1) == last
            end
        end
        j = i
        i += 1
    end
    return true
end

function _lc_partition_equals_right!(k::Ptr{S}, v::Ptr{T}, first::Int, last::Int)::Tuple{Int,Bool} where {S,T}
    begin_idx = first
    pk = unsafe_load(k, first)
    pv = unsafe_load(v, first)

    first += 1
    while unsafe_load(k, first) < pk
        first += 1
    end

    if begin_idx == first - 1
        while first < last
            last -= 1
            if unsafe_load(k, last) < pk
                break
            end
        end
    else
        while true
            last -= 1
            if unsafe_load(k, last) < pk
                break
            end
        end
    end

    already_partitioned = first >= last
    while first < last
        _swap2!(k, v, first, last)
        first += 1
        while unsafe_load(k, first) < pk
            first += 1
        end
        last -= 1
        while !(unsafe_load(k, last) < pk)
            last -= 1
        end
    end

    pivot_pos = first - 1
    if begin_idx != pivot_pos
        unsafe_store!(k, unsafe_load(k, pivot_pos), begin_idx)
        unsafe_store!(v, unsafe_load(v, pivot_pos), begin_idx)
    end
    unsafe_store!(k, pk, pivot_pos)
    unsafe_store!(v, pv, pivot_pos)
    return pivot_pos, already_partitioned
end

function _lc_partition_equals_left!(k::Ptr{S}, v::Ptr{T}, first::Int, last::Int)::Int where {S,T}
    begin_idx = first
    pk = unsafe_load(k, first)
    pv = unsafe_load(v, first)

    if pk < unsafe_load(k, last - 1)
        first += 1
        while !(pk < unsafe_load(k, first))
            first += 1
        end
    else
        first += 1
        while first < last && !(pk < unsafe_load(k, first))
            first += 1
        end
    end

    if first < last
        last -= 1
        while pk < unsafe_load(k, last)
            last -= 1
        end
    end

    while first < last
        _swap2!(k, v, first, last)
        first += 1
        while !(pk < unsafe_load(k, first))
            first += 1
        end
        last -= 1
        while pk < unsafe_load(k, last)
            last -= 1
        end
    end

    pivot_pos = first - 1
    if begin_idx != pivot_pos
        unsafe_store!(k, unsafe_load(k, pivot_pos), begin_idx)
        unsafe_store!(v, unsafe_load(v, pivot_pos), begin_idx)
    end
    unsafe_store!(k, pk, pivot_pos)
    unsafe_store!(v, pv, pivot_pos)
    return first
end

function _lc_introsort!(k::Ptr{S}, v::Ptr{T}, first::Int, last::Int, depth::Int, leftmost::Bool)::Nothing where {S,T}
    while true
        len = last - first
        if len <= 1
            return nothing
        elseif len == 2
            if unsafe_load(k, last - 1) < unsafe_load(k, first)
                _swap2!(k, v, first, last - 1)
            end
            return nothing
        elseif len == 3
            _lc_sort3!(k, v, first, first + 1, last - 1)
            return nothing
        elseif len == 4
            _lc_sort4!(k, v, first, first + 1, first + 2, last - 1)
            return nothing
        elseif len == 5
            _lc_sort5!(k, v, first, first + 1, first + 2, first + 3, last - 1)
            return nothing
        end

        if len < 24
            if leftmost
                _lc_insertion_sort!(k, v, first, last)
            else
                _lc_insertion_sort_unguarded!(k, v, first, last)
            end
            return nothing
        end

        if depth == 0
            _heapsort_2!(k, v, first, last - 1)
            return nothing
        end
        depth -= 1

        half_len = len >> 1
        if len > 128
            _lc_sort3!(k, v, first, first + half_len, last - 1)
            _lc_sort3!(k, v, first + 1, first + (half_len - 1), last - 2)
            _lc_sort3!(k, v, first + 2, first + (half_len + 1), last - 3)
            _lc_sort3!(k, v, first + (half_len - 1), first + half_len, first + (half_len + 1))
            _swap2!(k, v, first, first + half_len)
        else
            _lc_sort3!(k, v, first + half_len, first, last - 1)
        end

        if !leftmost && !(unsafe_load(k, first - 1) < unsafe_load(k, first))
            first = _lc_partition_equals_left!(k, v, first, last)
            continue
        end

        pivot_pos, already_part = _lc_partition_equals_right!(k, v, first, last)
        if already_part
            fs = _lc_insertion_sort_incomplete!(k, v, first, pivot_pos)
            if _lc_insertion_sort_incomplete!(k, v, pivot_pos + 1, last)
                if fs
                    return nothing
                end
                last = pivot_pos
                continue
            else
                if fs
                    first = pivot_pos + 1
                    continue
                end
            end
        end

        _lc_introsort!(k, v, first, pivot_pos, depth, leftmost)
        leftmost = false
        first = pivot_pos + 1
    end
end

function coin_sort_2_std!(k::Ptr{S}, v::Ptr{T}, n::Int)::Nothing where {S,T}
    if n <= 1
        return nothing
    end
    depth_limit = 2 * (63 - leading_zeros(UInt64(n)))
    _lc_introsort!(k, v, 1, n + 1, depth_limit, true)
    return nothing
end

# Vector convenience overloads
function coin_sort_2_std!(k::Vector{S}, v::Vector{T}, n::Int = length(k))::Nothing where {S,T}
    GC.@preserve k v coin_sort_2_std!(pointer(k), pointer(v), n)
    return nothing
end

function coin_sort_2!(k::Vector{S}, v::Vector{T}, n::Int = length(k))::Nothing where {S,T}
    GC.@preserve k v coin_sort_2!(pointer(k), pointer(v), n)
    return nothing
end

function coin_sort_2_desc!(k::Vector{S}, v::Vector{T}, n::Int = length(k))::Nothing where {S,T}
    GC.@preserve k v coin_sort_2_desc!(pointer(k), pointer(v), n)
    return nothing
end

function coin_sort_3!(k::Vector{S}, v::Vector{T}, w::Vector{U}, n::Int = length(k))::Nothing where {S,T,U}
    GC.@preserve k v w coin_sort_3!(pointer(k), pointer(v), pointer(w), n)
    return nothing
end
