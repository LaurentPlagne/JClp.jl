# Faithful translation of CoinUtils/src/CoinHelperFunctions.hpp
# Raw-pointer (0-based) and Vector helpers; zero-allocation and --trim=safe compatible.

@inline function coin_max(a::T, b::T)::T where {T}
    return a > b ? a : b
end

@inline function coin_min(a::T, b::T)::T where {T}
    return a < b ? a : b
end

@inline function coin_copy_n!(from::Ptr{T}, size::Int, to::Ptr{T})::Nothing where {T}
    if size <= 0 || from == to
        return nothing
    end
    ccall(:memmove, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
          to, from, Csize_t(size) * Csize_t(sizeof(T)))
    return nothing
end

@inline function coin_memcpy_n!(from::Ptr{T}, size::Int, to::Ptr{T})::Nothing where {T}
    if size <= 0
        return nothing
    end
    ccall(:memcpy, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
          to, from, Csize_t(size) * Csize_t(sizeof(T)))
    return nothing
end

@inline function coin_zero_n!(to::Ptr{T}, size::Int)::Nothing where {T}
    if size <= 0
        return nothing
    end
    ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t),
          to, Cint(0), Csize_t(size) * Csize_t(sizeof(T)))
    return nothing
end

@inline function coin_fill_n!(to::Ptr{T}, size::Int, value::T)::Nothing where {T}
    i = 1
    while i <= size
        unsafe_store!(to, value, i)
        i += 1
    end
    return nothing
end

@inline function coin_iota_n!(first::Ptr{T}, size::Int, init::T)::Nothing where {T}
    v = init
    i = 1
    while i <= size
        unsafe_store!(first, v, i)
        v += one(T)
        i += 1
    end
    return nothing
end

@inline function coin_is_sorted(first::Ptr{T}, size::Int)::Bool where {T}
    if size <= 1
        return true
    end
    prev = unsafe_load(first, 1)
    i = 2
    while i <= size
        cur = unsafe_load(first, i)
        if cur < prev
            return false
        end
        prev = cur
        i += 1
    end
    return true
end

# Convenience Vector overloads (1-based Julia slices)
@inline function coin_zero_n!(v::Vector{T}, n::Int = length(v))::Nothing where {T}
    GC.@preserve v coin_zero_n!(pointer(v), n)
    return nothing
end

@inline function coin_fill_n!(v::Vector{T}, n::Int, val::T)::Nothing where {T}
    GC.@preserve v coin_fill_n!(pointer(v), n, val)
    return nothing
end

@inline function coin_iota_n!(v::Vector{T}, n::Int, init::T = zero(T))::Nothing where {T}
    GC.@preserve v coin_iota_n!(pointer(v), n, init)
    return nothing
end

@inline function coin_is_sorted(v::Vector{T}, n::Int = length(v))::Bool where {T}
    return GC.@preserve v coin_is_sorted(pointer(v), n)
end
