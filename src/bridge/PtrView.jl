# Zero-allocation, isbits AbstractVector view over a raw C buffer.
# Strictly confined to src/bridge/ so the algorithmic core in src/coinutils/ and src/clp/
# never touches Ptr, unsafe_load, or unsafe_store!.

struct PtrView{T} <: AbstractVector{T}
    ptr::Ptr{T}
    len::Int
end

@inline Base.size(v::PtrView) = (v.len,)
@inline Base.length(v::PtrView) = v.len
@inline Base.IndexStyle(::Type{<:PtrView}) = IndexLinear()
@inline Base.getindex(v::PtrView{T}, i::Int) where {T} = unsafe_load(v.ptr, i)
@inline function Base.setindex!(v::PtrView{T}, x, i::Int) where {T}
    unsafe_store!(v.ptr, convert(T, x), i)
    return v
end
@inline Base.pointer(v::PtrView{T}) where {T} = v.ptr
@inline Base.pointer(v::PtrView{T}, i::Integer) where {T} = v.ptr + (Int(i) - 1) * sizeof(T)

# Zero-allocation, isbits 1-based view over a 0-based C integer index/start buffer.
# Reading v[i] returns (c_val + 1); writing v[i] = x stores (x - 1).
# Allows pure 1-based Julia kernels in src/clp/ to operate directly on C++ CSC/CSR arrays
# with zero copy and zero Ptr exposure outside src/bridge/.
struct ShiftedPtrView{T<:Integer} <: AbstractVector{T}
    ptr::Ptr{T}
    len::Int
end

@inline Base.size(v::ShiftedPtrView) = (v.len,)
@inline Base.length(v::ShiftedPtrView) = v.len
@inline Base.IndexStyle(::Type{<:ShiftedPtrView}) = IndexLinear()
@inline Base.getindex(v::ShiftedPtrView{T}, i::Int) where {T} = unsafe_load(v.ptr, i) + one(T)
@inline function Base.setindex!(v::ShiftedPtrView{T}, x, i::Int) where {T}
    unsafe_store!(v.ptr, convert(T, x) - one(T), i)
    return v
end

# Sparse index 0-based <-> 1-based boundary shifts (only touches the `nnz` active indices)
@inline function shift_to_1based!(idx::PtrView{Int32}, n::Int)::Nothing
    @inbounds @simd for i in 1:n
        idx[i] = idx[i] + Int32(1)
    end
    return nothing
end

@inline function shift_to_0based!(idx::PtrView{Int32}, n::Int)::Nothing
    @inbounds @simd for i in 1:n
        idx[i] = idx[i] - Int32(1)
    end
    return nothing
end
