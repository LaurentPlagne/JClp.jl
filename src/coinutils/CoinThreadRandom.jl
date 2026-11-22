# Faithful translation of CoinUtils/src/CoinHelperFunctions.hpp: CoinThreadRandom
# Active build defines COIN_OWN_RANDOM_32 (32-bit unsigned LCG).

const COIN_RANDOM_DEFAULT_SEED = UInt32(12345678)
const COIN_RANDOM_MULT = UInt32(1664525)
const COIN_RANDOM_INC = UInt32(1013904223)
const COIN_RANDOM_INV_2P32 = 1.0 / 4294967296.0

mutable struct CoinThreadRandom
    seed::UInt32
end

CoinThreadRandom() = CoinThreadRandom(COIN_RANDOM_DEFAULT_SEED)
CoinThreadRandom(seed::Integer) = CoinThreadRandom(UInt32(seed % UInt32))

@inline function set_seed!(rng::CoinThreadRandom, seed::Integer)::Nothing
    rng.seed = UInt32(seed & 0xffffffff)
    return nothing
end

@inline function get_seed(rng::CoinThreadRandom)::UInt32
    return rng.seed
end

@inline function coin_random_step(seed::UInt32)::Tuple{UInt32,Float64}
    next_seed = COIN_RANDOM_MULT * seed + COIN_RANDOM_INC
    val = Float64(next_seed) / 4294967296.0
    return next_seed, val
end

@inline function random_double!(rng::CoinThreadRandom)::Float64
    next_seed, val = coin_random_step(rng.seed)
    rng.seed = next_seed
    return val
end

@inline function randomize!(rng::CoinThreadRandom, n::Int32 = Int32(0))::Nothing
    iters = n == Int32(0) ? Int32(rng.seed & UInt32(255)) : n
    i = Int32(0)
    while i < iters
        random_double!(rng)
        i += Int32(1)
    end
    return nothing
end

# Raw-pointer versions for zero-allocation C ABI calls from C++ CoinThreadRandom
@inline function coin_random_double_ptr!(seed_ptr::Ptr{UInt32})::Float64
    s = unsafe_load(seed_ptr)
    next_seed, val = coin_random_step(s)
    unsafe_store!(seed_ptr, next_seed)
    return val
end

@inline function coin_randomize_ptr!(seed_ptr::Ptr{UInt32}, n::Int32)::Nothing
    s = unsafe_load(seed_ptr)
    iters = n == Int32(0) ? Int32(s & UInt32(255)) : n
    i = Int32(0)
    while i < iters
        s = COIN_RANDOM_MULT * s + COIN_RANDOM_INC
        i += Int32(1)
    end
    unsafe_store!(seed_ptr, s)
    return nothing
end
