module JClp

# Phase A — CoinUtils (Bottom-Up)
include("coinutils/CoinFinite.jl")
include("coinutils/CoinThreadRandom.jl")
include("coinutils/CoinHelperFunctions.jl")
include("coinutils/CoinSort.jl")
include("coinutils/CoinIndexedVector.jl")
include("coinutils/CoinPackedMatrix.jl")
include("coinutils/CoinDenseFactorization.jl")
include("coinutils/CoinFactorizationPod.jl")
include("coinutils/CoinFactorizationBtran.jl")
include("coinutils/CoinFactorizationFtran.jl")
include("coinutils/CoinFactorizationReplace.jl")
include("coinutils/CoinFactorizationFactor.jl")
include("coinutils/CoinMpsIO.jl")
include("coinutils/CoinPresolvePod.jl")
include("coinutils/CoinPresolveBasic.jl")

# C ABI Bridge (@ccallable exports for JuliaC --trim=safe)
include("bridge/capi.jl")

end # module
