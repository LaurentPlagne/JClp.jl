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
include("coinutils/CoinPresolveMedium.jl")
include("coinutils/CoinPresolveAdvanced1.jl")
include("coinutils/CoinPresolveAdvanced2.jl")

# Pure Idiomatic 1-based Julia Core (zero Ptr / unsafe_load / PodView)
include("coinutils/SparseLU/SparseLU.jl")

# C ABI Bridge (@ccallable exports for JuliaC --trim=safe)
include("bridge/PtrView.jl")
include("bridge/handles.jl")
include("bridge/capi.jl")

end # module
