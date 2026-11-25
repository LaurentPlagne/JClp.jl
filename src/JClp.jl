module JClp

# Pure Idiomatic 1-based Julia Core (zero Ptr / unsafe_load / PodView)
include("coinutils/SparseLU/SparseLU.jl")
include("clp/PackedMatrix.jl")
include("clp/DualRowPivot.jl")
include("clp/PrimalColumnPivot.jl")
include("clp/NonLinearCost.jl")
include("clp/Factorization.jl")
include("clp/SimplexDual.jl")
include("clp/SimplexPrimal.jl")
include("clp/SimplexProgress.jl")
include("clp/SimplexBase.jl")
include("clp/DualLoop.jl")
include("clp/PrimalLoop.jl")
include("clp/MpsReader.jl")
include("clp/SimplexModel.jl")
include("clp/SimplexSolve.jl")
include("clp/Presolve.jl")

end # module
