using JClp
using Test

@testset "JClp.jl" begin
    include("test_sparse_lu_native.jl")
    include("test_b8_clp_packed_matrix.jl")
    include("test_b9a_dual_row_pivot.jl")
    include("test_b9b_primal_column_pivot.jl")
    include("test_b9c_nonlinear_cost.jl")
    include("test_b10_clp_factorization.jl")
    include("test_b11a_simplex_dual.jl")
    include("test_b11b_simplex_primal.jl")
    include("test_b11c1_simplex_base.jl")
    include("test_b11c2_simplex_base.jl")
    include("test_b11c3_simplex_base.jl")
    include("test_b12a_dual_loop.jl")
    include("test_b12b_primal_loop.jl")
    include("test_b12c1_simplex_model.jl")
    include("test_b12c2_simplex_solve.jl")
    include("test_b13_presolve.jl")
    include("test_b14_kkt_warmstart_fuzz.jl")
    include("test_b15_moi.jl")
end
