using JClp
using Test

@testset "B13: Pure-Julia Presolve, Postsolve & initial_solve!" begin
    @testset "Presolve reductions (fixed, singletons, forcing, doubleton, free col)" begin
        # min -3x1 - 2x2 + 5x3 + 0*x4
        # s.t. x3 == 2          (row singleton -> fixes x3 = 2)
        #      x1 + x2 + x3 <= 6  (after x3=2: x1 + x2 <= 4)
        #      2x1 + x2 <= 6
        #      x4 - x1 == 0     (doubleton eq -> x4 = x1, x4 free)
        #      x1, x2 >= 0, x3 >= 0, x4 free
        # Optimal: x1=2, x2=2, x3=2, x4=2, obj = -6 - 4 + 10 = 0
        m = JClp.ClpSimplexModel()
        JClp.load_problem!(
            m, 4, 4,
            Int32[1, 4, 6, 8, 9],
            Int32[2, 3, 4, 2, 3, 1, 2, 4],
            Float64[1.0, 2.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            Float64[0.0, 0.0, 0.0, -1e30],
            Float64[1e30, 1e30, 1e30, 1e30],
            Float64[-3.0, -2.0, 5.0, 0.0],
            Float64[2.0, -1e30, -1e30, 0.0],
            Float64[2.0, 6.0, 6.0, 0.0],
        )
        ps = JClp.presolve(m)
        @test ps.status == 0
        @test ps.red_nrows == 2
        @test ps.red_ncols == 2

        st = JClp.initial_solve!(m)
        @test st == JClp.CLP_STATUS_OPTIMAL
        @test m.objective_value ≈ 0.0 atol=1e-9
        @test m.col_activity[1] ≈ 2.0 atol=1e-9
        @test m.col_activity[2] ≈ 2.0 atol=1e-9
        @test m.col_activity[3] ≈ 2.0 atol=1e-9
        @test m.col_activity[4] ≈ 2.0 atol=1e-9
    end

    @testset "Presolve detects infeasibility immediately (0 simplex iterations)" begin
        # x1 + x2 <= 2, x1 >= 3, x2 >= 0 -> infeasible in presolve!
        m = JClp.ClpSimplexModel()
        JClp.load_problem!(
            m, 1, 2,
            Int32[1, 2, 3], Int32[1, 1], Float64[1.0, 1.0],
            Float64[3.0, 0.0], Float64[1e30, 1e30], Float64[1.0, 1.0],
            Float64[-1e30], Float64[2.0],
        )
        st = JClp.initial_solve!(m)
        @test st == JClp.CLP_STATUS_PRIMAL_INFEAS
        @test m.number_iterations == 0
    end

    @testset "initial_solve! on Sample MPS models matches exact Clp objectives" begin
        sample_dir = "/Users/laurentplagne/Projects/Clp/parity/data/Sample"
        if isdir(sample_dir)
            cases = [
                ("afiro.mps",  -464.7531428571429),
                ("brandy.mps", 1518.509896488128),
                ("e226.mps",    -11.6389290663705),
                ("finnis.mps", 172791.0655956116),
                ("p0201.mps",  6875.0),
                ("p0548.mps",   315.2549),
            ]
            for (fname, exp_obj) in cases
                path = joinpath(sample_dir, fname)
                isfile(path) || continue
                m = JClp.ClpSimplexModel()
                JClp.read_mps!(m, path)
                st = JClp.initial_solve!(m)
                @test st == JClp.CLP_STATUS_OPTIMAL
                @test m.objective_value ≈ exp_obj rtol=1e-5 atol=1e-4
            end
        end
    end
end
