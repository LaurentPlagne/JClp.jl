using JClp
using Test

@testset "B12c-2: Standalone pure-Julia dual!, primal! & warm start" begin
    @testset "Toy 2x2 LP (min & max) with dual! and primal!" begin
        # min -3x1 - 2x2 s.t. x1 + x2 <= 4, 2x1 + x2 <= 6, x >= 0 -> opt x1=2, x2=2, obj=-10
        for solver in (JClp.dual!, JClp.primal!)
            m = JClp.ClpSimplexModel()
            JClp.load_problem!(
                m, 2, 2,
                Int32[1, 3, 5], Int32[1, 2, 1, 2], Float64[1.0, 2.0, 1.0, 1.0],
                Float64[0.0, 0.0], Float64[1e30, 1e30], Float64[-3.0, -2.0],
                Float64[-1e30, -1e30], Float64[4.0, 6.0],
            )
            st = solver(m)
            @test st == JClp.CLP_STATUS_OPTIMAL
            @test m.objective_value ≈ -10.0 atol=1e-9
            @test m.col_activity[1] ≈ 2.0 atol=1e-9
            @test m.col_activity[2] ≈ 2.0 atol=1e-9
        end

        # Maximization: max 3x1 + 2x2 s.t. same constraints -> obj = +10
        for solver in (JClp.dual!, JClp.primal!)
            m = JClp.ClpSimplexModel()
            JClp.load_problem!(
                m, 2, 2,
                Int32[1, 3, 5], Int32[1, 2, 1, 2], Float64[1.0, 2.0, 1.0, 1.0],
                Float64[0.0, 0.0], Float64[1e30, 1e30], Float64[3.0, 2.0],
                Float64[-1e30, -1e30], Float64[4.0, 6.0];
                obj_sense = -1.0,
            )
            st = solver(m)
            @test st == JClp.CLP_STATUS_OPTIMAL
            @test m.objective_value ≈ 10.0 atol=1e-9
            @test m.col_activity[1] ≈ 2.0 atol=1e-9
            @test m.col_activity[2] ≈ 2.0 atol=1e-9
        end
    end

    @testset "Infeasible and unbounded LPs" begin
        # Infeasible: x1 + x2 <= 2, x1 + x2 >= 5, x >= 0
        for solver in (JClp.dual!, JClp.primal!)
            m = JClp.ClpSimplexModel()
            JClp.load_problem!(
                m, 2, 2,
                Int32[1, 3, 5], Int32[1, 2, 1, 2], Float64[1.0, 1.0, 1.0, 1.0],
                Float64[0.0, 0.0], Float64[1e30, 1e30], Float64[1.0, 1.0],
                Float64[-1e30, 5.0], Float64[2.0, 1e30],
            )
            @test solver(m) == JClp.CLP_STATUS_PRIMAL_INFEAS
        end

        # Unbounded: min -x1 - x2 s.t. x1 - x2 <= 1, x >= 0
        for solver in (JClp.dual!, JClp.primal!)
            m = JClp.ClpSimplexModel()
            JClp.load_problem!(
                m, 1, 2,
                Int32[1, 2, 3], Int32[1, 1], Float64[1.0, -1.0],
                Float64[0.0, 0.0], Float64[1e30, 1e30], Float64[-1.0, -1.0],
                Float64[-1e30], Float64[1.0],
            )
            @test solver(m) == JClp.CLP_STATUS_DUAL_INFEAS
        end
    end

    @testset "Warm start (RHS perturbation -> dual!, cost perturbation -> primal!)" begin
        sample_dir = "/Users/laurentplagne/Projects/Clp/parity/data/Sample"
        afiro_path = joinpath(sample_dir, "afiro.mps")
        if isfile(afiro_path)
            m = JClp.ClpSimplexModel()
            JClp.read_mps!(m, afiro_path)
            @test JClp.dual!(m) == JClp.CLP_STATUS_OPTIMAL
            cold_iters = m.number_iterations
            @test m.objective_value ≈ -464.7531428571429 rtol=1e-7

            # Re-solve unchanged with warm start -> 0 iterations!
            @test JClp.dual!(m; warm_start=true) == JClp.CLP_STATUS_OPTIMAL
            @test m.number_iterations == 0
            @test m.objective_value ≈ -464.7531428571429 rtol=1e-7

            # Perturb an objective coefficient slightly and warm-start primal!
            old_c = m.objective[1]
            JClp.set_obj_coeff!(m, 1, old_c + 0.01)
            @test JClp.primal!(m; warm_start=true) == JClp.CLP_STATUS_OPTIMAL
            @test m.number_iterations <= cold_iters
        end
    end

    @testset "Sample MPS suite solved 100% pure Julia (dual! & primal!)" begin
        sample_dir = "/Users/laurentplagne/Projects/Clp/parity/data/Sample"
        if isdir(sample_dir)
            # Known exact Clp optimal objectives on Sample MPS models
            cases = [
                ("afiro.mps",  -464.7531428571429),
                ("brandy.mps", 1518.509896488128),
                ("e226.mps",    -11.6389290663705),
                ("finnis.mps", 172791.0655956116),
                ("lseu.mps",    834.6823529411765),
                ("p0033.mps",  2520.571739130435),
                ("p0201.mps",  6875.0),
                ("p0548.mps",   315.2549),
            ]
            for (fname, exp_obj) in cases
                path = joinpath(sample_dir, fname)
                isfile(path) || continue
                m_d = JClp.ClpSimplexModel()
                JClp.read_mps!(m_d, path)
                st_d = JClp.dual!(m_d; max_iterations=5000)
                @test st_d == JClp.CLP_STATUS_OPTIMAL
                @test m_d.objective_value ≈ exp_obj rtol=1e-5 atol=1e-4

                m_p = JClp.ClpSimplexModel()
                JClp.read_mps!(m_p, path)
                st_p = JClp.primal!(m_p; max_iterations=5000)
                @test st_p == JClp.CLP_STATUS_OPTIMAL
                @test m_p.objective_value ≈ exp_obj rtol=1e-5 atol=1e-4
            end
        end
    end
end
