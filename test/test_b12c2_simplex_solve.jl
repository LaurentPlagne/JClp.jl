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

    @testset "Warm-start bound transitions & dual KKT cleanliness" begin
        # 1. Hydraulic-style equality balance LP:
        #   row 1: -x1 + x2 + 3*x3 = 10
        #   row 2: -x2 - x4 + x5 + 2*x6 = 5
        #   row 3:  x1 + x4 + x7 + x8 = 12
        col_starts   = Int32[1, 3, 5, 6, 8, 9, 10, 11, 12]
        col_indices  = Int32[1, 3, 1, 2, 1, 2, 3, 2, 2, 3, 3]
        col_elements = Float64[-1.0, 1.0, 1.0, -1.0, 3.0, -1.0, 1.0, 1.0, 2.0, 1.0, 1.0]
        obj = Float64[2.0, 5.0, 1.0, 0.0, -3.0, 2.0, 1.0, 1.0]
        rhs = Float64[10.0, 5.0, 12.0]

        function dual_kkt_residual(model::JClp.ClpSimplexModel)
            max_res = 0.0
            for j in 1:model.ncols
                aty = 0.0
                for k in col_starts[j]:(col_starts[j + 1] - 1)
                    aty += col_elements[k] * model.dual_row_sol[col_indices[k]]
                end
                max_res = max(max_res, abs(model.reduced_cost[j] - (model.objective[j] - aty)))
            end
            return max_res
        end

        m = JClp.ClpSimplexModel()
        JClp.load_problem!(
            m, 3, 8,
            col_starts, col_indices, col_elements,
            Float64[4.0, 2.0, 0.0, 0.0, 0.0, 0.0, -1e30, -1e30],
            Float64[4.0, 8.0, 10.0, 10.0, 6.0, 10.0, 1e30, 1e30],
            obj, rhs, rhs,
        )

        # Cold solve
        @test JClp.dual!(m; warm_start=false) == JClp.CLP_STATUS_OPTIMAL
        @test m.objective_value ≈ 13.0 atol=1e-9
        @test dual_kkt_residual(m) < 1e-10
        @test JClp._sb_get_status(m.status[1]) == JClp.CLP_STATUS_IS_FIXED
        @test JClp._sb_get_status(m.status[2]) == JClp.CLP_STATUS_AT_LOWER
        @test JClp._sb_get_status(m.status[5]) == JClp.CLP_STATUS_AT_UPPER
        @test JClp._sb_get_status(m.status[8]) == JClp.CLP_STATUS_IS_FREE

        # 1a. Unfix previously IS_FIXED column (lo == hi -> lo < hi)
        JClp.set_col_bounds!(m, 1, 1.0, 6.0)
        @test JClp.dual!(m; warm_start=true) == JClp.CLP_STATUS_OPTIMAL
        @test m.objective_value ≈ 9.0 atol=1e-9
        @test dual_kkt_residual(m) < 1e-10

        # 1b. Fix a previously unfixed column (lo < hi -> lo == hi)
        JClp.set_col_bounds!(m, 1, 3.0, 3.0)
        @test JClp.dual!(m; warm_start=true) == JClp.CLP_STATUS_OPTIMAL
        @test m.objective_value ≈ 35.0 / 3.0 atol=1e-9
        @test dual_kkt_residual(m) < 1e-10

        # 1c. Relax an AT_LOWER bound to -1e30 and an AT_UPPER bound to +1e30
        @test JClp._sb_get_status(m.status[2]) == JClp.CLP_STATUS_AT_LOWER
        @test JClp._sb_get_status(m.status[5]) == JClp.CLP_STATUS_AT_UPPER
        JClp.set_col_bounds!(m, 2, -1e30, 8.0)
        JClp.set_col_bounds!(m, 5, 0.0, 1e30)
        @test JClp.dual!(m; warm_start=true) == JClp.CLP_STATUS_OPTIMAL
        @test m.objective_value ≈ -182.0 / 3.0 atol=1e-9
        @test dual_kkt_residual(m) < 1e-10

        # 1d. Bound a previously IS_FREE variable
        @test JClp._sb_get_status(m.status[8]) == JClp.CLP_STATUS_IS_FREE
        JClp.set_col_bounds!(m, 8, 1.0, 5.0)
        @test JClp.dual!(m; warm_start=true) == JClp.CLP_STATUS_OPTIMAL
        @test m.objective_value ≈ -182.0 / 3.0 atol=1e-9
        @test 1.0 - 1e-8 <= m.col_activity[8] <= 5.0 + 1e-8
        @test dual_kkt_residual(m) < 1e-10

        # 2. Infeasible LP solved via primal! must not leak 1e14 Phase-1 penalty weights into duals
        mi = JClp.ClpSimplexModel()
        JClp.load_problem!(
            mi, 2, 2,
            Int32[1, 3, 5], Int32[1, 2, 1, 2], Float64[1.0, 1.0, 1.0, 1.0],
            Float64[0.0, 0.0], Float64[1e30, 1e30], Float64[1.0, 1.0],
            Float64[-1e30, 5.0], Float64[2.0, 1e30],
        )
        @test JClp.primal!(mi) == JClp.CLP_STATUS_PRIMAL_INFEAS
        @test maximum(abs, mi.dual_row_sol) < 1e6
    end
end
