using JClp
using Test
using Random

if !isdefined(@__MODULE__, :check_kkt_optimal)
    include("kkt_utils.jl")
end

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

    @testset "Dual perturbation (set_perturbation!, perturbation, KKT exactness & 0-alloc warm-start)" begin
        # 1. API getters/setters and persistence across load_problem! & presolve
        m_api = JClp.ClpSimplexModel()
        @test JClp.perturbation(m_api) == 50
        @test JClp.set_perturbation!(m_api, 100) == 100
        @test JClp.perturbation(m_api) == 100
        @test JClp.set_perturbation!(m_api, 102) == 102
        @test JClp.perturbation(m_api) == 102
        @test JClp.set_perturbation!(m_api, 50) == 50
        @test JClp.perturbation(m_api) == 50

        JClp.set_perturbation!(m_api, 102)
        JClp.load_problem!(
            m_api, 2, 2,
            Int32[1, 3, 5], Int32[1, 2, 1, 2], Float64[1.0, 2.0, 1.0, 1.0],
            Float64[0.0, 0.0], Float64[1e30, 1e30], Float64[-3.0, -2.0],
            Float64[-1e30, -1e30], Float64[4.0, 6.0],
        )
        @test JClp.perturbation(m_api) == 102
        ps_api = JClp.presolve(m_api)
        @test ps_api.reduced.perturbation == 102
        @test JClp.initial_solve!(m_api) == JClp.CLP_STATUS_OPTIMAL
        assert_kkt_optimal(m_api; tol = 1e-8)

        # 2. Synthetic dual-degenerate LP (tied costs & degenerate assignment structure)
        n_assign = 12
        nr_a = 2 * n_assign
        nc_a = n_assign * n_assign
        cs_a = Vector{Int32}(undef, nc_a + 1)
        ci_a = Vector{Int32}(undef, 2 * nc_a)
        ce_a = fill(1.0, 2 * nc_a)
        obj_a = Vector{Float64}(undef, nc_a)
        p_idx = 0
        for i in 1:n_assign, j in 1:n_assign
            col = (i - 1) * n_assign + j
            cs_a[col] = Int32(p_idx + 1)
            ci_a[p_idx + 1] = Int32(i)
            ci_a[p_idx + 2] = Int32(n_assign + j)
            p_idx += 2
            obj_a[col] = float(((i * 5 + j * 7) % 4) + 1)
        end
        cs_a[nc_a + 1] = Int32(p_idx + 1)
        clo_a = zeros(Float64, nc_a)
        chi_a = fill(1.0, nc_a)
        rhs_a = ones(Float64, nr_a)

        objs_assign = Float64[]
        for pert in (100, 50, 102)
            md = JClp.ClpSimplexModel()
            JClp.set_perturbation!(md, pert)
            JClp.load_problem!(md, nr_a, nc_a, cs_a, ci_a, ce_a, clo_a, chi_a, obj_a, rhs_a, rhs_a)
            @test JClp.dual!(md; max_iterations = 5000) == JClp.CLP_STATUS_OPTIMAL
            assert_kkt_optimal(md; tol = 1e-8)
            push!(objs_assign, md.objective_value)
        end
        @test objs_assign[1] ≈ objs_assign[2] atol=1e-9
        @test objs_assign[1] ≈ objs_assign[3] atol=1e-9

        # Synthetic tied-zero-cost sparse degenerate LP (min & max) + warm-start
        rng = MersenneTwister(77)
        nr_d = 40
        nc_d = 80
        cs_d = Int32[1]
        ci_d = Int32[]
        ce_d = Float64[]
        for j in 1:nc_d
            for r in sort(randperm(rng, nr_d)[1:4])
                push!(ci_d, Int32(r))
                push!(ce_d, float(rand(rng, (-2, -1, 1, 2))))
            end
            push!(cs_d, Int32(length(ci_d) + 1))
        end
        obj_d = [j <= div(nc_d, 2) ? 0.0 : float((j % 3) + 1) for j in 1:nc_d]
        clo_d = zeros(Float64, nc_d)
        chi_d = fill(5.0, nc_d)
        x_feas = [j <= div(nc_d, 3) ? 0.0 : 1.0 for j in 1:nc_d]
        ax_d = zeros(Float64, nr_d)
        for j in 1:nc_d, k in cs_d[j]:(cs_d[j + 1] - 1)
            ax_d[ci_d[k]] += ce_d[k] * x_feas[j]
        end
        rlo_d = [i <= div(nr_d, 2) ? ax_d[i] : -1e30 for i in 1:nr_d]
        rhi_d = [i <= div(nr_d, 2) ? ax_d[i] : ax_d[i] + (i % 2 == 0 ? 0.0 : 1.0) for i in 1:nr_d]

        objs_d = Float64[]
        for pert in (100, 50, 102)
            md = JClp.ClpSimplexModel()
            JClp.set_perturbation!(md, pert)
            JClp.load_problem!(md, nr_d, nc_d, cs_d, ci_d, ce_d, clo_d, chi_d, obj_d, rlo_d, rhi_d)
            @test JClp.dual!(md; max_iterations = 5000) == JClp.CLP_STATUS_OPTIMAL
            assert_kkt_optimal(md; tol = 1e-8)
            push!(objs_d, md.objective_value)

            # Warm-start bound perturbations under pert = 50 and pert = 102 (and 100)
            JClp.set_col_bounds!(md, 1, 0.25, 4.0)
            JClp.set_row_bounds!(md, nr_d, -1e30, rhi_d[nr_d] + 0.5)
            @test JClp.dual!(md; warm_start = true, max_iterations = 5000) == JClp.CLP_STATUS_OPTIMAL
            assert_kkt_optimal(md; tol = 1e-8)
        end
        @test objs_d[1] ≈ objs_d[2] atol=1e-8
        @test objs_d[1] ≈ objs_d[3] atol=1e-8

        # 3. Infeasible LP & max_iterations limit restore unperturbed costs cleanly
        m_inf = JClp.ClpSimplexModel()
        JClp.set_perturbation!(m_inf, 102)
        JClp.load_problem!(
            m_inf, 2, 2,
            Int32[1, 3, 5], Int32[1, 2, 1, 2], Float64[1.0, 1.0, 1.0, 1.0],
            Float64[0.0, 0.0], Float64[1e30, 1e30], Float64[1.0, 1.0],
            Float64[-1e30, 5.0], Float64[2.0, 1e30],
        )
        @test JClp.dual!(m_inf) == JClp.CLP_STATUS_PRIMAL_INFEAS

        # 4. e226.mps with perturbation = 50, 100, 102 (cold, warm-start, initial_solve!, max_iterations)
        sample_dir = "/Users/laurentplagne/Projects/Clp/parity/data/Sample"
        e226_path = joinpath(sample_dir, "e226.mps")
        if isfile(e226_path)
            e226_objs = Float64[]
            for pert in (100, 50, 102)
                me = JClp.ClpSimplexModel()
                JClp.set_perturbation!(me, pert)
                JClp.read_mps!(me, e226_path)
                @test JClp.dual!(me; max_iterations = 10000) == JClp.CLP_STATUS_OPTIMAL
                assert_kkt_optimal(me; tol = 1e-8)
                push!(e226_objs, me.objective_value)

                # Warm-start under perturbation = 50 / 102 / 100
                JClp.set_row_bounds!(me, 10, me.row_lower[10], me.row_upper[10] + 5.0)
                @test JClp.dual!(me; warm_start = true, max_iterations = 10000) == JClp.CLP_STATUS_OPTIMAL
                assert_kkt_optimal(me; tol = 1e-8)

                # Also test initial_solve! with kwarg perturbation
                me_init = JClp.ClpSimplexModel()
                JClp.read_mps!(me_init, e226_path)
                @test JClp.initial_solve!(me_init; perturbation = pert, max_iterations = 10000) == JClp.CLP_STATUS_OPTIMAL
                @test me_init.objective_value ≈ -11.6389290663705 rtol=1e-7
                assert_kkt_optimal(me_init; tol = 1e-8)
            end
            @test e226_objs[1] ≈ -11.6389290663705 rtol=1e-8
            @test e226_objs[1] ≈ e226_objs[2] atol=1e-9
            @test e226_objs[1] ≈ e226_objs[3] atol=1e-9

            # Hit MAX_ITERATIONS while perturbed -> costs restored cleanly
            me_lim = JClp.ClpSimplexModel()
            JClp.set_perturbation!(me_lim, 102)
            JClp.read_mps!(me_lim, e226_path)
            @test JClp.dual!(me_lim; max_iterations = 2) == JClp.CLP_STATUS_MAX_ITERATIONS
            @test me_lim.cost == me_lim.saved_cost
            @test all(isfinite, me_lim.dual_row_sol)
            @test all(isfinite, me_lim.reduced_cost)
        end

        # 5. Strictly 0 bytes allocated in dual!(m; warm_start=true) for perturbation = 100 and 50
        function _measure_ws_dual_alloc(pert::Int)
            m = JClp.ClpSimplexModel()
            JClp.set_perturbation!(m, pert)
            JClp.load_problem!(m, nr_d, nc_d, cs_d, ci_d, ce_d, clo_d, chi_d, obj_d, rlo_d, rhi_d)
            @test JClp.dual!(m; max_iterations = 5000) == JClp.CLP_STATUS_OPTIMAL
            JClp.set_col_bounds!(m, 1, 0.25, 4.0)
            JClp.set_row_bounds!(m, nr_d, -1e30, rhi_d[nr_d] + 0.5)
            @test JClp.dual!(m; warm_start = true, max_iterations = 5000) == JClp.CLP_STATUS_OPTIMAL
            JClp.set_col_bounds!(m, 1, 0.0, 5.0)
            JClp.set_row_bounds!(m, nr_d, -1e30, rhi_d[nr_d])
            return @allocated JClp.dual!(m; warm_start = true, max_iterations = 5000)
        end
        @test _measure_ws_dual_alloc(100) == 0
        @test _measure_ws_dual_alloc(50) == 0

        if isfile(e226_path)
            function _measure_e226_ws_alloc(pert::Int)
                m = JClp.ClpSimplexModel()
                JClp.set_perturbation!(m, pert)
                JClp.read_mps!(m, e226_path)
                @test JClp.dual!(m; max_iterations = 10000) == JClp.CLP_STATUS_OPTIMAL
                JClp.set_row_bounds!(m, 10, m.row_lower[10], m.row_upper[10] + 5.0)
                @test JClp.dual!(m; warm_start = true, max_iterations = 10000) == JClp.CLP_STATUS_OPTIMAL
                JClp.set_row_bounds!(m, 10, m.row_lower[10], m.row_upper[10] - 5.0)
                allocs = @allocated JClp.dual!(m; warm_start = true, max_iterations = 10000)
                @test m.problem_status == JClp.CLP_STATUS_OPTIMAL
                assert_kkt_optimal(m; tol = 1e-8)
                return allocs
            end
            @test _measure_e226_ws_alloc(100) == 0
            @test _measure_e226_ws_alloc(50) == 0
            @test _measure_e226_ws_alloc(102) == 0
        end

        # 6. Regression: portable pure-Julia LU (lu_lapack_dgetrf!/dgetrs!) + 0-alloc
        @testset "Portable pure-Julia dense LU (N/T + singular + 0-alloc)" begin
            A0 = Float64[0.5 3.0 1.0; 4.0 1.0 2.0; 2.0 5.0 0.2]
            a = copy(vec(A0))
            ipiv = zeros(Int32, 3)
            @test JClp.lu_lapack_dgetrf!(a, 3, ipiv) == 0
            b = Float64[1.0, 2.0, 3.0]
            xN = copy(b)
            JClp.lu_lapack_dgetrs!(UInt8('N'), a, 3, ipiv, xN, 1)
            @test maximum(abs.(A0 * xN - b)) < 1e-14
            xT = copy(b)
            JClp.lu_lapack_dgetrs!(UInt8('T'), a, 3, ipiv, xT, 1)
            @test maximum(abs.(A0' * xT - b)) < 1e-14
            # 0-byte allocation on both dgetrf! and dgetrs!
            a_work = copy(vec(A0))
            rhs_work = copy(b)
            JClp.lu_lapack_dgetrf!(a_work, 3, ipiv)
            JClp.lu_lapack_dgetrs!(UInt8('N'), a_work, 3, ipiv, rhs_work, 1)
            copyto!(a_work, vec(A0))
            @test (@allocated JClp.lu_lapack_dgetrf!(a_work, 3, ipiv)) == 0
            @test (@allocated JClp.lu_lapack_dgetrs!(UInt8('N'), a_work, 3, ipiv, rhs_work, 1)) == 0
            @test (@allocated JClp.lu_lapack_dgetrs!(UInt8('T'), a_work, 3, ipiv, rhs_work, 1)) == 0
            # Singular matrix returns info > 0
            a_sing = Float64[1.0, 2.0, 2.0, 4.0]
            ipiv2 = zeros(Int32, 2)
            @test JClp.lu_lapack_dgetrf!(a_sing, 2, ipiv2) == 2
        end

        # 7. Regression: iteration-0 distinct-cost heuristic (_sm_should_perturb_init!)
        @testset "Iteration-0 distinct-cost heuristic" begin
            m_rep = JClp.ClpSimplexModel()
            JClp.load_problem!(
                m_rep, 1, 8, Int32[1,2,3,4,5,6,7,8,9], Int32[1,1,1,1,1,1,1,1], ones(8),
                zeros(8), ones(8), Float64[0, 0, 0, 0, 0, 0, 1, 1], [0.5], [1.0],
            )
            @test JClp._sm_should_perturb_init!(m_rep) == true
            @test (@allocated JClp._sm_should_perturb_init!(m_rep)) == 0
            @test all(==(0.0), m_rep.c0_val)

            m_dist = JClp.ClpSimplexModel()
            JClp.load_problem!(
                m_dist, 1, 4, Int32[1,2,3,4,5], Int32[1,1,1,1], ones(4),
                zeros(4), ones(4), Float64[1, 2, 3, 4], [0.5], [1.0],
            )
            @test JClp._sm_should_perturb_init!(m_dist) == false
        end

        # 8. Regression: primal! / dual! on primal-infeasible LP with a free variable
        #    (clp_pcs_redo_infeasibilities! must skip flagged variables so primal! does not stall)
        @testset "Primal-infeasible LP with free variable does not stall" begin
            # x1 + x2 = 10, 0 <= x1, x2 <= 1 (infeasible), plus free variable x3 with cost -1
            m_inf = JClp.ClpSimplexModel()
            JClp.load_problem!(
                m_inf, 1, 3, Int32[1, 2, 3, 3], Int32[1, 1], Float64[1.0, 1.0],
                Float64[0.0, 0.0, -floatmax(Float64)], Float64[1.0, 1.0, floatmax(Float64)],
                Float64[0.0, 0.0, -1.0], Float64[10.0], Float64[10.0],
            )
            @test JClp.primal!(m_inf; max_iterations = 500) == JClp.CLP_STATUS_PRIMAL_INFEAS
            m_inf2 = JClp.ClpSimplexModel()
            JClp.load_problem!(
                m_inf2, 1, 3, Int32[1, 2, 3, 3], Int32[1, 1], Float64[1.0, 1.0],
                Float64[0.0, 0.0, -floatmax(Float64)], Float64[1.0, 1.0, floatmax(Float64)],
                Float64[0.0, 0.0, -1.0], Float64[10.0], Float64[10.0],
            )
            @test JClp.dual!(m_inf2; max_iterations = 500) == JClp.CLP_STATUS_PRIMAL_INFEAS
        end
    end
end
