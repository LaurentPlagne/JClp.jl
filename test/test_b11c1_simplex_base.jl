using Test
using JClp

@testset "B11c-1: ClpSimplexProgress & ClpSimplex autonomous kernels" begin
    @testset "ClpSimplexProgress reset & cycle" begin
        st = JClp.ClpSimplexProgressState()
        JClp.clp_progress_reset!(
            st.objective, st.infeasibility, st.real_infeasibility,
            st.number_infeasibilities, st.iteration_number,
            st.in_seq, st.out_seq, st.way, -1,
        )
        @test all(x -> x < -1e250, st.objective)
        @test all(==(Int32(-1)), st.in_seq)
        @test all(==(Int32(-1)), st.out_seq)
        @test all(==(Int8(0)), st.way)

        # Feed a clean period-2 cycle: (1,2), (2,1), (1,2), ...
        res = 0
        for iter in 1:15
            in_s = isodd(iter) ? Int32(1) : Int32(2)
            out_s = isodd(iter) ? Int32(2) : Int32(1)
            res = JClp.clp_progress_cycle!(st.in_seq, st.out_seq, st.way, in_s, out_s, 1, -1)
        end
        @test res == 2

        # Zero-allocation check on cycle
        function _alloc_cycle(in_a, out_a, way_a)
            JClp.clp_progress_cycle!(in_a, out_a, way_a, Int32(3), Int32(4), 1, 1)
            return @allocated JClp.clp_progress_cycle!(in_a, out_a, way_a, Int32(4), Int32(3), -1, 1)
        end
        @test _alloc_cycle(st.in_seq, st.out_seq, st.way) == 0
    end

    @testset "ClpSimplexProgress looping_step" begin
        st = JClp.ClpSimplexProgressState()
        JClp.clp_progress_reset!(
            st.objective, st.infeasibility, st.real_infeasibility,
            st.number_infeasibilities, st.iteration_number,
            st.in_seq, st.out_seq, st.way, -1,
        )
        ntimes = 0
        nbad = 0
        nrbad = 0
        ffac = 100
        # 10 calls with identical (obj, infeas, ninf) on distinct iterations -> triggers loop at call 10
        ret = -1
        act = 0
        for iter in 1:10
            ret, act, _, _, ntimes, nbad, nrbad, ffac, _ = JClp.clp_progress_looping_step!(
                st.objective, st.infeasibility, st.real_infeasibility,
                st.number_infeasibilities, st.iteration_number,
                st.in_seq, st.out_seq, st.way,
                -1, 42.0, 1.5, 0.0, 3, iter,
                10, 20, false, 0, 1,
                ntimes, nbad, nrbad, ffac,
            )
        end
        @test ret == -2
        @test act == JClp.CLP_LOOP_ACT_DUAL_TOL
        @test nbad == 1
        @test ffac == 1
    end

    @testset "clp_clean_status!" begin
        nrows, ncols = 2, 3
        status = UInt8[
            JClp.CLP_STATUS_BASIC,
            JClp.CLP_STATUS_BASIC,
            JClp.CLP_STATUS_SUPER_BASIC,
            JClp.CLP_STATUS_BASIC,
            JClp.CLP_STATUS_AT_LOWER,
        ]
        row_act = [0.0, 10.0 + 1e-9]
        row_lo  = [0.0, 0.0]
        row_hi  = [5.0, 10.0]
        col_act = [0.0, 1.0 - 1e-9, 5.0]
        col_lo  = [0.0, 0.0, -1e35]
        col_hi  = [2.0, 1.0,  1e35]
        JClp.clp_clean_status!(status, row_act, row_lo, row_hi, col_act, col_lo, col_hi, nrows, ncols, 1e-7)
        # Row 1 was basic (n_basic=1); Row 2 snapped to upper
        @test (status[4] & JClp.CLP_STATUS_MASK) == JClp.CLP_STATUS_BASIC
        @test (status[5] & JClp.CLP_STATUS_MASK) == JClp.CLP_STATUS_AT_UPPER
        @test row_act[2] == 10.0
        # Col 1 basic (n_basic=2 == nrows); Col 2 evicted from basis & snapped to upper; Col 3 isFree
        @test (status[1] & JClp.CLP_STATUS_MASK) == JClp.CLP_STATUS_BASIC
        @test (status[2] & JClp.CLP_STATUS_MASK) == JClp.CLP_STATUS_AT_UPPER
        @test col_act[2] == 1.0
        @test (status[3] & JClp.CLP_STATUS_MASK) == JClp.CLP_STATUS_IS_FREE
    end

    @testset "clp_check_both_solutions + zero alloc" begin
        solution = [0.0, 2.5, 5.0, -0.5, 1.0]
        cost     = [1.0, 2.0, 3.0, 0.0, 0.0]
        lower    = [0.0, 0.0, 0.0, 0.0, 0.0]
        upper    = [10.0, 5.0, 5.0, 10.0, 10.0]
        dj       = [-0.5, 0.01, 0.2, 0.0, 0.0]
        status   = UInt8[
            JClp.CLP_STATUS_AT_LOWER,
            JClp.CLP_STATUS_SUPER_BASIC,
            JClp.CLP_STATUS_AT_UPPER,
            JClp.CLP_STATUS_BASIC,
            JClp.CLP_STATUS_BASIC,
        ]
        res = JClp.clp_check_both_solutions(
            solution, cost, lower, upper, dj, status, 5,
            1e-7, 1e-7, 0.0, 0.0, -1, 0, 0, 0, 0.0, 1.0, 1.0,
        )
        obj_val, sum_p, sum_rp, num_p, sum_d, sum_rd, num_d, num_dnf, best_imp, mso, ff = res
        @test obj_val ≈ 20.0
        @test num_p == 1
        @test sum_p ≈ 0.5 - 1e-7
        @test num_d == 3  # col 1 (-0.5 at lower), col 2 (sb with dj=0.01), col 3 (+0.2 at upper)
        @test num_dnf == 3
        @test ff == 2     # col 2 is first superbasic with dj
        @test (mso & 8) == 0

        function _alloc_both(s, c, l, u, d, st)
            JClp.clp_check_both_solutions(s, c, l, u, d, st, 5, 1e-7, 1e-7, 0.0, 0.0, -1, 0, 0, 0, 0.0, 1.0, 1.0)
            return @allocated JClp.clp_check_both_solutions(s, c, l, u, d, st, 5, 1e-7, 1e-7, 0.0, 0.0, -1, 0, 0, 0, 0.0, 1.0, 1.0)
        end
        @test _alloc_both(solution, cost, lower, upper, dj, status) == 0
    end

    @testset "clp_check_primal_solution & clp_check_dual_solution" begin
        row_act = [1.0, -0.2]
        row_obj = [0.0, 0.0]
        row_lo  = [0.0, 0.0]
        row_hi  = [2.0, 2.0]
        col_act = [3.0, 0.0]
        col_obj = [2.0, 5.0]
        col_lo  = [0.0, 0.0]
        col_hi  = [2.0, 10.0]
        pvar    = Int32[1, 2]
        obj_v, sp, srp, np = JClp.clp_check_primal_solution(
            row_act, row_obj, row_lo, row_hi, col_act, col_obj, col_lo, col_hi, pvar,
            2, 2, 1e-7, 0.0, false, 0.0, 1.0, 1.0,
        )
        @test obj_v ≈ 6.0
        @test np == 2
        @test sp ≈ (0.2 - 1e-7) + (1.0 - 1e-7)

        col_dj = [0.0, -0.5]
        row_dj = [0.0, 0.0]
        status = UInt8[JClp.CLP_STATUS_BASIC, JClp.CLP_STATUS_AT_LOWER,
                       JClp.CLP_STATUS_BASIC, JClp.CLP_STATUS_BASIC]
        obj_d, sd, srd, nd, ndnf, bimp, ff = JClp.clp_check_dual_solution(
            col_act, col_obj, col_lo, col_hi, col_dj,
            row_act, row_obj, row_lo, row_hi, row_dj,
            status, 2, 2, 1e-7, 1e-7, 0.0, -1, 0, 0, 0.0, 1.0, 1.0,
        )
        @test obj_d ≈ 6.0
        @test nd == 1
        @test ndnf == 1
        @test sd ≈ 0.5 - 1e-7
        @test bimp ≈ 5.0
        @test ff == 0
    end

    @testset "clp_all_slack_basis! & clp_check_solution_internal!" begin
        col_act = zeros(4)
        status  = fill(JClp.CLP_STATUS_BASIC, 5)
        col_lo  = [1.0, -5.0, -1e25, -2.0]
        col_hi  = [5.0, -1.0,  1e25,  3.0]
        sol     = zeros(4)
        JClp.clp_all_slack_basis!(col_act, status, col_lo, col_hi, 4, sol, nothing, 1.0)
        @test col_act == [1.0, -1.0, 0.0, 0.0]
        @test sol == [1.0, -1.0, 0.0, 0.0]
        @test (status[1] & JClp.CLP_STATUS_MASK) == JClp.CLP_STATUS_AT_LOWER
        @test (status[2] & JClp.CLP_STATUS_MASK) == JClp.CLP_STATUS_AT_UPPER
        @test (status[3] & JClp.CLP_STATUS_MASK) == JClp.CLP_STATUS_IS_FREE
        @test (status[4] & JClp.CLP_STATUS_MASK) == JClp.CLP_STATUS_AT_LOWER

        # Optimal feasible solution check
        st2 = UInt8[JClp.CLP_STATUS_AT_LOWER, JClp.CLP_STATUS_BASIC, JClp.CLP_STATUS_AT_UPPER]
        obj_i, sp, np, sd, nd, pstat = JClp.clp_check_solution_internal!(
            st2, [0.0], [5.0], [0.0], [5.0],
            [1.0, 0.0], [0.0, 2.0], [0.0, 0.0], [10.0, 10.0], [1.0, 3.0],
            1, 2, 1e-7, 1e-7, 1e-7, 1.0, 0.0, 0.0,
        )
        @test obj_i ≈ 6.0
        @test np == 0
        @test nd == 0
        @test pstat == 0
    end
end
