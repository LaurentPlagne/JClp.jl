using JClp
using Test

@testset "B12c-1: ClpSimplexModel standalone core" begin
    @testset "2x2 LP load, non-trivial basis factorize, guts_of_solution!, delete_rim!" begin
        # min -3x1 - 2x2 s.t. x1 + x2 <= 4, 2x1 + x2 <= 6, x >= 0
        # Optimal basis: x1, x2 basic -> x1 = 2, x2 = 2, s1 = 4 (at upper), s2 = 6 (at upper), obj = -10
        # Dual: y1 = -1, y2 = -1 -> d_s1 = 0 + (-1) = -1 <= 0 (at upper -> dual feasible!), d_s2 = -1 <= 0
        m = JClp.ClpSimplexModel()
        JClp.load_problem!(
            m, 2, 2,
            Int32[1, 3, 5], Int32[1, 2, 1, 2], Float64[1.0, 2.0, 1.0, 1.0],
            Float64[0.0, 0.0], Float64[1e30, 1e30], Float64[-3.0, -2.0],
            Float64[-1e30, -1e30], Float64[4.0, 6.0];
            name = "toy2x2",
        )
        @test m.nrows == 2 && m.ncols == 2
        @test m.name == "toy2x2"

        # Set optimal basis (x1, x2 basic; s1, s2 at upper bound)
        m.status[1] = UInt8(JClp.CLP_STATUS_BASIC)
        m.status[2] = UInt8(JClp.CLP_STATUS_BASIC)
        m.status[3] = UInt8(JClp.CLP_STATUS_AT_UPPER)
        m.status[4] = UInt8(JClp.CLP_STATUS_AT_UPPER)

        JClp.create_rim!(m)
        @test m.has_row_copy
        @test JClp.internal_factorize!(m) == 0
        JClp.guts_of_solution!(m)

        @test m.largest_primal_err < 1e-12
        @test m.largest_dual_err < 1e-12
        @test m.num_primal_infeas == 0
        @test m.num_dual_infeas == 0
        @test m.solution[1] ≈ 2.0 atol=1e-12
        @test m.solution[2] ≈ 2.0 atol=1e-12
        @test m.dual_work[1] ≈ -1.0 atol=1e-12
        @test m.dual_work[2] ≈ -1.0 atol=1e-12
        @test m.objective_value ≈ -10.0 atol=1e-12

        JClp.delete_rim!(m)
        @test m.col_activity[1] ≈ 2.0 atol=1e-12
        @test m.col_activity[2] ≈ 2.0 atol=1e-12
        @test m.row_activity[1] ≈ 4.0 atol=1e-12
        @test m.row_activity[2] ≈ 6.0 atol=1e-12
        @test m.dual_row_sol[1] ≈ -1.0 atol=1e-12
        @test m.dual_row_sol[2] ≈ -1.0 atol=1e-12
        @test m.objective_value ≈ -10.0 atol=1e-12
    end

    @testset "Scaled 2x2 LP round-trip (equilibrium + geometric)" begin
        for sc_mode in (1, 3)
            m = JClp.ClpSimplexModel()
            JClp.load_problem!(
                m, 2, 2,
                Int32[1, 3, 5], Int32[1, 2, 1, 2], Float64[100.0, 200.0, 0.01, 0.01],
                Float64[0.0, 0.0], Float64[1e30, 1e30], Float64[-300.0, -0.02],
                Float64[-1e30, -1e30], Float64[4.0, 6.0];
                name = "scaled2x2",
            )
            @test JClp.scale_model!(m; mode=sc_mode)
            @test m.has_scale
            m.status[1] = UInt8(JClp.CLP_STATUS_BASIC)
            m.status[2] = UInt8(JClp.CLP_STATUS_BASIC)
            m.status[3] = UInt8(JClp.CLP_STATUS_AT_UPPER)
            m.status[4] = UInt8(JClp.CLP_STATUS_AT_UPPER)
            JClp.create_rim!(m)
            @test JClp.internal_factorize!(m) == 0
            JClp.guts_of_solution!(m)
            @test m.largest_primal_err < 1e-12
            @test m.largest_dual_err < 1e-12
            @test m.num_primal_infeas == 0
            @test m.num_dual_infeas == 0
            JClp.delete_rim!(m)
            # x1 = 0.02, x2 = 200.0 -> obj = -300*0.02 - 0.02*200 = -10.0
            @test m.col_activity[1] ≈ 0.02 atol=1e-11
            @test m.col_activity[2] ≈ 200.0 atol=1e-9
            @test m.objective_value ≈ -10.0 atol=1e-10
        end
    end

    @testset "MPS reader + scaling + factorize + guts_of_solution! on Sample/Netlib" begin
        sample_dir = "/Users/laurentplagne/Projects/Clp/parity/data/Sample"
        netlib_dir = "/Users/laurentplagne/Projects/Clp/parity/data/Netlib"
        if isdir(sample_dir)
            cases = [
                (joinpath(sample_dir, "afiro.mps"), 27, 32),
                (joinpath(sample_dir, "adlittle.mps"), 56, 97),
                (joinpath(sample_dir, "brandy.mps"), 220, 249),
                (joinpath(sample_dir, "e226.mps"), 223, 282),
            ]
            if isdir(netlib_dir)
                push!(cases, (joinpath(netlib_dir, "25fv47.mps"), 821, 1571))
            end
            for (path, exp_r, exp_c) in cases
                isfile(path) || continue
                m = JClp.ClpSimplexModel()
                JClp.read_mps!(m, path)
                @test m.nrows == exp_r
                @test m.ncols == exp_c
                JClp.scale_model!(m; mode=1)
                JClp.create_rim!(m)
                @test JClp.internal_factorize!(m) == 0
                JClp.guts_of_solution!(m)
                @test m.largest_primal_err < 1e-10
                @test m.largest_dual_err < 1e-10
                JClp.delete_rim!(m)
                @test isfinite(m.objective_value)
            end
        end
    end
end
