using JClp
using Test
using Random

if !isdefined(@__MODULE__, :check_kkt_optimal)
    include("kkt_utils.jl")
end

function _clone_model_data(src::JClp.ClpSimplexModel; scaling::Int = src.scaling_flag)::JClp.ClpSimplexModel
    dst = JClp.ClpSimplexModel()
    dst.scaling_flag = scaling
    dst.primal_tol = src.primal_tol
    dst.dual_tol = src.dual_tol
    JClp.load_problem!(
        dst,
        src.nrows,
        src.ncols,
        src.mat_starts,
        src.mat_indices,
        src.mat_elements,
        src.col_lower,
        src.col_upper,
        src.objective,
        src.row_lower,
        src.row_upper;
        obj_sense = src.obj_sense,
        obj_offset = src.obj_offset,
        name = src.name,
    )
    return dst
end

function _build_random_structured_lp(
    rng::AbstractRNG,
    nrows::Int,
    ncols::Int;
    obj_sense::Float64 = 1.0,
    kind::Symbol = :optimal,
)
    # Build sparse CSC matrix with ~3-5 nonzeros per column, ensuring every row has entries
    col_rows = [Int32[] for _ in 1:ncols]
    col_vals = [Float64[] for _ in 1:ncols]
    row_has = falses(nrows)

    for j in 1:ncols
        deg = min(nrows, rand(rng, 2:4))
        rows = randperm(rng, nrows)[1:deg]
        sort!(rows)
        for r in rows
            val = round(rand(rng, (-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 3.0)) * (0.5 + rand(rng)), digits=3)
            val == 0.0 && (val = 1.0)
            push!(col_rows[j], Int32(r))
            push!(col_vals[j], val)
            row_has[r] = true
        end
    end
    for r in 1:nrows
        if !row_has[r]
            j = rand(rng, 1:ncols)
            push!(col_rows[j], Int32(r))
            push!(col_vals[j], 1.0)
        end
    end
    # Ensure deduplicated sorted row indices per col
    col_starts = Vector{Int32}(undef, ncols + 1)
    col_indices = Int32[]
    col_elements = Float64[]
    pos = 1
    for j in 1:ncols
        col_starts[j] = Int32(pos)
        perm = sortperm(col_rows[j])
        last_r = Int32(0)
        for p in perm
            r = col_rows[j][p]
            v = col_vals[j][p]
            if r == last_r
                col_elements[end] += v
            else
                push!(col_indices, r)
                push!(col_elements, v)
                last_r = r
                pos += 1
            end
        end
    end
    col_starts[ncols + 1] = Int32(pos)

    # Column bounds: mix fixed, negative-bounded, bounded, lower-only, upper-only, free
    col_lo = Vector{Float64}(undef, ncols)
    col_hi = Vector{Float64}(undef, ncols)
    x_ref = Vector{Float64}(undef, ncols)
    for j in 1:ncols
        r = rand(rng)
        if kind == :unbounded && j == 1
            col_lo[j] = 0.0
            col_hi[j] = 1.0e30
            x_ref[j] = 1.0
        elseif r < 0.10
            v = round(rand(rng) * 4.0 - 1.5, digits=2)
            col_lo[j] = v
            col_hi[j] = v
            x_ref[j] = v
        elseif r < 0.25
            lo = round(-1.0 - rand(rng) * 4.0, digits=2)
            hi = round(lo + 1.0 + rand(rng) * 6.0, digits=2)
            col_lo[j] = lo
            col_hi[j] = hi
            x_ref[j] = 0.5 * (lo + hi)
        elseif r < 0.45
            lo = round(rand(rng) * 2.0, digits=2)
            hi = round(lo + 2.0 + rand(rng) * 8.0, digits=2)
            col_lo[j] = lo
            col_hi[j] = hi
            x_ref[j] = lo + 0.3 * (hi - lo)
        elseif r < 0.65
            lo = round(rand(rng) * 2.0 - 1.0, digits=2)
            col_lo[j] = lo
            col_hi[j] = 1.0e30
            x_ref[j] = lo + 1.5
        elseif r < 0.85
            hi = round(rand(rng) * 6.0 - 1.0, digits=2)
            col_lo[j] = -1.0e30
            col_hi[j] = hi
            x_ref[j] = hi - 1.5
        else
            # Free variable only if not forcing boundedness, else wide finite box
            if kind == :optimal && rand(rng) < 0.5
                col_lo[j] = -1.0e30
                col_hi[j] = 1.0e30
                x_ref[j] = round(rand(rng) * 4.0 - 2.0, digits=2)
            else
                col_lo[j] = -20.0
                col_hi[j] = 20.0
                x_ref[j] = 0.0
            end
        end
    end

    # Compute A * x_ref
    ax_ref = zeros(Float64, nrows)
    for j in 1:ncols
        for k in col_starts[j]:(col_starts[j + 1] - 1)
            ax_ref[col_indices[k]] += col_elements[k] * x_ref[j]
        end
    end

    # Row bounds: mix equality, <=, >=, ranged
    row_lo = Vector{Float64}(undef, nrows)
    row_hi = Vector{Float64}(undef, nrows)
    for i in 1:nrows
        r = rand(rng)
        b = round(ax_ref[i], digits=4)
        if r < 0.35
            row_lo[i] = b
            row_hi[i] = b
        elseif r < 0.65
            row_lo[i] = -1.0e30
            row_hi[i] = b + round(rand(rng) * 4.0, digits=2)
        elseif r < 0.85
            row_lo[i] = b - round(rand(rng) * 4.0, digits=2)
            row_hi[i] = 1.0e30
        else
            row_lo[i] = b - round(rand(rng) * 3.0, digits=2)
            row_hi[i] = b + round(rand(rng) * 3.0 + 0.5, digits=2)
        end
    end

    # Costs: mix zero, positive, negative; ensure bounded dual when kind == :optimal
    obj = Vector{Float64}(undef, ncols)
    if kind == :optimal
        # Choose random dual y_ref respecting row signs so c = A' y_ref + rc_ref is dual feasible!
        y_ref = zeros(Float64, nrows)
        for i in 1:nrows
            if row_lo[i] == row_hi[i]
                y_ref[i] = round(randn(rng) * 2.0, digits=2)
            elseif row_lo[i] > -1.0e20 && row_hi[i] >= 1.0e20
                y_ref[i] = round(rand(rng) * 2.0, digits=2) * obj_sense
            elseif row_hi[i] < 1.0e20 && row_lo[i] <= -1.0e20
                y_ref[i] = -round(rand(rng) * 2.0, digits=2) * obj_sense
            else
                y_ref[i] = round((rand(rng) - 0.5) * 2.0, digits=2)
            end
        end
        for j in 1:ncols
            aty = 0.0
            for k in col_starts[j]:(col_starts[j + 1] - 1)
                aty += col_elements[k] * y_ref[col_indices[k]]
            end
            if col_lo[j] <= -1.0e20 && col_hi[j] >= 1.0e20
                rc_min = 0.0
            elseif col_lo[j] <= -1.0e20
                rc_min = -round(rand(rng) * 3.0, digits=2)
            elseif col_hi[j] >= 1.0e20
                rc_min = round(rand(rng) * 3.0, digits=2)
            else
                rc_min = round(randn(rng) * 3.0, digits=2)
            end
            if rand(rng) < 0.15
                obj[j] = 0.0
                # If col is bounded on both sides, 0.0 cost stays bounded; if one-sided/free, perturb only if needed
                if col_lo[j] <= -1.0e20 || col_hi[j] >= 1.0e20
                    obj[j] = round(aty + obj_sense * rc_min, digits=3)
                end
            else
                obj[j] = round(aty + obj_sense * rc_min, digits=3)
            end
        end
    else
        for j in 1:ncols
            obj[j] = rand(rng) < 0.2 ? 0.0 : round(randn(rng) * 5.0, digits=2)
        end
    end

    if kind == :infeasible
        # Make row 1 strictly contradictory with row 2 or column bounds
        if rand(rng) < 0.5
            row_lo[1] = 100.0
            row_hi[1] = 50.0
        else
            # Contradictory equality on bounded cols: force sum of row 1 far above max_act
            max_act = 0.0
            for j in 1:ncols
                col_lo[j] = 0.0
                col_hi[j] = 2.0
                for k in col_starts[j]:(col_starts[j + 1] - 1)
                    if col_indices[k] == 1
                        max_act += abs(col_elements[k]) * 2.0
                    end
                end
            end
            row_lo[1] = max_act + 10.0
            row_hi[1] = max_act + 10.0
        end
    elseif kind == :unbounded
        # Make col 1 free of upper bound and strictly improving, zeroing its entries in bounded rows or making its ray feasible
        # Simple guaranteed unbounded ray along col 1: all entries of col 1 >= 0 in <= rows?
        # Even cleaner: replace col 1 coefficients so A[:,1] == 0 except in one >= row with coeff +1, col_lo[1]=0, col_hi[1]=1e30, obj[1] = -10*obj_sense
        for k in col_starts[1]:(col_starts[1 + 1] - 1)
            r = col_indices[k]
            col_elements[k] = 0.0
            if k == col_starts[1]
                col_elements[k] = 1.0
                row_lo[r] = -1.0e30
                row_hi[r] = 1.0e30
            end
        end
        col_lo[1] = 0.0
        col_hi[1] = 1.0e30
        obj[1] = -10.0 * obj_sense
    end

    m = JClp.ClpSimplexModel()
    JClp.load_problem!(
        m, nrows, ncols,
        col_starts, col_indices, col_elements,
        col_lo, col_hi, obj, row_lo, row_hi;
        obj_sense = obj_sense,
    )
    return m
end

function _build_flow_balance_lp(rng::AbstractRNG; nrows::Int = 20, ncols::Int = 40)
    # 20 rows: 12 equality balance nodes, 5 <= capacity rows, 3 >= / ranged rows
    # 40 cols: network-style arcs + production/external variables
    col_rows = [Int32[] for _ in 1:ncols]
    col_vals = [Float64[] for _ in 1:ncols]

    # First 24 cols: balance arcs between nodes 1..12 with varying gains, plus optional capacity row
    for j in 1:24
        u = ((j - 1) % 12) + 1
        v = ((j + 2) % 12) + 1
        u == v && (v = (u % 12) + 1)
        r1, r2 = min(u, v), max(u, v)
        push!(col_rows[j], Int32(r1)); push!(col_vals[j], -round(0.8 + 0.4 * rand(rng), digits=2))
        push!(col_rows[j], Int32(r2)); push!(col_vals[j], round(0.8 + 0.4 * rand(rng), digits=2))
        if j <= 18
            cap_r = 13 + (j % 5)
            push!(col_rows[j], Int32(cap_r)); push!(col_vals[j], round(0.5 + rand(rng), digits=2))
        end
        if j % 3 == 0
            req_r = 18 + (j % 3)
            push!(col_rows[j], Int32(req_r)); push!(col_vals[j], round(0.5 + rand(rng), digits=2))
        end
    end
    # Next 12 cols: slack/supply/demand injection at each of the 12 balance nodes
    for k in 1:12
        j = 24 + k
        push!(col_rows[j], Int32(k)); push!(col_vals[j], 1.0)
        if k <= 5
            push!(col_rows[j], Int32(12 + k)); push!(col_vals[j], 0.5)
        end
    end
    # Last 4 cols: global coupling variables across capacity and requirement rows
    for k in 1:4
        j = 36 + k
        for r in (13 + (k % 5), 18 + (k % 3), (k % 12) + 1)
            push!(col_rows[j], Int32(r))
            push!(col_vals[j], round((rand(rng) < 0.5 ? 1.0 : -1.0) * (0.5 + rand(rng)), digits=2))
        end
    end

    col_starts = Vector{Int32}(undef, ncols + 1)
    col_indices = Int32[]
    col_elements = Float64[]
    pos = 1
    for j in 1:ncols
        col_starts[j] = Int32(pos)
        perm = sortperm(col_rows[j])
        last_r = Int32(0)
        for p in perm
            r = col_rows[j][p]
            v = col_vals[j][p]
            if r == last_r
                col_elements[end] += v
            else
                push!(col_indices, r)
                push!(col_elements, v)
                last_r = r
                pos += 1
            end
        end
    end
    col_starts[ncols + 1] = Int32(pos)

    col_lo = Vector{Float64}(undef, ncols)
    col_hi = Vector{Float64}(undef, ncols)
    x_ref = Vector{Float64}(undef, ncols)
    for j in 1:ncols
        if j <= 24
            if j % 6 == 0
                # Fixed flow
                v = round(1.0 + rand(rng) * 3.0, digits=2)
                col_lo[j] = v; col_hi[j] = v; x_ref[j] = v
            elseif j % 4 == 0
                # Bidirectional (negative to positive)
                col_lo[j] = -5.0; col_hi[j] = 5.0; x_ref[j] = 1.0
            else
                col_lo[j] = 0.0; col_hi[j] = 10.0; x_ref[j] = 2.0
            end
        elseif j <= 36
            if j % 5 == 0
                # Free supply/demand variable
                col_lo[j] = -1.0e30; col_hi[j] = 1.0e30; x_ref[j] = 0.5
            else
                col_lo[j] = -4.0; col_hi[j] = 12.0; x_ref[j] = 1.0
            end
        else
            col_lo[j] = 0.0; col_hi[j] = 8.0; x_ref[j] = 1.5
        end
    end

    ax_ref = zeros(Float64, nrows)
    for j in 1:ncols
        for k in col_starts[j]:(col_starts[j + 1] - 1)
            ax_ref[col_indices[k]] += col_elements[k] * x_ref[j]
        end
    end

    row_lo = Vector{Float64}(undef, nrows)
    row_hi = Vector{Float64}(undef, nrows)
    for i in 1:12
        b = round(ax_ref[i], digits=4)
        row_lo[i] = b; row_hi[i] = b
    end
    for i in 13:17
        b = round(ax_ref[i], digits=4)
        row_lo[i] = -1.0e30; row_hi[i] = b + 5.0
    end
    for i in 18:20
        b = round(ax_ref[i], digits=4)
        if i == 18
            row_lo[i] = b - 3.0; row_hi[i] = 1.0e30
        else
            row_lo[i] = b - 2.0; row_hi[i] = b + 4.0
        end
    end

    obj = Float64[
        j % 7 == 0 ? 0.0 :
        (j in 25:36 && col_lo[j] <= -1.0e20 ? 2.0 : round(rand(rng) * 6.0 - 2.0, digits=2))
        for j in 1:ncols
    ]

    m = JClp.ClpSimplexModel()
    JClp.load_problem!(
        m, nrows, ncols,
        col_starts, col_indices, col_elements,
        col_lo, col_hi, obj, row_lo, row_hi;
        obj_sense = 1.0,
        name = "flow_balance_20x40",
    )
    return m
end

@testset "B14: Universal KKT Checker & Warm-Start / Multi-Mode Parity Fuzzer" begin
    @testset "1. Cross-mode parity (dual!, primal!, initial_solve!) across sense & scaling" begin
        rng = MersenneTwister(20261125)
        for inst in 1:30
            kind = inst <= 24 ? :optimal : (inst <= 27 ? :infeasible : :unbounded)
            nrows = rand(rng, 8:20)
            ncols = rand(rng, 12:35)
            base_min = _build_random_structured_lp(rng, nrows, ncols; obj_sense = 1.0, kind = kind)

            base_min.obj_offset = (inst % 5 == 0) ? round(randn(rng) * 7.5, digits=2) : 0.0
            if kind == :infeasible && inst == 27
                # Pure linear infeasibility (all individual bounds valid lo <= hi)
                for j in 1:ncols
                    base_min.col_lower[j] = 0.0
                    base_min.col_upper[j] = 1.0
                end
                base_min.row_lower[1] = 1.0e4
                base_min.row_upper[1] = 1.0e4
            end

            for sense in (1.0, -1.0), scaling in (0, 1)
                m_d = _clone_model_data(base_min; scaling = scaling)
                m_p = _clone_model_data(base_min; scaling = scaling)
                m_i = _clone_model_data(base_min; scaling = scaling)
                for mm in (m_d, m_p, m_i)
                    mm.obj_sense = sense
                    if sense == -1.0 && kind == :optimal
                        # Keep same cost vector (testing both min and max on it)
                    elseif sense == -1.0 && kind == :unbounded
                        mm.objective[1] = -10.0 * sense
                    end
                end

                st_d = JClp.dual!(m_d; scaling = scaling, max_iterations = 5000)
                st_p = JClp.primal!(m_p; scaling = scaling, max_iterations = 5000)
                st_i = JClp.initial_solve!(m_i; scaling = scaling, max_iterations = 5000)

                @test st_d == st_p
                @test st_d == st_i
                if kind == :infeasible
                    @test st_d == JClp.CLP_STATUS_PRIMAL_INFEAS
                elseif kind == :unbounded
                    @test st_d == JClp.CLP_STATUS_DUAL_INFEAS
                end

                if st_d == JClp.CLP_STATUS_OPTIMAL
                    @test m_d.objective_value ≈ m_p.objective_value rtol=1e-6 atol=1e-6
                    @test m_d.objective_value ≈ m_i.objective_value rtol=1e-6 atol=1e-6
                    assert_kkt_optimal(m_d; tol = 1.0e-6)
                    assert_kkt_optimal(m_p; tol = 1.0e-6)
                    assert_kkt_optimal(m_i; tol = 1.0e-6)
                elseif st_d == JClp.CLP_STATUS_PRIMAL_INFEAS
                    @test maximum(abs, m_p.dual_row_sol; init = 0.0) < 1.0e6
                    @test maximum(abs, m_d.dual_row_sol; init = 0.0) < 1.0e6
                end
            end
        end

        # 0-row LP and presolve-to-0-row LP across min/max and all 3 solvers
        for sense in (1.0, -1.0), scaling in (0, 1)
            m0 = JClp.ClpSimplexModel()
            JClp.load_problem!(
                m0, 0, 3, Int32[1, 1, 1, 1], Int32[], Float64[],
                Float64[-2.0, 1.0, 3.0], Float64[4.0, 5.0, 3.0], Float64[2.0, -3.0, 7.0],
                Float64[], Float64[]; obj_sense = sense, obj_offset = 0.5,
            )
            for solver in (JClp.dual!, JClp.primal!, JClp.initial_solve!)
                mc = _clone_model_data(m0; scaling = scaling)
                @test solver(mc; scaling = scaling) == JClp.CLP_STATUS_OPTIMAL
                assert_kkt_optimal(mc; tol = 1.0e-6)
            end

            mp0 = JClp.ClpSimplexModel()
            JClp.load_problem!(
                mp0, 2, 4,
                Int32[1, 2, 3, 3, 4], Int32[1, 2, 1], Float64[2.0, -3.0, 1.0],
                Float64[0.0, 0.0, -5.0, 3.0], Float64[10.0, 10.0, 5.0, 3.0],
                Float64[4.0, -2.0, 0.0, 1.0],
                Float64[2.0, -15.0], Float64[1.0e30, 1.0e30];
                obj_sense = sense, obj_offset = 1.5,
            )
            mds = _clone_model_data(mp0; scaling = scaling)
            mps = _clone_model_data(mp0; scaling = scaling)
            mis = _clone_model_data(mp0; scaling = scaling)
            @test JClp.dual!(mds; scaling = scaling) == JClp.CLP_STATUS_OPTIMAL
            @test JClp.primal!(mps; scaling = scaling) == JClp.CLP_STATUS_OPTIMAL
            @test JClp.initial_solve!(mis; scaling = scaling) == JClp.CLP_STATUS_OPTIMAL
            @test mds.objective_value ≈ mps.objective_value rtol=1e-6 atol=1e-6
            @test mds.objective_value ≈ mis.objective_value rtol=1e-6 atol=1e-6
            assert_kkt_optimal(mds; tol = 1.0e-6)
            assert_kkt_optimal(mps; tol = 1.0e-6)
            assert_kkt_optimal(mis; tol = 1.0e-6)
        end

        # Sample MPS cross-mode parity & KKT (min & negated max, scaling 0 & 1)
        sample_dir = "/Users/laurentplagne/Projects/Clp/parity/data/Sample"
        if isdir(sample_dir)
            for fname in ("afiro.mps", "brandy.mps", "e226.mps", "finnis.mps", "lseu.mps", "p0033.mps", "p0201.mps", "p0548.mps")
                path = joinpath(sample_dir, fname)
                isfile(path) || continue
                m_base = JClp.ClpSimplexModel()
                JClp.read_mps!(m_base, path)
                for sense in (1.0, -1.0), scaling in (0, 1)
                    md = _clone_model_data(m_base; scaling = scaling)
                    mp = _clone_model_data(m_base; scaling = scaling)
                    mi = _clone_model_data(m_base; scaling = scaling)
                    if sense == -1.0
                        for mm in (md, mp, mi)
                            mm.obj_sense = -1.0
                            mm.objective .= .-mm.objective
                            mm.obj_offset = -mm.obj_offset
                        end
                    end
                    @test JClp.dual!(md; scaling = scaling, max_iterations = 10000) == JClp.CLP_STATUS_OPTIMAL
                    @test JClp.primal!(mp; scaling = scaling, max_iterations = 10000) == JClp.CLP_STATUS_OPTIMAL
                    @test JClp.initial_solve!(mi; scaling = scaling, max_iterations = 10000) == JClp.CLP_STATUS_OPTIMAL
                    @test md.objective_value ≈ mp.objective_value rtol=1e-5 atol=1e-4
                    @test md.objective_value ≈ mi.objective_value rtol=1e-5 atol=1e-4
                    assert_kkt_optimal(md; tol = 1.0e-5)
                    assert_kkt_optimal(mp; tol = 1.0e-5)
                    assert_kkt_optimal(mi; tol = 1.0e-5)
                end
            end
        end
    end

    @testset "2. Warm-start mutation fuzzer (120 dual! + 40 primal! sequential steps)" begin
        rng = MersenneTwister(987654321)
        m = _build_flow_balance_lp(rng; nrows = 20, ncols = 40)
        base_clo = copy(m.col_lower)
        base_chi = copy(m.col_upper)
        base_rlo = copy(m.row_lower)
        base_rhi = copy(m.row_upper)
        base_obj = copy(m.objective)

        st0 = JClp.dual!(m; warm_start = false)
        @test st0 == JClp.CLP_STATUS_OPTIMAL
        assert_kkt_optimal(m; tol = 1.0e-6)

        infeas_induced = false

        for step in 1:160
            use_primal = (step > 120)
            solver = use_primal ? JClp.primal! : JClp.dual!

            if infeas_induced
                # Restore feasible bounds cleanly and continue warm-starting
                for j in 1:m.ncols
                    JClp.set_col_bounds!(m, j, base_clo[j], base_chi[j])
                end
                for i in 1:m.nrows
                    JClp.set_row_bounds!(m, i, base_rlo[i], base_rhi[i])
                end
                infeas_induced = false
            elseif step % 15 == 0
                # Temporarily induce primal infeasibility (lo > hi OR contradictory linear RHS)
                r_inf = rand(rng)
                if r_inf < 0.33
                    j = rand(rng, 1:m.ncols)
                    JClp.set_col_bounds!(m, j, 10.0, 5.0)
                elseif r_inf < 0.66
                    i = rand(rng, 1:m.nrows)
                    JClp.set_row_bounds!(m, i, 50.0, 40.0)
                else
                    for j in 1:m.ncols
                        JClp.set_col_bounds!(m, j, max(-10.0, m.col_lower[j]), min(10.0, m.col_upper[j]))
                    end
                    JClp.set_row_bounds!(m, 1, 1.0e5, 1.0e5)
                end
                infeas_induced = true
            else
                # Apply 1-3 random structured mutations
                nmut = rand(rng, 1:3)
                for _ in 1:nmut
                    op = rand(rng, 1:7)
                    if op == 1
                        # Fix / unfix a column
                        j = rand(rng, 1:m.ncols)
                        if m.col_lower[j] == m.col_upper[j]
                            lo = base_clo[j] <= -1.0e20 ? -5.0 : base_clo[j]
                            hi = base_chi[j] >= 1.0e20 ? 5.0 : base_chi[j]
                            lo == hi && (hi = lo + 3.0)
                            JClp.set_col_bounds!(m, j, lo, hi)
                        else
                            lo = max(-5.0, m.col_lower[j] > -1.0e20 ? m.col_lower[j] : -5.0)
                            hi = min(5.0, m.col_upper[j] < 1.0e20 ? m.col_upper[j] : 5.0)
                            v = round(clamp(m.col_activity[j], lo, hi), digits=2)
                            JClp.set_col_bounds!(m, j, v, v)
                        end
                    elseif op == 2
                        # Fix / unfix a row
                        i = rand(rng, 1:m.nrows)
                        if m.row_lower[i] == m.row_upper[i]
                            JClp.set_row_bounds!(m, i, m.row_lower[i] - 2.0, m.row_upper[i] + 2.0)
                        else
                            lo = max(-10.0, m.row_lower[i] > -1.0e20 ? m.row_lower[i] : -10.0)
                            hi = min(10.0, m.row_upper[i] < 1.0e20 ? m.row_upper[i] : 10.0)
                            v = round(clamp(m.row_activity[i], lo, hi), digits=2)
                            JClp.set_row_bounds!(m, i, v, v)
                        end
                    elseif op == 3
                        # Relax one bound to +-1e30 (keep finite if col is one of the 12 free cols? keep bounded enough)
                        if rand(rng) < 0.6
                            j = rand(rng, 1:24)
                            if rand(rng) < 0.5
                                JClp.set_col_bounds!(m, j, -1.0e30, min(m.col_upper[j], 10.0))
                            else
                                JClp.set_col_bounds!(m, j, max(m.col_lower[j], -5.0), 1.0e30)
                            end
                        else
                            i = rand(rng, 13:m.nrows)
                            if rand(rng) < 0.5
                                JClp.set_row_bounds!(m, i, -1.0e30, m.row_upper[i] < 1.0e20 ? m.row_upper[i] : 15.0)
                            else
                                JClp.set_row_bounds!(m, i, m.row_lower[i] > -1.0e20 ? m.row_lower[i] : -15.0, 1.0e30)
                            end
                        end
                    elseif op == 4
                        # Tighten column or row bounds around feasible point
                        j = rand(rng, 1:24)
                        lo = m.col_lower[j] > -1.0e20 ? m.col_lower[j] : -5.0
                        hi = m.col_upper[j] < 1.0e20 ? m.col_upper[j] : 10.0
                        if hi - lo > 0.5
                            mid = clamp(m.col_activity[j], lo, hi)
                            JClp.set_col_bounds!(m, j, round(max(lo, mid - 2.0), digits=2), round(min(hi, mid + 2.0), digits=2))
                        end
                    elseif op == 5
                        # Change costs (including 0.0 and sign flips)
                        for _ in 1:2
                            j = rand(rng, 1:m.ncols)
                            r = rand(rng)
                            nc = r < 0.2 ? 0.0 : (r < 0.5 ? -m.objective[j] : round(randn(rng) * 5.0, digits=2))
                            JClp.set_obj_coeff!(m, j, nc)
                        end
                    elseif op == 6
                        # Flip obj_sense
                        m.obj_sense = -m.obj_sense
                    else
                        # Periodic partial reset to avoid drifting permanently unbounded
                        j = rand(rng, 1:m.ncols)
                        JClp.set_col_bounds!(m, j, base_clo[j], base_chi[j])
                        JClp.set_obj_coeff!(m, j, base_obj[j])
                    end
                end
            end

            m_ref = _clone_model_data(m; scaling = m.scaling_flag)
            st_ws = solver(m; warm_start = true, max_iterations = 5000)
            st_ref = JClp.dual!(m_ref; warm_start = false, max_iterations = 5000)

            @test st_ws == st_ref
            if st_ws == JClp.CLP_STATUS_OPTIMAL
                @test m.objective_value ≈ m_ref.objective_value rtol=1e-6 atol=1e-6
                assert_kkt_optimal(m; tol = 1.0e-6)
                assert_kkt_optimal(m_ref; tol = 1.0e-6)
            elseif st_ws == JClp.CLP_STATUS_PRIMAL_INFEAS
                @test all(isfinite, m.dual_row_sol)
                @test maximum(abs, m.dual_row_sol; init = 0.0) < 1.0e6
                @test maximum(abs, m.reduced_cost; init = 0.0) < 1.0e6
            end
        end
    end

    @testset "3. Warm-start mutation fuzzer on Sample MPS models" begin
        sample_dir = "/Users/laurentplagne/Projects/Clp/parity/data/Sample"
        if isdir(sample_dir)
            rng = MersenneTwister(424242)
            for fname in ("afiro.mps", "e226.mps", "finnis.mps")
                path = joinpath(sample_dir, fname)
                isfile(path) || continue
                m = JClp.ClpSimplexModel()
                JClp.read_mps!(m, path)
                st0 = JClp.dual!(m; warm_start = false, max_iterations = 5000)
                @test st0 == JClp.CLP_STATUS_OPTIMAL
                assert_kkt_optimal(m; tol = 1.0e-6)

                fin_rows = findall(i -> m.row_lower[i] > -1.0e20 || m.row_upper[i] < 1.0e20, 1:m.nrows)
                fin_cols = findall(j -> m.col_lower[j] > -1.0e20 || m.col_upper[j] < 1.0e20, 1:m.ncols)

                for step in 1:20
                    # Perturb finite row bounds, col bounds, or costs
                    if !isempty(fin_rows)
                        i = fin_rows[rand(rng, 1:length(fin_rows))]
                        rlo = m.row_lower[i]
                        rhi = m.row_upper[i]
                        delta = 0.05 * (1.0 + max(rlo > -1.0e20 ? abs(rlo) : 0.0, rhi < 1.0e20 ? abs(rhi) : 0.0)) * (rand(rng) - 0.5)
                        if rlo == rhi
                            JClp.set_row_bounds!(m, i, rlo + delta, rhi + delta)
                        else
                            new_lo = rlo > -1.0e20 ? rlo + delta : rlo
                            new_hi = rhi < 1.0e20 ? rhi + delta : rhi
                            if new_lo <= new_hi
                                JClp.set_row_bounds!(m, i, new_lo, new_hi)
                            end
                        end
                    end
                    if !isempty(fin_cols)
                        j = fin_cols[rand(rng, 1:length(fin_cols))]
                        clo = m.col_lower[j]
                        chi = m.col_upper[j]
                        if clo < chi
                            if chi < 1.0e20
                                delta = 0.05 * (1.0 + abs(chi)) * rand(rng)
                                JClp.set_col_bounds!(m, j, clo, chi + delta)
                            elseif clo > -1.0e20
                                delta = 0.05 * (1.0 + abs(clo)) * rand(rng)
                                JClp.set_col_bounds!(m, j, max(0.0, clo - delta), chi)
                            end
                        end
                    end
                    # Perturb 1-2 objective coefficients
                    for _ in 1:2
                        j = rand(rng, 1:m.ncols)
                        c_old = m.objective[j]
                        c_new = c_old == 0.0 ? round(rand(rng) - 0.5, digits=3) : c_old * (0.8 + 0.4 * rand(rng))
                        JClp.set_obj_coeff!(m, j, c_new)
                    end

                    m_ref = _clone_model_data(m; scaling = m.scaling_flag)
                    st_ws = JClp.dual!(m; warm_start = true, max_iterations = 5000)
                    st_ref = JClp.dual!(m_ref; warm_start = false, max_iterations = 5000)

                    @test st_ws == st_ref
                    if st_ws == JClp.CLP_STATUS_OPTIMAL
                        @test m.objective_value ≈ m_ref.objective_value rtol=1e-5 atol=1e-4
                        assert_kkt_optimal(m; tol = 1.0e-5)
                    end
                end
            end
        end
    end
end
