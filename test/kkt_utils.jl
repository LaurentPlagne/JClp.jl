using JClp
using Test

const _KKT_INF = 1.0e20

"""
    check_kkt_optimal(m::JClp.ClpSimplexModel; tol::Float64=1e-6)

Verify full primal-dual KKT optimality conditions for `m` after solving:
1. Primal feasibility (`col_lower <= col_activity <= col_upper`, `row_lower <= A*x <= row_upper`, `row_activity ≈ A*x`)
2. Dual stationarity (`reduced_cost ≈ objective - A' * dual_row_sol`)
3. Dual feasibility (respecting `obj_sense`: working minimization `rc_min = obj_sense * rc`, `y_min = obj_sense * y`)
4. Complementary slackness
5. Basis cardinality (`count(BASIC) == nrows`)
6. Strong duality (`c'x ≈ sum(y_i * active_rhs_i) + sum(rc_j * active_bound_j)`) and `objective_value` consistency.

Returns a NamedTuple `(ok, primal_viol, dual_stat_err, dual_feas_viol, comp_viol, gap_rel, basis_ok, details)`.
"""
function check_kkt_optimal(m::JClp.ClpSimplexModel; tol::Float64 = 1.0e-6)
    if m.problem_status != JClp.CLP_STATUS_OPTIMAL
        return (
            ok = false,
            primal_viol = Inf,
            dual_stat_err = Inf,
            dual_feas_viol = Inf,
            comp_viol = Inf,
            gap_rel = Inf,
            basis_ok = false,
            details = "problem_status = $(m.problem_status) (expected CLP_STATUS_OPTIMAL = $(JClp.CLP_STATUS_OPTIMAL))",
        )
    end

    nrows = m.nrows
    ncols = m.ncols
    x = m.col_activity
    ra = m.row_activity
    y = m.dual_row_sol
    rc = m.reduced_cost
    c = m.objective

    if !isfinite(m.objective_value) ||
       any(!isfinite, x) || any(!isfinite, ra) || any(!isfinite, y) || any(!isfinite, rc)
        return (
            ok = false,
            primal_viol = Inf,
            dual_stat_err = Inf,
            dual_feas_viol = Inf,
            comp_viol = Inf,
            gap_rel = Inf,
            basis_ok = false,
            details = "Non-finite value in solution, duals, reduced costs, or objective_value",
        )
    end

    # Compute A*x, |A|*|x|, and A'*y directly from original unscaled matrix
    ax = zeros(Float64, nrows)
    ax_abs = zeros(Float64, nrows)
    aty = zeros(Float64, ncols)
    starts = m.mat_starts
    lengths = m.mat_lengths
    indices = m.mat_indices
    elems = m.mat_elements

    @inbounds for j in 1:ncols
        xj = x[j]
        s_aty = 0.0
        k0 = Int(starts[j])
        k1 = k0 + Int(lengths[j]) - 1
        for k in k0:k1
            i = Int(indices[k])
            a = elems[k]
            ax[i] += a * xj
            ax_abs[i] += abs(a * xj)
            s_aty += a * y[i]
        end
        aty[j] = s_aty
    end

    # 1. Primal feasibility & row_activity consistency
    primal_viol = 0.0
    p_detail = ""
    @inbounds for j in 1:ncols
        lo = m.col_lower[j]
        hi = m.col_upper[j]
        xj = x[j]
        if lo > -_KKT_INF
            v = max(0.0, lo - xj) / max(1.0, abs(lo))
            if v > primal_viol
                primal_viol = v
                p_detail = "col $j below lower: x=$xj < lo=$lo"
            end
        end
        if hi < _KKT_INF
            v = max(0.0, xj - hi) / max(1.0, abs(hi))
            if v > primal_viol
                primal_viol = v
                p_detail = "col $j above upper: x=$xj > hi=$hi"
            end
        end
    end
    @inbounds for i in 1:nrows
        rlo = m.row_lower[i]
        rhi = m.row_upper[i]
        axi = ax[i]
        if rlo > -_KKT_INF
            v = max(0.0, rlo - axi) / max(1.0, abs(rlo))
            if v > primal_viol
                primal_viol = v
                p_detail = "row $i below lower: Ax=$axi < rlo=$rlo"
            end
        end
        if rhi < _KKT_INF
            v = max(0.0, axi - rhi) / max(1.0, abs(rhi))
            if v > primal_viol
                primal_viol = v
                p_detail = "row $i above upper: Ax=$axi > rhi=$rhi"
            end
        end
        ra_err = abs(ra[i] - axi) / max(1.0, abs(axi), ax_abs[i])
        if ra_err > primal_viol
            primal_viol = ra_err
            p_detail = "row $i row_activity=$(ra[i]) != Ax=$axi"
        end
    end

    # 2. Dual stationarity: ||rc - (c - A'y)||_inf / max(1, ||c||_inf, ||y||_inf)
    norm_c = isempty(c) ? 0.0 : maximum(abs, c)
    norm_y = isempty(y) ? 0.0 : maximum(abs, y)
    d_scale = max(1.0, norm_c, norm_y)
    dual_stat_err = 0.0
    stat_detail = ""
    @inbounds for j in 1:ncols
        err = abs(rc[j] - (c[j] - aty[j])) / d_scale
        if err > dual_stat_err
            dual_stat_err = err
            stat_detail = "col $j rc=$(rc[j]) vs c-A'y=$(c[j] - aty[j])"
        end
    end

    # 3. Dual feasibility (in working minimization sense: s = obj_sense)
    s = m.obj_sense
    dual_feas_viol = 0.0
    df_detail = ""
    @inbounds for j in 1:ncols
        st = m.status[j] & JClp.CLP_STATUS_MASK
        lo = m.col_lower[j]
        hi = m.col_upper[j]
        rc_min = s * rc[j]
        v = 0.0
        if st == JClp.CLP_STATUS_BASIC || st == JClp.CLP_STATUS_IS_FREE || st == JClp.CLP_STATUS_SUPER_BASIC
            v = abs(rc_min) / d_scale
        elseif st == JClp.CLP_STATUS_AT_LOWER
            if lo <= -_KKT_INF
                v = Inf
            elseif lo != hi
                v = max(0.0, -rc_min) / d_scale
            end
        elseif st == JClp.CLP_STATUS_AT_UPPER
            if hi >= _KKT_INF
                v = Inf
            elseif lo != hi
                v = max(0.0, rc_min) / d_scale
            end
        elseif st == JClp.CLP_STATUS_IS_FIXED
            if lo <= -_KKT_INF || hi >= _KKT_INF || abs(hi - lo) > tol * max(1.0, abs(lo))
                v = Inf
            end
        else
            v = Inf
        end
        if v > dual_feas_viol
            dual_feas_viol = v
            df_detail = "col $j st=$st rc_min=$rc_min lo=$lo hi=$hi"
        end
    end
    @inbounds for i in 1:nrows
        st = m.status[ncols + i] & JClp.CLP_STATUS_MASK
        rlo = m.row_lower[i]
        rhi = m.row_upper[i]
        y_min = s * y[i]
        v = 0.0
        if st == JClp.CLP_STATUS_BASIC || st == JClp.CLP_STATUS_IS_FREE || st == JClp.CLP_STATUS_SUPER_BASIC
            v = abs(y_min) / d_scale
        elseif st == JClp.CLP_STATUS_AT_LOWER
            if rlo <= -_KKT_INF
                v = Inf
            elseif rlo != rhi
                v = max(0.0, -y_min) / d_scale
            end
        elseif st == JClp.CLP_STATUS_AT_UPPER
            if rhi >= _KKT_INF
                v = Inf
            elseif rlo != rhi
                v = max(0.0, y_min) / d_scale
            end
        elseif st == JClp.CLP_STATUS_IS_FIXED
            if rlo <= -_KKT_INF || rhi >= _KKT_INF || abs(rhi - rlo) > tol * max(1.0, abs(rlo))
                v = Inf
            end
        else
            v = Inf
        end
        if v > dual_feas_viol
            dual_feas_viol = v
            df_detail = "row $i st=$st y_min=$y_min rlo=$rlo rhi=$rhi"
        end
    end

    # 4. Complementarity
    tol_d = tol * d_scale
    comp_viol = 0.0
    comp_detail = ""
    @inbounds for j in 1:ncols
        lo = m.col_lower[j]
        hi = m.col_upper[j]
        xj = x[j]
        rc_min = s * rc[j]
        if lo > -_KKT_INF && rc_min > tol_d
            v = max(0.0, xj - lo) / max(1.0, abs(lo))
            if v > comp_viol
                comp_viol = v
                comp_detail = "col $j comp lower: rc_min=$rc_min x=$xj lo=$lo"
            end
        end
        if hi < _KKT_INF && rc_min < -tol_d
            v = max(0.0, hi - xj) / max(1.0, abs(hi))
            if v > comp_viol
                comp_viol = v
                comp_detail = "col $j comp upper: rc_min=$rc_min x=$xj hi=$hi"
            end
        end
    end
    @inbounds for i in 1:nrows
        rlo = m.row_lower[i]
        rhi = m.row_upper[i]
        axi = ax[i]
        y_min = s * y[i]
        if rlo > -_KKT_INF && y_min > tol_d
            v = max(0.0, axi - rlo) / max(1.0, abs(rlo))
            if v > comp_viol
                comp_viol = v
                comp_detail = "row $i comp lower: y_min=$y_min Ax=$axi rlo=$rlo"
            end
        end
        if rhi < _KKT_INF && y_min < -tol_d
            v = max(0.0, rhi - axi) / max(1.0, abs(rhi))
            if v > comp_viol
                comp_viol = v
                comp_detail = "row $i comp upper: y_min=$y_min Ax=$axi rhi=$rhi"
            end
        end
    end

    # 5. Basis cardinality
    nbasic = count(st -> (st & JClp.CLP_STATUS_MASK) == JClp.CLP_STATUS_BASIC, m.status)
    basis_ok = (nbasic == nrows)

    # 6. Strong duality & objective_value consistency
    primal_cx = 0.0
    @inbounds for j in 1:ncols
        primal_cx += c[j] * x[j]
    end
    dual_obj_no_off = 0.0
    @inbounds for i in 1:nrows
        st = m.status[ncols + i] & JClp.CLP_STATUS_MASK
        if st == JClp.CLP_STATUS_AT_LOWER || st == JClp.CLP_STATUS_IS_FIXED
            rlo = m.row_lower[i]
            if rlo > -_KKT_INF
                dual_obj_no_off += y[i] * rlo
            end
        elseif st == JClp.CLP_STATUS_AT_UPPER
            rhi = m.row_upper[i]
            if rhi < _KKT_INF
                dual_obj_no_off += y[i] * rhi
            end
        end
    end
    @inbounds for j in 1:ncols
        st = m.status[j] & JClp.CLP_STATUS_MASK
        if st == JClp.CLP_STATUS_AT_LOWER || st == JClp.CLP_STATUS_IS_FIXED
            lo = m.col_lower[j]
            if lo > -_KKT_INF
                dual_obj_no_off += rc[j] * lo
            end
        elseif st == JClp.CLP_STATUS_AT_UPPER
            hi = m.col_upper[j]
            if hi < _KKT_INF
                dual_obj_no_off += rc[j] * hi
            end
        end
    end
    gap_dual = abs(primal_cx - dual_obj_no_off) / max(1.0, abs(primal_cx), abs(dual_obj_no_off))
    obj_err = abs((primal_cx - m.obj_offset) - m.objective_value) /
              max(1.0, abs(m.objective_value), abs(primal_cx), abs(m.obj_offset))
    gap_rel = max(gap_dual, obj_err)

    ok = basis_ok &&
         primal_viol <= tol &&
         dual_stat_err <= tol &&
         dual_feas_viol <= tol &&
         comp_viol <= tol &&
         gap_rel <= tol

    msgs = String[]
    basis_ok || push!(msgs, "basis count=$nbasic != nrows=$nrows")
    primal_viol <= tol || push!(msgs, "primal_viol=$primal_viol ($p_detail)")
    dual_stat_err <= tol || push!(msgs, "dual_stat_err=$dual_stat_err ($stat_detail)")
    dual_feas_viol <= tol || push!(msgs, "dual_feas_viol=$dual_feas_viol ($df_detail)")
    comp_viol <= tol || push!(msgs, "comp_viol=$comp_viol ($comp_detail)")
    gap_rel <= tol || push!(msgs, "gap_rel=$gap_rel (cx=$primal_cx dual=$dual_obj_no_off obj_val=$(m.objective_value) off=$(m.obj_offset))")
    details = isempty(msgs) ? "ok" : join(msgs, "; ")

    return (
        ok = ok,
        primal_viol = primal_viol,
        dual_stat_err = dual_stat_err,
        dual_feas_viol = dual_feas_viol,
        comp_viol = comp_viol,
        gap_rel = gap_rel,
        basis_ok = basis_ok,
        details = details,
    )
end

function assert_kkt_optimal(m::JClp.ClpSimplexModel; tol::Float64 = 1.0e-6)
    res = check_kkt_optimal(m; tol = tol)
    if !res.ok
        @info "KKT check failed" res.primal_viol res.dual_stat_err res.dual_feas_viol res.comp_viol res.gap_rel res.basis_ok res.details
    end
    @test res.basis_ok
    @test res.primal_viol <= tol
    @test res.dual_stat_err <= tol
    @test res.dual_feas_viol <= tol
    @test res.comp_viol <= tol
    @test res.gap_rel <= tol
    @test res.ok
    return res
end
