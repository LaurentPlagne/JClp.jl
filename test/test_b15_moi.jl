using JClp
using Test
import MathOptInterface as MOI

@testset "B15: MathOptInterface (JClpMOIExt) LP Compliance & Solver Attributes" begin
    @testset "Direct Optimizer attributes, kwargs, algorithms & edge cases" begin
        opt = JClp.Optimizer()
        @test MOI.is_empty(opt)
        @test MOI.get(opt, MOI.SolverName()) == "JClp"
        @test MOI.get(opt, MOI.SolverVersion()) == string(pkgversion(JClp))
        @test MOI.get(opt, MOI.Silent()) == false
        @test MOI.get(opt, MOI.RawOptimizerAttribute("presolve")) === true
        @test MOI.get(opt, MOI.RawOptimizerAttribute("algorithm")) == "initial"
        @test MOI.get(opt, MOI.RawOptimizerAttribute("max_iterations")) == Int(typemax(Int32))
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
        @test MOI.get(opt, MOI.PrimalStatus()) == MOI.NO_SOLUTION
        @test MOI.get(opt, MOI.DualStatus()) == MOI.NO_SOLUTION
        @test MOI.get(opt, MOI.ResultCount()) == 0
        @test MOI.get(opt, MOI.RawStatusString()) == "Optimize not called"

        # Persistent optimizer attributes survive MOI.empty!
        MOI.set(opt, MOI.Silent(), true)
        MOI.set(opt, MOI.RawOptimizerAttribute("presolve"), false)
        MOI.set(opt, MOI.RawOptimizerAttribute("max_iterations"), 1234)
        MOI.set(opt, MOI.RawOptimizerAttribute("algorithm"), "dual")
        MOI.set(opt, MOI.Name(), "my_lp")
        MOI.empty!(opt)
        @test MOI.is_empty(opt)
        @test MOI.get(opt, MOI.Silent()) == true
        @test MOI.get(opt, MOI.RawOptimizerAttribute("presolve")) === false
        @test MOI.get(opt, MOI.RawOptimizerAttribute("max_iterations")) == 1234
        @test MOI.get(opt, MOI.RawOptimizerAttribute("algorithm")) == "dual"
        @test MOI.get(opt, MOI.Name()) == ""

        # Invalid RawOptimizerAttribute errors
        @test !MOI.supports(opt, MOI.RawOptimizerAttribute("unknown_param"))
        @test_throws MOI.UnsupportedAttribute MOI.get(opt, MOI.RawOptimizerAttribute("unknown_param"))
        @test_throws MOI.UnsupportedAttribute MOI.set(opt, MOI.RawOptimizerAttribute("unknown_param"), 1)
        @test_throws MOI.SetAttributeNotAllowed MOI.set(opt, MOI.RawOptimizerAttribute("algorithm"), "barrier")
        @test_throws MOI.SetAttributeNotAllowed MOI.set(opt, MOI.RawOptimizerAttribute("presolve"), "yes")
        @test_throws MOI.SetAttributeNotAllowed MOI.set(opt, MOI.RawOptimizerAttribute("max_iterations"), -5)

        # Keyword constructor, duplicate terms, affine constant, obj constant, & all 3 algorithms
        # min 2*x1 + 3*x2 + 10 s.t. x1 + x2 + 1 >= 5, 0 <= x1 <= 10, x2 >= 0 -> opt x1=4, x2=0, obj=18
        for alg in ("initial", "dual", "primal"), presolve in (true, false)
            src = MOI.Utilities.Model{Float64}()
            x = MOI.add_variables(src, 2)
            MOI.add_constraint(src, x[1], MOI.GreaterThan(0.0))
            MOI.add_constraint(src, x[1], MOI.LessThan(10.0))
            MOI.add_constraint(
                src,
                MOI.ScalarAffineFunction(
                    [MOI.ScalarAffineTerm(0.5, x[1]), MOI.ScalarAffineTerm(0.5, x[1]), MOI.ScalarAffineTerm(1.0, x[2])],
                    1.0,
                ),
                MOI.GreaterThan(5.0),
            )
            MOI.add_constraint(src, x[2], MOI.GreaterThan(0.0))
            MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            MOI.set(
                src,
                MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction(
                    [MOI.ScalarAffineTerm(2.0, x[1]), MOI.ScalarAffineTerm(3.0, x[2])],
                    10.0,
                ),
            )
            o = JClp.Optimizer(; presolve = presolve, algorithm = alg, silent = true)
            imap, copied = MOI.optimize!(o, src)
            @test copied
            @test MOI.get(o, MOI.TerminationStatus()) == MOI.OPTIMAL
            @test MOI.get(o, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
            @test MOI.get(o, MOI.DualStatus()) == MOI.FEASIBLE_POINT
            @test MOI.get(o, MOI.ResultCount()) == 1
            @test MOI.get(o, MOI.ObjectiveValue()) ≈ 18.0 atol=1e-8
            @test MOI.get(o, MOI.DualObjectiveValue()) ≈ 18.0 atol=1e-8
            @test MOI.get(o, MOI.VariablePrimal(), imap[x[1]]) ≈ 4.0 atol=1e-8
            @test MOI.get(o, MOI.VariablePrimal(), imap[x[2]]) ≈ 0.0 atol=1e-8
            @test MOI.get(o, MOI.SimplexIterations()) >= 0
            @test MOI.get(o, MOI.SolveTimeSec()) >= 0.0
            @test MOI.get(o, MOI.RawStatusString()) == "Optimal"
            @test_throws MOI.ResultIndexBoundsError MOI.get(o, MOI.ObjectiveValue(2))
            @test_throws MOI.ResultIndexBoundsError MOI.get(o, MOI.VariablePrimal(2), imap[x[1]])

            # Re-optimizing in place succeeds identically
            MOI.optimize!(o)
            @test MOI.get(o, MOI.TerminationStatus()) == MOI.OPTIMAL
            @test MOI.get(o, MOI.ObjectiveValue()) ≈ 18.0 atol=1e-8
        end

        # Multiple bounds on same variable, MAX_SENSE, objective constant, and basis status
        src2 = MOI.Utilities.Model{Float64}()
        v = MOI.add_variable(src2)
        c_gt = MOI.add_constraint(src2, v, MOI.GreaterThan(1.0))
        c_lt = MOI.add_constraint(src2, v, MOI.LessThan(5.0))
        MOI.set(src2, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        MOI.set(
            src2,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(3.0, v)], 2.5),
        )
        o2 = JClp.Optimizer()
        imap2, _ = MOI.optimize!(o2, src2)
        @test MOI.get(o2, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(o2, MOI.ObjectiveValue()) ≈ 17.5 atol=1e-9
        @test MOI.get(o2, MOI.DualObjectiveValue()) ≈ 17.5 atol=1e-9
        @test MOI.get(o2, MOI.ConstraintDual(), imap2[c_gt]) ≈ 0.0 atol=1e-9
        @test MOI.get(o2, MOI.ConstraintDual(), imap2[c_lt]) ≈ -3.0 atol=1e-9
        @test MOI.get(o2, MOI.VariableBasisStatus(), imap2[v]) == MOI.NONBASIC_AT_UPPER

        # Infinite/contradictory bounds edge cases & ITERATION_LIMIT
        src_inf = MOI.Utilities.Model{Float64}()
        vi = MOI.add_variable(src_inf)
        MOI.add_constraint(src_inf, vi, MOI.GreaterThan(Inf))
        oi = JClp.Optimizer()
        MOI.optimize!(oi, src_inf)
        @test MOI.get(oi, MOI.TerminationStatus()) == MOI.INFEASIBLE
        @test MOI.get(oi, MOI.ResultCount()) == 0
        @test_throws MOI.ResultIndexBoundsError MOI.get(oi, MOI.ObjectiveValue())

        # Empty-variable / empty-row models
        src_empty = MOI.Utilities.Model{Float64}()
        MOI.set(src_empty, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(
            src_empty,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm{Float64}[], 7.0),
        )
        oe = JClp.Optimizer()
        MOI.optimize!(oe, src_empty)
        @test MOI.get(oe, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(oe, MOI.ObjectiveValue()) ≈ 7.0 atol=1e-9
        @test MOI.get(oe, MOI.DualObjectiveValue()) ≈ 7.0 atol=1e-9

        # max_iterations = 0 on a 2x2 LP needing pivots (presolve=false)
        src_lim = MOI.Utilities.Model{Float64}()
        xl = MOI.add_variables(src_lim, 2)
        MOI.add_constraint.(src_lim, xl, MOI.GreaterThan(0.0))
        MOI.add_constraint(src_lim, 1.0 * xl[1] + 1.0 * xl[2], MOI.LessThan(4.0))
        MOI.add_constraint(src_lim, 2.0 * xl[1] + 1.0 * xl[2], MOI.LessThan(6.0))
        MOI.set(src_lim, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(src_lim, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), -3.0 * xl[1] - 2.0 * xl[2])
        ol = JClp.Optimizer(; presolve = false, max_iterations = 0)
        MOI.optimize!(ol, src_lim)
        @test MOI.get(ol, MOI.TerminationStatus()) == MOI.ITERATION_LIMIT
        @test MOI.get(ol, MOI.RawStatusString()) == "Iteration limit reached"
        @test MOI.get(ol, MOI.ResultCount()) == 0
    end

    @testset "MOI.Test.runtests (core LP compliance)" begin
        model = MOI.Bridges.full_bridge_optimizer(
            MOI.Utilities.CachingOptimizer(
                MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
                JClp.Optimizer(),
            ),
            Float64,
        )
        config = MOI.Test.Config(
            Float64;
            atol = 1e-6,
            rtol = 1e-6,
        )
        excludes = Any[
            # Integer / binary / SOS / indicator / semicontinuous / semiinteger constraints (LP solver only)
            "test_linear_integer_",
            "test_linear_SOS",
            "test_linear_Indicator",
            "test_linear_Semicontinuous",
            "test_linear_Semiinteger",
            "test_variable_solve_Integer_",
            "test_variable_solve_ZeroOne_",
            "test_constraint_ZeroOne_",
            "test_constraint_Indicator_",
            "test_solve_SOS2_",
            "test_solve_ObjectiveBound_",
            # Quadratic / conic / nonlinear / CP-SAT
            "test_objective_qp_",
            "test_objective_ScalarQuadraticFunction_",
            "test_constraint_qcp_",
            "test_constraint_PrimalStart_DualStart_SecondOrderCone",
            "test_variable_delete_SecondOrderCone",
            r"^test_basic_(ScalarQuadraticFunction|VectorQuadraticFunction|ScalarNonlinearFunction|VectorNonlinearFunction)_",
            r"^test_basic_.*_(Integer|ZeroOne|Semicontinuous|Semiinteger|SOS1|SOS2|Indicator_LessThan|Indicator_GreaterThan|SecondOrderCone|RotatedSecondOrderCone|ExponentialCone|DualExponentialCone|GeometricMeanCone|DualGeometricMeanCone|PowerCone|DualPowerCone|RelativeEntropyCone|DualRelativeEntropyCone|NormInfinityCone|NormOneCone|NormCone|NormSpectralCone|NormNuclearCone|PositiveSemidefiniteConeTriangle|PositiveSemidefiniteConeSquare|HermitianPositiveSemidefiniteConeTriangle|ScaledPositiveSemidefiniteConeTriangle|LogDetConeTriangle|LogDetConeSquare|RootDetConeTriangle|RootDetConeSquare|Complements|AllDifferent|CountDistinct|CountBelongs|CountAtLeast|CountGreaterThan|BinPacking|Circuit|Cumulative|Table|Path|VectorNonlinearOracle)$",
            # Warm-start attributes not supported yet
            "test_model_VariablePrimalStart",
            "test_model_ModelFilter_AbstractVariableAttribute",
            "test_model_ModelFilter_AbstractConstraintAttribute",
            "test_constraint_ConstraintPrimalStart",
            "test_constraint_ConstraintDualStart",
            # Infeasibility certificates & IIS conflict finder not implemented
            "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE",
            "test_solve_conflict_",
            # Unsupported optimizer attributes
            "TimeLimitSec",
            "NumberThreads",
            "NumberOfThreads",
        ]
        MOI.Test.runtests(
            model,
            config;
            include = [
                "test_linear_",
                "test_objective_",
                "test_variable_",
                "test_constraint_",
                "test_solve_",
                "test_basic_",
                "test_model_",
            ],
            exclude = excludes,
            warn_unsupported = false,
        )
    end

    @testset "MOI.Test.runtests (attributes, dual objective & modifications)" begin
        model = MOI.Bridges.full_bridge_optimizer(
            MOI.Utilities.CachingOptimizer(
                MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
                JClp.Optimizer(),
            ),
            Float64,
        )
        config = MOI.Test.Config(
            Float64;
            atol = 1e-6,
            rtol = 1e-6,
        )
        MOI.Test.runtests(
            model,
            config;
            include = [
                "test_attribute_",
                "test_DualObjectiveValue_",
                "test_modification_",
            ],
            exclude = Any[
                "TimeLimitSec",
                "NumberThreads",
                "NumberOfThreads",
                "AbsoluteGapTolerance",
                "RelativeGapTolerance",
                "NodeLimit",
                "ObjectiveLimit",
                "SolutionLimit",
                "scalarquadraticcoefficientchange",
            ],
            warn_unsupported = false,
        )
    end
end
