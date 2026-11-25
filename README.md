# JClp.jl

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
![Julia](https://img.shields.io/badge/Julia-1.10%2B-9558B2.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**JClp.jl** is a pure-Julia, zero-dependency Linear Programming (LP) solver implementing the Dual and Primal Revised Simplex algorithms, LP Presolve/Postsolve, and Forrest-Tomlin Sparse LU factorization, inspired by [COIN-OR Clp](https://github.com/coin-or/Clp).

## Highlights

- **Pure, idiomatic 1-based Julia**: zero `unsafe_load`/`unsafe_store!`, zero runtime package dependencies (`[deps]` is empty in `Project.toml`).
- **Allocation-free simplex core**: zero dynamic dispatch and zero per-iteration heap allocations in the simplex loop (`0.0 ms` GC across Netlib benchmarks).
- **Full LP pipeline**:
  - Fixed- and free-format `.mps` reader (`MpsReader.jl`)
  - LP Presolve & exact primal/dual Postsolve (`Presolve.jl`: fixed/empty columns, row/column singletons, forcing/redundant rows, primal bound tightening, zero-cost slack singletons, doubletons, length-2 implied-free substitutions)
  - Dual Simplex with Steepest-Edge row pricing (`DualLoop.jl`, `DualRowPivot.jl`, `SimplexDual.jl`)
  - Primal Simplex with Steepest-Edge / Devex column pricing and piecewise-linear Phase-I cost (`PrimalLoop.jl`, `PrimalColumnPivot.jl`, `NonLinearCost.jl`, `SimplexPrimal.jl`)
  - Markowitz Sparse LU factorization with Forrest-Tomlin updates (`src/coinutils/SparseLU/`)
- **Verified parity**: full unit test suite and 114 / 114 Sample + Netlib LP benchmark parity against C++ Clp (all 91 Netlib models solved in ~9.9 s with dual KKT residual `‖rc - (c - A'y)‖_∞ ≤ 5.5e-7`).

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/LaurentPlagne/JClp.jl")
```

## Quick Start

```julia
using JClp

m = JClp.ClpSimplexModel()
JClp.read_mps!(m, "path/to/problem.mps")

# Presolve + Dual Simplex + Postsolve (recommended)
status = JClp.initial_solve!(m; max_iterations = 200_000)

println("Status:     ", status)               # 0 = OPTIMAL, 1 = PRIMAL_INFEASIBLE, 2 = DUAL_INFEASIBLE
println("Iterations: ", m.number_iterations)
println("Objective:  ", m.objective_value)
```

You can also invoke the Dual or Primal simplex drivers directly (without presolve):

```julia
status = JClp.dual!(m; max_iterations = 200_000)
status = JClp.primal!(m; max_iterations = 200_000)
```

## Representative Benchmarks (vs C++ Clp)

Measured on Apple Silicon (Julia 1.13.2, `initial_solve!` with presolve enabled, warmed up, `0.0 ms` GC):

| Instance | Size (`m × n`) | `JClp.jl` (ms) | C++ Clp (ms) | Ratio (`JClp / Clp`) | `JClp` Iters |
|---|---:|---:|---:|---:|---:|
| `e226` | 223 × 282 | **3.3** | 7.4 | **0.44×** | 342 |
| `stocfor2` | 2,157 × 2,031 | 19.3 | **12.3** | 1.57× | 1,597 |
| `25fv47` | 821 × 1,571 | **68.0** | 108.8 | **0.63×** | 2,441 |
| `greenbea` | 2,392 × 5,405 | **86.7** | 168.2 | **0.52×** | 3,393 |
| `maros-r7` | 3,136 × 9,408 | **446.7** | 1,420.5 | **0.31×** | 2,492 |
| `pilot87` | 2,030 × 4,883 | **2,364.7** | 3,238.7 | **0.73×** | 9,962 |
| `dfl001` | 6,071 × 12,230 | **3,847.4** | 4,154.4 | **0.93×** | 21,847 |

## Architecture

```text
src/
├── JClp.jl                        # Module entry point
├── coinutils/SparseLU/            # Markowitz Sparse LU, FTRAN/BTRAN, Forrest-Tomlin updates
│   ├── SparseLU.jl
│   ├── types.jl
│   ├── factor.jl
│   ├── ftran.jl
│   ├── btran.jl
│   └── update.jl
└── clp/                           # Pure-Julia Simplex & Presolve engine
    ├── PackedMatrix.jl            # CSC/CSR sparse matrix & transpose-times kernels
    ├── Factorization.jl           # Basis factorization wrapper
    ├── DualRowPivot.jl            # Dual steepest-edge pricing
    ├── PrimalColumnPivot.jl       # Primal steepest-edge / Devex pricing
    ├── NonLinearCost.jl           # Phase-I piecewise-linear infeasibility cost
    ├── SimplexBase.jl             # Basis housekeeping, tolerances, solution checks
    ├── SimplexDual.jl             # Dual ratio test & reduced-cost updates
    ├── SimplexPrimal.jl           # Primal ratio test & bound flips
    ├── SimplexProgress.jl         # Anti-cycling / progress tracking
    ├── DualLoop.jl                # Dual simplex inner loop
    ├── PrimalLoop.jl              # Primal simplex inner loop
    ├── MpsReader.jl               # Fixed/free-format MPS parser
    ├── SimplexModel.jl            # Model container, scaling, rim setup
    ├── SimplexSolve.jl            # Top-level dual! / primal! drivers
    └── Presolve.jl                # LP presolve, postsolve & initial_solve!
```

## Running Tests

```julia
using Pkg
Pkg.test("JClp")
```

## Acknowledgements & License

JClp.jl is inspired by the algorithms and design of [COIN-OR Clp](https://github.com/coin-or/Clp) and [CoinUtils](https://github.com/coin-or/CoinUtils).

Licensed under the [MIT License](LICENSE.md) — Copyright (c) 2026 Laurent Plagne.
