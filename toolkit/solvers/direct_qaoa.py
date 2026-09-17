"""QAOA directo con warm-start (Egger et al. 2021), sobre el QUBO de un solo
bloque. Portado de poc-qaoa-agnostic-refinement/src/quantum/qaoa_refine.py;
el ansatz en si vive en _qaoa_core.py (compartido con qaoa_square.py).
"""

from __future__ import annotations

from toolkit.qubo import qubo_cost, qubo_to_ising
from toolkit.solvers._qaoa_core import run_warm_start_qaoa
from toolkit.solvers.base import BlockQUBO, BlockResult, SubproblemMeta

name = "direct_qaoa"


def solve(block: BlockQUBO, meta: SubproblemMeta, config, seed: int) -> BlockResult:
    n = block.n
    h, j_coeffs = qubo_to_ising(block.linear, block.quad, n)

    best_bits, best_cost = run_warm_start_qaoa(
        n, h, j_coeffs,
        p=config.direct_qaoa_p,
        shots=config.direct_qaoa_shots,
        maxiter=config.direct_qaoa_maxiter,
        epsilon=config.direct_qaoa_epsilon,
        top_k=config.direct_qaoa_top_k,
        seed=seed,
        evaluate_fn=lambda bits: qubo_cost(bits, block.linear, block.quad),
    )

    return BlockResult(
        x_bits=best_bits,
        solver_name=name,
        strategy_name=block.strategy_name,
        meta={"qubo_cost": best_cost},
    )
