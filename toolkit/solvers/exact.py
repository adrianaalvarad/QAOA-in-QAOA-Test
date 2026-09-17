"""Solver exacto por fuerza bruta: para bloques chicos (<= exact_threshold
qubits), es mas barato y mas confiable enumerar las 2^n combinaciones que
invocar un circuito cuantico -- optimo garantizado, sirve tambien como
oraculo para validar los otros tres solvers (ver toolkit/tests/test_toolkit.py).
"""

from __future__ import annotations

import itertools

from toolkit.qubo import qubo_cost
from toolkit.solvers.base import BlockQUBO, BlockResult, SubproblemMeta

name = "exact"


def solve(block: BlockQUBO, meta: SubproblemMeta, config, seed: int) -> BlockResult:
    n = block.n
    best_bits = [0] * n
    best_cost = qubo_cost(best_bits, block.linear, block.quad)

    for combo in itertools.product([0, 1], repeat=n):
        cost = qubo_cost(list(combo), block.linear, block.quad)
        if cost < best_cost:
            best_cost = cost
            best_bits = list(combo)

    return BlockResult(
        x_bits=best_bits,
        solver_name=name,
        strategy_name=block.strategy_name,
        meta={"qubo_cost": best_cost, "n_combinations": 2**n},
    )
