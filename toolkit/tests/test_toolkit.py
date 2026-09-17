"""Verificacion end-to-end del toolkit (items 1-3 del plan de
implementacion). Se puede correr con pytest o directamente:

    python -m toolkit.tests.test_toolkit
"""

from __future__ import annotations

import itertools

import networkx as nx

from toolkit.config import ToolkitConfig
from toolkit.qubo import (
    build_alt_color_strategies,
    build_qubo,
    decode_bits_to_coloring,
    qubo_cost,
    qubo_to_ising,
)
from toolkit.solvers._ising_adapter import block_to_ising_graph
from toolkit.solvers._qaoa_core import ising_energy
from toolkit.solvers.base import BlockQUBO, SubproblemMeta


def _toy_graph_and_coloring():
    """4 nodos, k=3: 0-1 (w=2, mismo color -> conflicto), 1-2 (w=3, distinto
    color), 2-3 (w=1, mismo color -> conflicto), 0 aislado de 3."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=2.0)
    g.add_edge(1, 2, weight=3.0)
    g.add_edge(2, 3, weight=1.0)
    coloring = {0: 0, 1: 0, 2: 1, 3: 1}
    return g, coloring


def test_qubo_matches_manual_cost():
    g, coloring = _toy_graph_and_coloring()
    k = 3
    block_nodes = [0, 1, 2, 3]
    strategies = build_alt_color_strategies(g, block_nodes, coloring, k)

    for strategy_name, alt_colors in strategies.items():
        linear, quad = build_qubo(g, block_nodes, coloring, alt_colors)
        for bits in itertools.product([0, 1], repeat=4):
            trial = decode_bits_to_coloring(list(bits), block_nodes, coloring, alt_colors)
            manual_cost = sum(
                data["weight"] for u, v, data in g.edges(data=True) if trial[u] == trial[v]
            )
            # qubo_cost + costo con bits=0 (referencia) debe reproducir el manual,
            # porque el offset constante de build_qubo es exactamente el costo en x=0.
            base_cost = sum(
                data["weight"] for u, v, data in g.edges(data=True) if coloring[u] == coloring[v]
            )
            zero_bits = [0, 0, 0, 0]
            offset = base_cost - qubo_cost(zero_bits, linear, quad)
            reproduced = qubo_cost(list(bits), linear, quad) + offset
            assert abs(reproduced - manual_cost) < 1e-9, (
                f"{strategy_name} bits={bits}: qubo={reproduced} manual={manual_cost}"
            )
    print("test_qubo_matches_manual_cost: OK")


def test_ising_adapter_scale():
    block = BlockQUBO(
        block_nodes=[0, 1], linear=[0.7, -0.3], quad={(0, 1): 1.6},
        alt_colors={}, strategy_name="individual",
    )
    h, j_coeffs = qubo_to_ising(block.linear, block.quad, 2)
    graph, boundary_fields, _ = block_to_ising_graph(block)

    assert graph.has_edge(0, 1), "el adaptador debe crear la arista (0,1)"
    w = graph[0][1]["weight"]
    assert abs(w - 2.0 * j_coeffs[(0, 1)]) < 1e-12, "el peso de arista debe ser 2*J, no J"

    for bits in itertools.product([0, 1], repeat=2):
        direct = ising_energy(list(bits), h, j_coeffs)
        z = [1.0 if b == 0 else -1.0 for b in bits]
        replicated = 0.5 * w * z[0] * z[1] + boundary_fields[0] * z[0] + boundary_fields[1] * z[1]
        assert abs(direct - replicated) < 1e-9, (
            f"bits={bits}: ising_energy={direct} formula_r_qaoa={replicated}"
        )
    print("test_ising_adapter_scale: OK")


def test_exact_is_oracle():
    from toolkit.solvers import direct_qaoa, exact, r_qaoa

    g, coloring = _toy_graph_and_coloring()
    g.add_edge(0, 3, weight=0.8)
    g.add_edge(1, 3, weight=0.4)
    block_nodes = [0, 1, 2, 3]
    config = ToolkitConfig(n_colors=3, exact_threshold=6, rqaoa_cutoff_nodes=2, rqaoa_maxiter=10, direct_qaoa_maxiter=15, direct_qaoa_shots=256)

    strategies = build_alt_color_strategies(g, block_nodes, coloring, config.n_colors)
    for strategy_name, alt_colors in strategies.items():
        linear, quad = build_qubo(g, block_nodes, coloring, alt_colors)
        block = BlockQUBO(block_nodes, linear, quad, alt_colors, strategy_name)
        meta = SubproblemMeta(n=4, n_edges_internal=g.number_of_edges(), density=1.0, hotspot_index=0, global_density=1.0)

        exact_result = exact.solve(block, meta, config, seed=1)
        exact_cost = qubo_cost(exact_result.x_bits, linear, quad)

        for module in (direct_qaoa, r_qaoa):
            result = module.solve(block, meta, config, seed=1)
            cost = qubo_cost(result.x_bits, linear, quad)
            assert cost >= exact_cost - 1e-6, (
                f"{module.name}/{strategy_name} dio costo {cost} mejor que el oraculo exacto {exact_cost}"
            )
    print("test_exact_is_oracle: OK (direct_qaoa y r_qaoa nunca superan al oraculo)")


if __name__ == "__main__":
    test_qubo_matches_manual_cost()
    test_ising_adapter_scale()
    test_exact_is_oracle()
    print("\nTodos los tests pasaron.")
