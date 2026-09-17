"""Orquestador de todo el kit: refine(graph, solution, n_colors, config) ->
(solucion_refinada, reporte). Cada decision de ruteo real queda anotada en
el reporte -- no solo el resultado final -- siguiendo el diagrama de
toolkit/docs/ARCHITECTURE.md paso a paso.
"""

from __future__ import annotations

from dataclasses import replace

import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection

from toolkit import _bootstrap  # noqa: F401  (side effect: src/ en sys.path)
from evaluation.cost_function import compute_interference_cost  # de QAOA-in-QAOA-Test/src

from toolkit.config import ToolkitConfig
from toolkit.evaluation.hotspots import extract_hotspots, subgraph_density
from toolkit.qubo import build_alt_color_strategies, build_qubo, decode_bits_to_coloring
from toolkit.router import (
    HOTSPOT_ROUTE_BLOCK_LEVEL,
    HOTSPOT_ROUTE_PARTITION_RECURSE,
    HOTSPOT_ROUTE_QAOA_SQUARE,
    decide_block_solver,
    decide_hotspot_route,
    get_solver,
)
from toolkit.solvers.base import BlockQUBO, SubproblemMeta


def _global_density(graph: nx.Graph) -> float:
    n = graph.number_of_nodes()
    if n <= 1:
        return 0.0
    return (2.0 * graph.number_of_edges()) / (n * (n - 1))


def _bisect_into_blocks(subgraph: nx.Graph, max_qubits: int) -> list:
    nodes = list(subgraph.nodes())
    if len(nodes) <= max_qubits:
        return [nodes]
    if subgraph.number_of_edges() == 0:
        return [nodes[i : i + max_qubits] for i in range(0, len(nodes), max_qubits)]
    try:
        part_a, part_b = kernighan_lin_bisection(subgraph, weight="weight", seed=0)
    except Exception:
        mid = len(nodes) // 2
        part_a, part_b = set(nodes[:mid]), set(nodes[mid:])

    blocks = []
    for part in (part_a, part_b):
        if not part:
            continue
        blocks.extend(_bisect_into_blocks(subgraph.subgraph(part).copy(), max_qubits))
    return blocks


def _try_block(graph, block_nodes, coloring, config, hotspot_index, route, seed, solver_name=None):
    """Prueba ambas estrategias de candidato sobre `block_nodes`, resuelve
    con el solver indicado (o el que decida decide_block_solver si no se da
    uno explicito), y devuelve (mejor_coloring, mejor_costo, detalle_por_estrategia).
    """
    block_subg = graph.subgraph(block_nodes)
    density = subgraph_density(block_subg)

    if solver_name is None:
        block_decision = decide_block_solver(len(block_nodes), density, config)
        solver_name = block_decision.route
        reason = block_decision.reason
    else:
        reason = f"ruta de hotspot ya decidida: {route}"

    solver_module = get_solver(solver_name)
    strategies = build_alt_color_strategies(graph, block_nodes, coloring, config.n_colors)

    best_coloring = coloring
    best_cost = compute_interference_cost(graph, coloring)
    tried = []

    for strategy_name, alt_colors in strategies.items():
        linear, quad = build_qubo(graph, block_nodes, coloring, alt_colors)
        block_qubo = BlockQUBO(
            block_nodes=block_nodes, linear=linear, quad=quad,
            alt_colors=alt_colors, strategy_name=strategy_name,
        )
        meta = SubproblemMeta(
            n=len(block_nodes), n_edges_internal=block_subg.number_of_edges(),
            density=density, hotspot_index=hotspot_index, global_density=0.0, route=route,
        )
        result = solver_module.solve(block_qubo, meta, config, seed)
        trial = decode_bits_to_coloring(result.x_bits, block_nodes, coloring, alt_colors)
        trial_cost = compute_interference_cost(graph, trial)
        tried.append({"strategy": strategy_name, "solver": solver_name, "cost": trial_cost})
        if trial_cost < best_cost:
            best_cost = trial_cost
            best_coloring = trial

    return best_coloring, best_cost, {"solver": solver_name, "reason": reason, "tried": tried}


def refine(graph: nx.Graph, solution: dict, n_colors: int, config: ToolkitConfig | None = None) -> tuple:
    if config is None:
        config = ToolkitConfig(n_colors=n_colors)
    elif config.n_colors != n_colors:
        config = replace(config, n_colors=n_colors)

    initial_cost = compute_interference_cost(graph, solution)
    coloring = dict(solution)
    global_density = _global_density(graph)

    hotspots = extract_hotspots(graph, coloring, config)
    report: dict = {
        "initial_cost": initial_cost,
        "global_density": global_density,
        "n_hotspots": len(hotspots),
        "hotspots": [],
    }

    for h_idx, hotspot in enumerate(hotspots):
        n = len(hotspot.nodes)
        route_decision = decide_hotspot_route(n, hotspot.density, config)
        cost_before_hotspot = compute_interference_cost(graph, coloring)

        hotspot_report = {
            "index": h_idx, "n": n, "density": round(hotspot.density, 3),
            "route": route_decision.route, "reason": route_decision.reason,
            "blocks": [],
        }

        working_coloring = coloring

        if route_decision.route == HOTSPOT_ROUTE_BLOCK_LEVEL:
            blocks = [hotspot.nodes]
        elif route_decision.route == HOTSPOT_ROUTE_PARTITION_RECURSE:
            blocks = _bisect_into_blocks(hotspot.subgraph, config.max_qubits_for(hotspot.density))
        else:
            blocks = None  # R_QAOA_DIRECT o QAOA_SQUARE: se resuelve el hotspot entero de una vez

        if blocks is not None:
            for block_nodes in blocks:
                working_coloring, _, block_info = _try_block(
                    graph, block_nodes, working_coloring, config,
                    hotspot_index=h_idx, route=route_decision.route, seed=config.seed + h_idx,
                )
                block_info["n"] = len(block_nodes)
                hotspot_report["blocks"].append(block_info)
        else:
            forced_solver = "qaoa_square" if route_decision.route == HOTSPOT_ROUTE_QAOA_SQUARE else "r_qaoa"
            working_coloring, _, block_info = _try_block(
                graph, hotspot.nodes, working_coloring, config,
                hotspot_index=h_idx, route=route_decision.route, seed=config.seed + h_idx,
                solver_name=forced_solver,
            )
            block_info["n"] = n
            hotspot_report["blocks"].append(block_info)

        cost_after_hotspot = compute_interference_cost(graph, working_coloring)
        if cost_after_hotspot < cost_before_hotspot:
            coloring = working_coloring
            hotspot_report["adopted"] = True
            hotspot_report["delta"] = round(cost_before_hotspot - cost_after_hotspot, 4)
        else:
            hotspot_report["adopted"] = False
            hotspot_report["delta"] = 0.0

        report["hotspots"].append(hotspot_report)

    final_cost = compute_interference_cost(graph, coloring)
    if final_cost >= initial_cost - 1e-9:
        coloring = dict(solution)
        final_cost = initial_cost

    report["final_cost"] = final_cost
    report["delta_abs"] = round(initial_cost - final_cost, 4)
    report["delta_pct"] = round((report["delta_abs"] / initial_cost) * 100.0, 2) if initial_cost > 0 else 0.0
    report["never_worsens"] = final_cost <= initial_cost + 1e-9

    return coloring, report
