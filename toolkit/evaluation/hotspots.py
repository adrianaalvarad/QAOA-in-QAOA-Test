"""Extraccion multi-hotspot: en vez de refinar una sola region por corrida,
detecta varios focos de conflicto independientes (via deteccion de
comunidades) y los devuelve ordenados por costo residual, sin solaparse.

Generaliza get_multiple_residual_subgraphs de
QAOA-in-QAOA-Test/src/evaluation/cost_function.py a k general: esa version
original calcula ademas un "boundary field" escalar h_u = 0.5*sum(w*(+-1))
que asume k=2 explicitamente. Aca NO se calcula ningun campo de frontera --
toolkit/qubo.py::build_qubo ya distingue vecino interno/externo del bloque
por si sola, para cualquier k, asi que esta funcion solo necesita devolver
listas de nodos.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from toolkit import _bootstrap  # noqa: F401  (side effect: src/ en sys.path)
from evaluation.cost_function import compute_node_residuals  # de QAOA-in-QAOA-Test/src


@dataclass
class Hotspot:
    nodes: list
    subgraph: nx.Graph
    density: float
    residual_total: float


def subgraph_density(subg: nx.Graph) -> float:
    n = subg.number_of_nodes()
    if n <= 1:
        return 0.0
    return (2.0 * subg.number_of_edges()) / (n * (n - 1))


def extract_hotspots(graph: nx.Graph, coloring: dict, config) -> list[Hotspot]:
    residuals = compute_node_residuals(graph, coloring)
    pos_res = [r for r in residuals.values() if r > 1e-6]
    if not pos_res:
        return []

    import numpy as np

    threshold = float(np.percentile(pos_res, config.threshold_percentile))
    hot_nodes = [n for n, r in residuals.items() if r >= threshold]
    if len(hot_nodes) < config.min_hotspot_size:
        hot_nodes = sorted(residuals.keys(), key=lambda n: residuals[n], reverse=True)[
            : config.min_hotspot_size
        ]

    sub_hot = graph.subgraph(hot_nodes)
    try:
        from networkx.algorithms.community import greedy_modularity_communities

        components = [list(c) for c in greedy_modularity_communities(sub_hot)]
    except Exception:
        components = [list(c) for c in nx.connected_components(sub_hot)]

    components.sort(key=lambda c: sum(residuals[n] for n in c), reverse=True)

    hotspots: list[Hotspot] = []
    used_nodes: set = set()

    for comp in components:
        comp_nodes = [n for n in comp if n not in used_nodes]
        if len(comp_nodes) < config.min_hotspot_size:
            continue
        if len(comp_nodes) > config.max_hotspot_size:
            comp_nodes = sorted(comp_nodes, key=lambda n: residuals[n], reverse=True)[
                : config.max_hotspot_size
            ]

        subg = graph.subgraph(comp_nodes).copy()
        hotspots.append(
            Hotspot(
                nodes=comp_nodes,
                subgraph=subg,
                density=subgraph_density(subg),
                residual_total=sum(residuals[n] for n in comp_nodes),
            )
        )
        used_nodes.update(comp_nodes)
        if len(hotspots) >= config.max_subgraphs:
            break

    if len(hotspots) < 2 and len(hot_nodes) >= config.min_hotspot_size:
        remaining = [n for n in sorted(hot_nodes, key=lambda n: residuals[n], reverse=True) if n not in used_nodes]
        while len(remaining) >= config.min_hotspot_size and len(hotspots) < config.max_subgraphs:
            chunk = remaining[: config.max_hotspot_size]
            subg = graph.subgraph(chunk).copy()
            hotspots.append(
                Hotspot(
                    nodes=chunk,
                    subgraph=subg,
                    density=subgraph_density(subg),
                    residual_total=sum(residuals[n] for n in chunk),
                )
            )
            used_nodes.update(chunk)
            remaining = remaining[config.max_hotspot_size :]

    hotspots.sort(key=lambda h: h.residual_total, reverse=True)
    return hotspots
