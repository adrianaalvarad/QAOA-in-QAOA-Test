"""Reduccion binaria por-nodo del problema de k-coloreo a QUBO/Ising, y las
estrategias de candidato por nodo. Portado de
poc-qaoa-agnostic-refinement/src/quantum/qaoa_refine.py -- es la pieza que
hace que todo lo demas en este kit sea agnostico a k: cada nodo del bloque
recibe una variable binaria x_i (0 = mantener su color actual, 1 = cambiar
al candidato alt_colors[i]), sin importar cuantos colores totales existan.
"""

from __future__ import annotations

import networkx as nx


def individual_best_alt_color(graph: nx.Graph, node, coloring: dict, n_colors: int) -> int:
    """Mejor color alternativo para `node`, asumiendo que sus vecinos no se mueven."""
    conflict_by_color = [0.0] * n_colors
    for nbr in graph.neighbors(node):
        conflict_by_color[coloring[nbr]] += graph[node][nbr].get("weight", 1.0)

    current = coloring[node]
    candidates = [c for c in range(n_colors) if c != current]
    return min(candidates, key=lambda c: (conflict_by_color[c], c))


def swap_alt_color(graph: nx.Graph, node, coloring: dict, block_set: set):
    """Color del vecino del mismo bloque con la arista de mayor peso.

    Ofrece explicitamente "tomar el color de mi vecino mas caro" para que el
    termino cuadratico del QUBO pueda evaluar un intercambio conjunto -- ver
    docs/Diagnostico_DSATUR.md de poc-qaoa-agnostic-refinement para el caso
    real que motivo esta estrategia. None si no hay vecino en el bloque.
    """
    heaviest_nbr, heaviest_w = None, -1.0
    for nbr in graph.neighbors(node):
        if nbr not in block_set:
            continue
        w = graph[node][nbr].get("weight", 1.0)
        if w > heaviest_w:
            heaviest_nbr, heaviest_w = nbr, w
    if heaviest_nbr is None or coloring[heaviest_nbr] == coloring[node]:
        return None
    return coloring[heaviest_nbr]


def build_alt_color_strategies(
    graph: nx.Graph, block_nodes: list, coloring: dict, n_colors: int
) -> dict[str, dict]:
    """Devuelve {"individual": {...}, "swap": {...}}, un diccionario alt_colors por estrategia."""
    block_set = set(block_nodes)
    individual = {
        node: individual_best_alt_color(graph, node, coloring, n_colors)
        for node in block_nodes
    }
    swap = {}
    for node in block_nodes:
        candidate = swap_alt_color(graph, node, coloring, block_set)
        swap[node] = candidate if candidate is not None else individual[node]
    return {"individual": individual, "swap": swap}


def build_qubo(graph: nx.Graph, block_nodes: list, coloring: dict, alt_colors: dict):
    """QUBO exacto: costo de las aristas que tocan el bloque, en funcion de
    las decisiones binarias x_i (0 = mantener, 1 = cambiar a la alternativa).

    Reformulacion exacta, no heuristica: para cualquier asignacion binaria,
    el valor devuelto reproduce el costo real del grafo completo restringido
    a las aristas del bloque -- sin importar cuantos colores totales (k) haya.
    """
    index = {node: i for i, node in enumerate(block_nodes)}
    block_set = set(block_nodes)
    linear = [0.0] * len(block_nodes)
    quad: dict[tuple[int, int], float] = {}
    seen_edges = set()

    for node in block_nodes:
        i = index[node]
        for nbr in graph.neighbors(node):
            edge_key = frozenset((node, nbr))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            w = graph[node][nbr].get("weight", 1.0)
            f00 = w if coloring[node] == coloring[nbr] else 0.0

            if nbr in block_set:
                j = index[nbr]
                f10 = w if alt_colors[node] == coloring[nbr] else 0.0
                f01 = w if coloring[node] == alt_colors[nbr] else 0.0
                f11 = w if alt_colors[node] == alt_colors[nbr] else 0.0
                linear[i] += f10 - f00
                linear[j] += f01 - f00
                key = (i, j) if i < j else (j, i)
                quad[key] = quad.get(key, 0.0) + (f11 - f10 - f01 + f00)
            else:
                f1 = w if alt_colors[node] == coloring[nbr] else 0.0
                linear[i] += f1 - f00

    return linear, quad


def qubo_cost(bits: list[int], linear: list[float], quad: dict[tuple[int, int], float]) -> float:
    """Evalua el QUBO directamente sobre un bitstring x. Es la funcion de
    costo generica que deben usar TODOS los solvers (exact, r_qaoa, qaoa_square,
    direct_qaoa) -- nunca "bits[u]==bits[v]" literal, que solo tendria sentido
    si el bit fuera el color crudo (valido solo para k=2, ver
    solvers/_qubo_cost.py y docs/ARCHITECTURE.md, seccion del adaptador).
    """
    cost = sum(linear[i] * bits[i] for i in range(len(bits)))
    cost += sum(coeff * bits[i] * bits[j] for (i, j), coeff in quad.items())
    return cost


def qubo_to_ising(linear: list[float], quad: dict[tuple[int, int], float], n: int):
    """QUBO (x in {0,1}) -> Ising (z in {-1,+1}) via x_i = (1-z_i)/2.

    h_i = -linear_i/2 - sum_j quad_ij/4 ; J_ij = quad_ij/4. El termino
    constante se descarta -- no afecta el argmin, y el costo real de cada
    propuesta se recalcula exacto al final via qubo_cost, nunca se infiere
    de la energia de Ising.
    """
    h = [0.0] * n
    j_coeffs: dict[tuple[int, int], float] = {}
    for i, coeff in enumerate(linear):
        h[i] += -coeff / 2
    for (i, j), coeff in quad.items():
        h[i] += -coeff / 4
        h[j] += -coeff / 4
        j_coeffs[(i, j)] = j_coeffs.get((i, j), 0.0) + coeff / 4
    return h, j_coeffs


def decode_bits_to_coloring(bits: list[int], block_nodes: list, coloring: dict, alt_colors: dict) -> dict:
    """x_i=1 -> el nodo toma alt_colors[node]; x_i=0 -> mantiene su color actual."""
    trial = dict(coloring)
    for i, node in enumerate(block_nodes):
        if bits[i]:
            trial[node] = alt_colors[node]
    return trial
