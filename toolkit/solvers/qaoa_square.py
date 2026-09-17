"""QAOA² / QAOA-in-QAOA jerárquico (adaptación de Zhou et al. 2023). Portado
de QAOA-in-QAOA-Test/src/quantum/qaoa_square.py.

La partición en clusters + contracción a super-grafo es algorítmicamente
genérica y se porta sin cambios de lógica. El solver de hoja original
(`run_qaoa_subgraph`/`compute_bitstring_cost`, en `qaoa.py`) asumía
literalmente "bit = color" (válido solo para k=2 crudo) y además usaba una
convención de escala de aristas DISTINTA a la de r_qaoa.py (`rzz(2*gamma*w)`
en vez de `rzz(gamma*w)`, es decir w=J directo, no w=2J) -- una
inconsistencia real entre los dos archivos del repo original. Para evitar
heredar esa inconsistencia, esta version resuelve cada hoja (cluster o
super-grafo) con el mismo nucleo compartido de _qaoa_core.py que usa
direct_qaoa.py, sobre la energia de Ising directa (h, J) -- generico para
cualquier k, y con una sola convencion de escala en todo el kit.

Esto tambien es la pieza que cierra el hueco de
poc-qaoa-agnostic-refinement/src/quantum/qaoa_refine.py::_partition_into_blocks:
ahi cada bloque se resuelve independiente, sin reconciliacion entre bloques.
Aca el paso 4 (resolver el super-grafo) es exactamente esa reconciliacion.
"""

from __future__ import annotations

import math

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

from toolkit.solvers._ising_adapter import block_to_ising_graph
from toolkit.solvers._qaoa_core import ising_energy, run_warm_start_qaoa
from toolkit.solvers.base import BlockQUBO, BlockResult, SubproblemMeta

name = "qaoa_square"


def _partition_into_clusters(subg: nx.Graph, max_cluster_size: int) -> list:
    nodes = list(subg.nodes())
    if len(nodes) <= max_cluster_size:
        return [nodes]

    try:
        communities = list(greedy_modularity_communities(subg))
    except Exception:
        communities = [nodes]

    clusters = []
    for comm in communities:
        comm_nodes = list(comm)
        if len(comm_nodes) <= max_cluster_size:
            clusters.append(comm_nodes)
        else:
            num_chunks = math.ceil(len(comm_nodes) / max_cluster_size)
            for k in range(num_chunks):
                chunk = comm_nodes[k * max_cluster_size : (k + 1) * max_cluster_size]
                if chunk:
                    clusters.append(chunk)
    return clusters


def _solve_ising_graph(g: nx.Graph, h_dict: dict, config, seed: int, max_cluster_size: int) -> dict:
    """Resuelve un grafo de Ising general (aristas en convencion weight=2*J,
    igual que _ising_adapter) de forma jerarquica. Devuelve {nodo: bit}.
    """
    nodes = list(g.nodes())
    n_nodes = len(nodes)
    if n_nodes == 0:
        return {}

    if n_nodes <= max_cluster_size:
        idx = {node: i for i, node in enumerate(nodes)}
        h = [h_dict.get(node, 0.0) for node in nodes]
        j_coeffs: dict = {}
        for u, v, data in g.edges(data=True):
            w = data.get("weight", 0.0)  # convencion 2*J -> J = w/2
            if abs(w) > 1e-12:
                key = (idx[u], idx[v])
                j_coeffs[key] = j_coeffs.get(key, 0.0) + w / 2.0

        best_bits, _ = run_warm_start_qaoa(
            n_nodes, h, j_coeffs,
            p=config.qaoa2_p, shots=config.direct_qaoa_shots, maxiter=config.qaoa2_maxiter,
            epsilon=config.direct_qaoa_epsilon, top_k=config.direct_qaoa_top_k, seed=seed,
            evaluate_fn=lambda bits: ising_energy(bits, h, j_coeffs),
        )
        return {nodes[i]: best_bits[i] for i in range(n_nodes)}

    # Paso 1: particionar en clusters
    clusters = _partition_into_clusters(g, max_cluster_size)
    m = len(clusters)

    # Paso 2: resolver cada cluster de forma independiente
    cluster_sols = {}
    for c_idx, cluster in enumerate(clusters):
        c_g = g.subgraph(cluster).copy()
        c_h = {u: h_dict.get(u, 0.0) for u in cluster}
        cluster_sols[c_idx] = _solve_ising_graph(c_g, c_h, config, seed + c_idx, max_cluster_size)

    # Paso 3: super-grafo contraido (espin relativo dentro de cada cluster)
    node_to_cluster = {}
    node_sigma = {}
    for c_idx, cluster in enumerate(clusters):
        for u in cluster:
            node_to_cluster[u] = c_idx
            node_sigma[u] = 1.0 if cluster_sols[c_idx][u] == 0 else -1.0

    super_edges: dict = {}
    for u, v, data in g.edges(data=True):
        c_u, c_v = node_to_cluster[u], node_to_cluster[v]
        if c_u != c_v:
            pair = (min(c_u, c_v), max(c_u, c_v))
            w = data.get("weight", 0.0)
            super_edges[pair] = super_edges.get(pair, 0.0) + w * node_sigma[u] * node_sigma[v]

    super_g = nx.Graph()
    super_g.add_nodes_from(range(m))
    for (i, j), w in super_edges.items():
        if abs(w) > 1e-12:
            super_g.add_edge(i, j, weight=w)

    super_h = {}
    for c_idx, cluster in enumerate(clusters):
        h_super = sum(node_sigma[u] * h_dict.get(u, 0.0) for u in cluster)
        if abs(h_super) > 1e-12:
            super_h[c_idx] = h_super

    # Paso 4: resolver el super-grafo -- esto ES la reconciliacion entre
    # bloques que qaoa_refine.py no hace (recursivo si m sigue siendo grande)
    super_bits_dict = _solve_ising_graph(super_g, super_h, config, seed + 1000, max_cluster_size)
    super_bits = [super_bits_dict.get(i, 0) for i in range(m)]

    # Paso 5: reconstruir asignacion global: bit final = bit del cluster XOR bit del super-nodo
    refined = {}
    for c_idx, cluster in enumerate(clusters):
        z_c = super_bits[c_idx]
        for u in cluster:
            refined[u] = cluster_sols[c_idx][u] ^ z_c

    return refined


def solve(block: BlockQUBO, meta: SubproblemMeta, config, seed: int) -> BlockResult:
    graph, boundary_fields, _ = block_to_ising_graph(block)
    bits_by_idx = _solve_ising_graph(graph, boundary_fields, config, seed, config.qaoa2_cluster_size)
    x_bits = [bits_by_idx.get(i, 0) for i in range(block.n)]
    return BlockResult(x_bits=x_bits, solver_name=name, strategy_name=block.strategy_name, meta={})
