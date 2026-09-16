"""
QAOA-in-QAOA (QAOA^2) Divide-and-Conquer Recursive Partitioning and Agnostic Refinement Layer.
Based on Zhou et al. (Phys. Rev. Appl. 2023) adapted to Qiskit with warm-starting and fallback safety.
"""

import math
from typing import Any, Dict, List, Tuple
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

from evaluation.cost_function import (
    compute_interference_cost,
    get_high_residual_subgraph,
    get_multiple_residual_subgraphs,
)
from quantum.qaoa import run_qaoa_subgraph
from quantum.r_qaoa import solve_rqaoa


def partition_into_subgraphs(subg: nx.Graph, max_sub_size: int = 8) -> List[List[int]]:
    """
    Partitions an induced subgraph into clusters of size <= max_sub_size
    using greedy modularity community detection.
    """
    nodes = list(subg.nodes())
    if len(nodes) <= max_sub_size:
        return [nodes]
        
    try:
        # Detect communities using NetworkX
        communities = list(greedy_modularity_communities(subg))
    except Exception:
        communities = [nodes]
        
    partition_list: List[List[int]] = []
    for comm in communities:
        comm_nodes = list(comm)
        if len(comm_nodes) <= max_sub_size:
            partition_list.append(comm_nodes)
        else:
            # Subdivide oversized community
            num_chunks = math.ceil(len(comm_nodes) / max_sub_size)
            for k in range(num_chunks):
                chunk = comm_nodes[k * max_sub_size : (k + 1) * max_sub_size]
                if chunk:
                    partition_list.append(chunk)
                    
    return partition_list


def qaoa_square_solve(
    subg: nx.Graph,
    boundary_fields: Dict[int, float],
    candidate_sub_sol: Dict[int, int],
    max_sub_size: int = 8,
) -> Dict[int, int]:
    """
    Solves the subregion optimization problem using Zhou et al.'s QAOA^2 recursive method.
    
    1. Partitions subg into clusters H_1, ..., H_m of size <= max_sub_size.
    2. Runs warm-started QAOA on each local cluster H_i to find relative spin configuration s_i.
    3. Builds contracted super-graph of m super-nodes with exact effective interactions J_ij and h_i.
    4. Solves contracted super-graph with QAOA to determine relative orientation Z_i of each cluster.
    5. Reconstructs assignments: c_u = s_i[u] ^ Z_i.
    """
    sub_nodes = list(subg.nodes())
    n_nodes = len(sub_nodes)
    
    if n_nodes == 0:
        return {}
    if n_nodes <= max_sub_size:
        # Direct local QAOA without hierarchical reduction
        node_to_idx = {node: i for i, node in enumerate(sub_nodes)}
        edges_idx = [
            (node_to_idx[u], node_to_idx[v], data.get("weight", 1.0))
            for u, v, data in subg.edges(data=True)
        ]
        fields_idx = {node_to_idx[u]: boundary_fields.get(u, 0.0) for u in sub_nodes}
        warm_bits = [candidate_sub_sol.get(u, 0) for u in sub_nodes]
        
        _, best_bits = run_qaoa_subgraph(
            num_qubits=n_nodes,
            edges=edges_idx,
            linear_fields=fields_idx,
            warm_start_bits=warm_bits,
            p=1,
            maxiter=20,
        )
        return {node: best_bits[node_to_idx[node]] for node in sub_nodes}

    # Step 1: Partition into clusters
    clusters = partition_into_subgraphs(subg, max_sub_size=max_sub_size)
    m = len(clusters)
    
    # Step 2: Solve each cluster independently with QAOA
    cluster_sols: Dict[int, Dict[int, int]] = {}
    for c_idx, cluster in enumerate(clusters):
        c_subg = subg.subgraph(cluster).copy()
        c_to_idx = {node: i for i, node in enumerate(cluster)}
        c_edges = [
            (c_to_idx[u], c_to_idx[v], data.get("weight", 1.0))
            for u, v, data in c_subg.edges(data=True)
        ]
        c_fields = {c_to_idx[u]: boundary_fields.get(u, 0.0) for u in cluster}
        c_warm = [candidate_sub_sol.get(u, 0) for u in cluster]
        
        _, c_bits = run_qaoa_subgraph(
            num_qubits=len(cluster),
            edges=c_edges,
            linear_fields=c_fields,
            warm_start_bits=c_warm,
            p=1,
            maxiter=15,
        )
        cluster_sols[c_idx] = {node: c_bits[c_to_idx[node]] for node in cluster}

    # Step 3: Form contracted super-graph (super-nodes 0, ..., m - 1)
    # Relative spin inside cluster: sigma_u = +1 if s_i[u] == 0 else -1
    node_to_cluster = {}
    node_sigma = {}
    for c_idx, cluster in enumerate(clusters):
        for u in cluster:
            node_to_cluster[u] = c_idx
            node_sigma[u] = 1.0 if cluster_sols[c_idx][u] == 0 else -1.0

    # Contracted edge interactions J_ij
    super_edges_dict: Dict[Tuple[int, int], float] = {}
    for u, v, data in subg.edges(data=True):
        c_u = node_to_cluster[u]
        c_v = node_to_cluster[v]
        if c_u != c_v:
            pair = (min(c_u, c_v), max(c_u, c_v))
            w = float(data.get("weight", 1.0))
            # J_ij = sum w_uv * sigma_u * sigma_v
            super_edges_dict[pair] = super_edges_dict.get(pair, 0.0) + w * node_sigma[u] * node_sigma[v]

    super_edges = [(pair[0], pair[1], weight) for pair, weight in super_edges_dict.items() if abs(weight) > 1e-5]
    
    # Contracted boundary fields
    super_fields: Dict[int, float] = {}
    for c_idx, cluster in enumerate(clusters):
        h_super = sum(node_sigma[u] * boundary_fields.get(u, 0.0) for u in cluster)
        if abs(h_super) > 1e-5:
            super_fields[c_idx] = h_super

    # Warm start for super-spins: majority orientation from classical solution
    super_warm: List[int] = []
    for c_idx, cluster in enumerate(clusters):
        votes_1 = sum(1 for u in cluster if candidate_sub_sol.get(u, 0) != cluster_sols[c_idx][u])
        super_warm.append(1 if votes_1 > len(cluster) // 2 else 0)

    # Step 4: Solve contracted super-graph
    if m <= max_sub_size:
        _, super_bits = run_qaoa_subgraph(
            num_qubits=m,
            edges=super_edges,
            linear_fields=super_fields,
            warm_start_bits=super_warm,
            p=1,
            maxiter=20,
        )
    else:
        # Recursive step if super-graph exceeds max_sub_size
        super_g = nx.Graph()
        for i in range(m):
            super_g.add_node(i)
        for u, v, w in super_edges:
            super_g.add_edge(u, v, weight=w)
        super_res_dict = qaoa_square_solve(
            super_g, super_fields, {i: super_warm[i] for i in range(m)}, max_sub_size=max_sub_size
        )
        super_bits = [super_res_dict.get(i, 0) for i in range(m)]

    # Step 5: Reconstruct global assignment for subg
    refined_sub_sol: Dict[int, int] = {}
    for c_idx, cluster in enumerate(clusters):
        z_c = super_bits[c_idx]  # 0 or 1
        for u in cluster:
            refined_sub_sol[u] = cluster_sols[c_idx][u] ^ z_c

    return refined_sub_sol


def refine(
    graph: nx.Graph,
    candidate_solution: Dict[int, int],
    target_fraction: float = 0.20,
    max_sub_size: int = 8,
    use_multi_subgraphs: bool = True,
    threshold: Optional[float] = None,
    threshold_percentile: float = 70.0,
    max_subgraphs: int = 4,
    density_threshold: float = 0.15,
) -> Tuple[Dict[int, int], Dict[str, Any]]:
    """
    Agnostic Adaptive Quantum Refinement Layer.
    
    Signature: refine(graph, candidate_solution, ...) -> (solution, info)
    
    Key Capabilities:
      1. Multi-Subgraph Hotspot Extraction: Selects multiple high-residual conflict
         subgraphs based on an interference threshold (instead of a single isolated block).
      2. Adaptive Quantum Routing:
           - If subgraph density < density_threshold (e.g. < 15%):
             Routes to R-QAOA (Recursive QAOA, Bravyi et al. 2020) for optimal
             correlation-based reduction on sparse paths.
           - If subgraph density >= density_threshold (e.g. >= 15%):
             Routes to QAOA-in-QAOA (QAOA^2, Zhou et al. 2023) with hierarchical
             community partitioning and super-graph contraction.
      3. Fallback Safety Invariant:
           Guarantees both local (per-subgraph) and global fallback protection,
           ensuring Cost(hybrid) <= Cost(candidate) in 100% of runs.
    """
    initial_cost = compute_interference_cost(graph, candidate_solution)
    current_solution = candidate_solution.copy()
    current_cost = initial_cost
    
    subgraphs_details = []
    
    if use_multi_subgraphs:
        subgraphs_data = get_multiple_residual_subgraphs(
            graph,
            current_solution,
            threshold=threshold,
            threshold_percentile=threshold_percentile,
            min_sub_size=4,
            max_sub_size=max_sub_size,
            max_subgraphs=max_subgraphs,
        )
    else:
        subgraphs_data = []

    if not subgraphs_data:
        # Fallback to single high-residual subgraph extraction
        target_nodes, subg, boundary_fields = get_high_residual_subgraph(
            graph,
            current_solution,
            target_fraction=target_fraction,
            min_nodes=6,
            max_nodes=max_sub_size,
        )
        n_c = len(target_nodes)
        n_e = subg.number_of_edges()
        dens = (2.0 * n_e) / (n_c * (n_c - 1)) if n_c > 1 else 0.0
        subgraphs_data = [(target_nodes, subg, boundary_fields, dens)]

    # Compute global graph density to assess macro topology
    num_total_nodes = graph.number_of_nodes()
    global_density = (
        (2.0 * graph.number_of_edges()) / (num_total_nodes * (num_total_nodes - 1))
        if num_total_nodes > 1
        else 0.0
    )

    for idx, (target_nodes, subg, _, density) in enumerate(subgraphs_data):
        target_set = set(target_nodes)
        
        # Calculate fresh boundary fields with respect to current_solution
        b_fields = {u: 0.0 for u in target_nodes}
        for u in target_nodes:
            for v in graph.neighbors(u):
                if v not in target_set:
                    w = graph[u][v].get("weight", 1.0)
                    c_v = current_solution.get(v, 0)
                    z_v = 1.0 if c_v == 0 else -1.0
                    b_fields[u] += 0.5 * float(w) * z_v
                    
        cand_sub = {u: current_solution[u] for u in target_nodes}
        
        # Adaptive quantum solver routing:
        # Sparse topologies (global density or subgraph density < density_threshold, e.g. < 15%)
        # route to R-QAOA (Bravyi et al. 2020), which excels on long paths and sparse graphs.
        # Dense topologies (>= density_threshold, e.g. dense_100 microcells)
        # route to QAOA-in-QAOA (QAOA^2, Zhou et al. 2023) using modular contraction.
        is_sparse = (global_density < density_threshold) or (density < density_threshold)
        
        if is_sparse:
            solver_name = "R-QAOA"
            refined_sub = solve_rqaoa(
                subg,
                boundary_fields=b_fields,
                candidate_sub_sol=cand_sub,
                cutoff_nodes=3,
                p=1,
                maxiter=15,
            )
        else:
            solver_name = "QAOA^2"
            refined_sub = qaoa_square_solve(
                subg,
                boundary_fields=b_fields,
                candidate_sub_sol=cand_sub,
                max_sub_size=max_sub_size,
            )

            
        # Test proposed local update
        test_solution = current_solution.copy()
        for u, color in refined_sub.items():
            test_solution[u] = color
            
        test_cost = compute_interference_cost(graph, test_solution)
        
        if test_cost < current_cost:
            delta_local = round(current_cost - test_cost, 4)
            current_solution = test_solution
            current_cost = test_cost
            sub_status = "IMPROVED"
        else:
            delta_local = 0.0
            sub_status = "FALLBACK_PRESERVED"
            
        subgraphs_details.append({
            "subgraph_index": idx + 1,
            "num_nodes": len(target_nodes),
            "num_edges": subg.number_of_edges(),
            "density": round(density, 3),
            "solver": solver_name,
            "status": sub_status,
            "delta_local": delta_local,
        })

    # Overall Fallback Verification
    if current_cost < initial_cost:
        final_solution = current_solution
        final_cost = current_cost
        status = "IMPROVED"
        delta_abs = round(initial_cost - current_cost, 4)
        delta_pct = round((delta_abs / initial_cost) * 100.0, 2) if initial_cost > 0 else 0.0
    else:
        final_solution = candidate_solution.copy()
        final_cost = initial_cost
        status = "FALLBACK_PRESERVED"
        delta_abs = 0.0
        delta_pct = 0.0

    info = {
        "status": status,
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "delta_abs": delta_abs,
        "delta_pct": delta_pct,
        "subregion_size": sum(s["num_nodes"] for s in subgraphs_details),
        "num_subgraphs": len(subgraphs_details),
        "subgraphs": subgraphs_details,
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "fallback_triggered": (status == "FALLBACK_PRESERVED"),
    }
    
    return final_solution, info

