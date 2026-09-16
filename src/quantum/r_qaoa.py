"""
Recursive QAOA (R-QAOA) Implementation in Qiskit for Sparse Graph Optimization.
Based on Bravyi, Kliesch, Koenig, and Tang (Phys. Rev. Lett. 125, 200501, 2020).

R-QAOA overcomes locality horizon limitations of standard QAOA on sparse graphs
by recursively eliminating variables with the highest two-point correlation |<Z_u Z_v>|
or fixing single-variable orientations |<Z_u>| until a small base graph remains.
"""

from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize


def build_rqaoa_circuit(
    num_qubits: int,
    edges: List[Tuple[int, int, float]],
    linear_fields: Dict[int, float],
    gamma: List[float],
    beta: List[float],
    warm_start_bits: Optional[List[int]] = None,
    warm_start_bias: float = 0.15,
) -> QuantumCircuit:
    """
    Constructs parameterized QAOA circuit for R-QAOA with optional warm-start.
    """
    p = len(gamma)
    qc = QuantumCircuit(num_qubits)

    # Initial state preparation
    if warm_start_bits is not None:
        for i in range(num_qubits):
            b_i = warm_start_bits[i]
            theta_i = (
                2.0 * np.arcsin(np.sqrt(warm_start_bias))
                if b_i == 0
                else np.pi - 2.0 * np.arcsin(np.sqrt(warm_start_bias))
            )
            qc.ry(theta_i, i)
    else:
        for i in range(num_qubits):
            qc.h(i)

    # Alternating Cost and Mixer Unitaries
    for l in range(p):
        g = gamma[l]
        b = beta[l]

        # Cost unitary
        for u, v, w in edges:
            qc.rzz(g * float(w), u, v)

        for u, h in linear_fields.items():
            qc.rz(2.0 * g * float(h), u)

        # Mixer unitary
        for i in range(num_qubits):
            qc.rx(2.0 * b, i)

    return qc


def compute_correlations(
    num_qubits: int,
    sv: Statevector,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes exact single-site <Z_i> and two-site correlation matrix <Z_i Z_j>
    from the quantum statevector.
    
    Returns:
        z_single: (N,) array of expectation values <Z_i> in [-1, 1].
        z_corr: (N, N) array of correlation values <Z_i Z_j> in [-1, 1].
    """
    probs = sv.probabilities()
    z_single = np.zeros(num_qubits)
    z_corr = np.zeros((num_qubits, num_qubits))

    for idx, prob in enumerate(probs):
        if prob > 1e-6:
            bits = [(idx >> (num_qubits - 1 - i)) & 1 for i in range(num_qubits)]
            # Spin mapping: bit 0 -> +1, bit 1 -> -1
            spins = np.array([1.0 if b == 0 else -1.0 for b in bits])
            z_single += prob * spins
            z_corr += prob * np.outer(spins, spins)

    return z_single, z_corr


def solve_rqaoa(
    subg: nx.Graph,
    boundary_fields: Dict[int, float],
    candidate_sub_sol: Dict[int, int],
    cutoff_nodes: int = 3,
    p: int = 1,
    maxiter: int = 15,
) -> Dict[int, int]:
    """
    Solves interference minimization on subg using Recursive QAOA (R-QAOA).
    
    1. Evaluates two-point quantum correlations <Z_u Z_v> and single-site <Z_u>.
    2. Identifies the strongest correlation |<Z_u Z_v>| or bias |<Z_u>|.
    3. Eliminates one variable by enforcing Z_u = sign(<Z_u Z_v>) Z_v or freezing Z_u.
    4. Recursively repeats until <= cutoff_nodes remain.
    5. Solves the base problem exactly and backtracks all constraints.
    
    Args:
        subg: Target induced subgraph.
        boundary_fields: Linear bias h_u from external neighbors outside subg.
        candidate_sub_sol: Classical solution bitstring for warm start.
        cutoff_nodes: Number of nodes at which to switch to exact solver.
        p: Number of QAOA layers.
        maxiter: Maximum COBYLA iterations per recursive level.
        
    Returns:
        Dict mapping node ID -> refined channel index in {0, 1}.
    """
    nodes = list(subg.nodes())
    if not nodes:
        return {}
    if len(nodes) <= cutoff_nodes:
        return _exact_solve_subgraph(subg, boundary_fields)

    # Elimination history: list of (eliminated_node, target_node, sign)
    # If target_node is None, eliminated_node was frozen to sign (+1 or -1)
    constraints: List[Tuple[int, Optional[int], int]] = []

    current_g = subg.copy()
    current_h = {u: boundary_fields.get(u, 0.0) for u in nodes}
    current_warm = {u: candidate_sub_sol.get(u, 0) for u in nodes}

    while current_g.number_of_nodes() > cutoff_nodes and current_g.number_of_edges() > 0:
        active_nodes = list(current_g.nodes())
        n_act = len(active_nodes)
        node_to_idx = {node: i for i, node in enumerate(active_nodes)}
        edges_idx = [
            (node_to_idx[u], node_to_idx[v], data.get("weight", 1.0))
            for u, v, data in current_g.edges(data=True)
        ]
        fields_idx = {node_to_idx[u]: current_h.get(u, 0.0) for u in active_nodes}
        warm_bits = [current_warm.get(u, 0) for u in active_nodes]

        # Optimize QAOA angles for reduced instance
        def obj_fn(params: np.ndarray) -> float:
            gammas = list(params[:p])
            betas = list(params[p:])
            qc = build_rqaoa_circuit(
                n_act, edges_idx, fields_idx, gammas, betas, warm_start_bits=warm_bits
            )
            sv = Statevector(qc)
            z_s, z_c = compute_correlations(n_act, sv)
            energy = sum(0.5 * w * z_c[u, v] for u, v, w in edges_idx)
            energy += sum(h * z_s[u] for u, h in fields_idx.items())
            return float(energy)

        res = minimize(
            obj_fn,
            np.array([0.4] * p + [0.4] * p),
            method="COBYLA",
            options={"maxiter": maxiter},
        )
        opt_params = res.x

        qc_opt = build_rqaoa_circuit(
            n_act,
            edges_idx,
            fields_idx,
            list(opt_params[:p]),
            list(opt_params[p:]),
            warm_start_bits=warm_bits,
        )
        sv_opt = Statevector(qc_opt)
        z_single, z_corr = compute_correlations(n_act, sv_opt)

        # 1. Find strongest correlation on active edges
        best_edge = None
        max_edge_corr = -1.0
        best_sign = 1
        for u, v, w in edges_idx:
            c_val = z_corr[u, v]
            if abs(c_val) > max_edge_corr:
                max_edge_corr = abs(c_val)
                best_edge = (active_nodes[u], active_nodes[v])
                best_sign = 1 if c_val >= 0 else -1

        # 2. Find strongest single-site bias
        best_single = None
        max_single_corr = -1.0
        best_single_sign = 1
        for u in range(n_act):
            s_val = z_single[u]
            if abs(s_val) > max_single_corr:
                max_single_corr = abs(s_val)
                best_single = active_nodes[u]
                best_single_sign = 1 if s_val >= 0 else -1

        # 3. Eliminate variable with strongest correlation
        if max_edge_corr >= max_single_corr and best_edge is not None:
            u_node, v_node = best_edge
            constraints.append((u_node, v_node, best_sign))

            # Transfer edge couplings from u_node to v_node
            for neighbor in list(current_g.neighbors(u_node)):
                if neighbor != v_node:
                    w_un = current_g[u_node][neighbor].get("weight", 1.0)
                    new_w = best_sign * float(w_un)
                    if current_g.has_edge(v_node, neighbor):
                        current_g[v_node][neighbor]["weight"] += new_w
                    else:
                        current_g.add_edge(v_node, neighbor, weight=new_w)

            # Transfer linear fields
            current_h[v_node] = current_h.get(v_node, 0.0) + best_sign * current_h.get(u_node, 0.0)
            current_g.remove_node(u_node)
            current_h.pop(u_node, None)
            current_warm.pop(u_node, None)
        elif best_single is not None:
            u_node = best_single
            s_val = best_single_sign
            constraints.append((u_node, None, s_val))
            for neighbor in list(current_g.neighbors(u_node)):
                w_un = current_g[u_node][neighbor].get("weight", 1.0)
                current_h[neighbor] = current_h.get(neighbor, 0.0) + 0.5 * s_val * float(w_un)
            current_g.remove_node(u_node)
            current_h.pop(u_node, None)
            current_warm.pop(u_node, None)
        else:
            break

    # Solve base graph exactly
    base_sol = _exact_solve_subgraph(current_g, current_h)

    # Backtrack elimination constraints in reverse
    final_sol = dict(base_sol)
    for u_node, v_node, sign in reversed(constraints):
        if v_node is not None:
            v_color = final_sol.get(v_node, 0)
            v_spin = 1 if v_color == 0 else -1
            u_spin = sign * v_spin
            final_sol[u_node] = 0 if u_spin >= 0 else 1
        else:
            final_sol[u_node] = 0 if sign >= 0 else 1

    return final_sol


def _exact_solve_subgraph(subg: nx.Graph, linear_fields: Dict[int, float]) -> Dict[int, int]:
    """Exact brute-force solver for base subgraphs of size <= 10."""
    nodes = list(subg.nodes())
    n = len(nodes)
    if n == 0:
        return {}
    if n > 12:
        return {node: 0 for node in nodes}

    best_cost = float("inf")
    best_config = [0] * n
    edges = [
        (nodes.index(u), nodes.index(v), data.get("weight", 1.0))
        for u, v, data in subg.edges(data=True)
    ]
    fields = [linear_fields.get(node, 0.0) for node in nodes]

    for bit_int in range(1 << n):
        bits = [(bit_int >> (n - 1 - i)) & 1 for i in range(n)]
        cost = 0.0
        for u_i, v_i, w in edges:
            if bits[u_i] == bits[v_i]:
                cost += float(w)
        for i, h in enumerate(fields):
            z_i = 1.0 if bits[i] == 0 else -1.0
            cost += float(h) * z_i

        if cost < best_cost:
            best_cost = cost
            best_config = bits

    return {nodes[i]: best_config[i] for i in range(n)}
