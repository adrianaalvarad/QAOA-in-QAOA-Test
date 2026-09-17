"""R-QAOA (Recursive QAOA, Bravyi, Kliesch, Koenig & Tang, Phys. Rev. Lett.
125, 200501, 2020). Portado de QAOA-in-QAOA-Test/src/quantum/r_qaoa.py.

La eliminacion recursiva por correlacion cuantica es algoritmicamente
generica y se porta SIN cambios de logica: en cada ronda corre un QAOA
circuit, calcula correlaciones exactas <Z_u> y <Z_u Z_v> via Statevector,
elimina la variable con la correlacion mas fuerte (fijandola en funcion de
su vecino, o congelandola si domina el sesgo de un punto), contrae el
grafo, y repite hasta un tamano base.

Lo que SI cambia respecto al original: el caso base (antes
`_exact_solve_subgraph`, que asumia literalmente "bit = color", valido solo
para k=2 crudo) ahora minimiza la energia de Ising estandar
h*z + 0.5*w*z_u*z_v -- generico para cualquier k, porque en nuestra
reduccion el bit representa "cambiar al candidato", no un color crudo.
Ver toolkit/solvers/_ising_adapter.py para el factor de escala w=2J que
esto exige en las aristas de entrada.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from toolkit.solvers._ising_adapter import block_to_ising_graph
from toolkit.solvers.base import BlockQUBO, BlockResult, SubproblemMeta

name = "r_qaoa"


def _build_rqaoa_circuit(num_qubits, edges, linear_fields, gamma, beta, warm_start_bits=None, warm_start_bias=0.15):
    p = len(gamma)
    qc = QuantumCircuit(num_qubits)

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

    for layer in range(p):
        g, b = gamma[layer], beta[layer]
        for u, v, w in edges:
            qc.rzz(g * float(w), u, v)
        for u, h_val in linear_fields.items():
            qc.rz(2.0 * g * float(h_val), u)
        for i in range(num_qubits):
            qc.rx(2.0 * b, i)

    return qc


def _compute_correlations(num_qubits, sv: Statevector):
    probs = sv.probabilities()
    z_single = np.zeros(num_qubits)
    z_corr = np.zeros((num_qubits, num_qubits))

    for idx, prob in enumerate(probs):
        if prob > 1e-6:
            bits = [(idx >> (num_qubits - 1 - i)) & 1 for i in range(num_qubits)]
            spins = np.array([1.0 if b == 0 else -1.0 for b in bits])
            z_single += prob * spins
            z_corr += prob * np.outer(spins, spins)

    return z_single, z_corr


def _exact_ising_base(subg, linear_fields: dict) -> dict:
    """Caso base: minimiza h*z + 0.5*w*z_u*z_v por fuerza bruta.

    `subg` trae peso=2*J_ij por arista (convencion de _ising_adapter,
    preservada por las actualizaciones de eliminacion de solve()) -- de ahi
    el factor 0.5 aca, que junto con w=2J reproduce J*z_u*z_v.
    """
    nodes = list(subg.nodes())
    n = len(nodes)
    if n == 0:
        return {}
    if n > 14:
        return {node: 0 for node in nodes}

    idx = {node: i for i, node in enumerate(nodes)}
    edges = [(idx[u], idx[v], data.get("weight", 0.0)) for u, v, data in subg.edges(data=True)]
    fields = [linear_fields.get(node, 0.0) for node in nodes]

    best_cost = None
    best_bits = [0] * n
    for bit_int in range(1 << n):
        bits = [(bit_int >> (n - 1 - i)) & 1 for i in range(n)]
        z = [1.0 if b == 0 else -1.0 for b in bits]
        cost = sum(fields[i] * z[i] for i in range(n))
        cost += sum(0.5 * w * z[u_i] * z[v_i] for u_i, v_i, w in edges)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_bits = bits

    return {nodes[i]: best_bits[i] for i in range(n)}


def _solve_rqaoa_on_graph(subg, boundary_fields, warm_start, cutoff_nodes, p, maxiter):
    nodes = list(subg.nodes())
    if not nodes:
        return {}
    if len(nodes) <= cutoff_nodes:
        return _exact_ising_base(subg, boundary_fields)

    constraints: list = []
    current_g = subg.copy()
    current_h = {u: boundary_fields.get(u, 0.0) for u in nodes}
    current_warm = {u: warm_start.get(u, 0) for u in nodes}

    while current_g.number_of_nodes() > cutoff_nodes and current_g.number_of_edges() > 0:
        active_nodes = list(current_g.nodes())
        n_act = len(active_nodes)
        node_to_idx = {node: i for i, node in enumerate(active_nodes)}
        edges_idx = [
            (node_to_idx[u], node_to_idx[v], data.get("weight", 0.0))
            for u, v, data in current_g.edges(data=True)
        ]
        fields_idx = {node_to_idx[u]: current_h.get(u, 0.0) for u in active_nodes}
        warm_bits = [current_warm.get(u, 0) for u in active_nodes]

        def obj_fn(params):
            gammas = list(params[:p])
            betas = list(params[p:])
            qc = _build_rqaoa_circuit(n_act, edges_idx, fields_idx, gammas, betas, warm_start_bits=warm_bits)
            sv = Statevector(qc)
            z_s, z_c = _compute_correlations(n_act, sv)
            energy = sum(0.5 * w * z_c[u, v] for u, v, w in edges_idx)
            energy += sum(h_val * z_s[u] for u, h_val in fields_idx.items())
            return float(energy)

        res = minimize(obj_fn, np.array([0.4] * p + [0.4] * p), method="COBYLA", options={"maxiter": maxiter})
        opt_params = res.x

        qc_opt = _build_rqaoa_circuit(
            n_act, edges_idx, fields_idx, list(opt_params[:p]), list(opt_params[p:]), warm_start_bits=warm_bits
        )
        sv_opt = Statevector(qc_opt)
        z_single, z_corr = _compute_correlations(n_act, sv_opt)

        best_edge = None
        max_edge_corr = -1.0
        best_sign = 1
        for u, v, w in edges_idx:
            c_val = z_corr[u, v]
            if abs(c_val) > max_edge_corr:
                max_edge_corr = abs(c_val)
                best_edge = (active_nodes[u], active_nodes[v])
                best_sign = 1 if c_val >= 0 else -1

        best_single = None
        max_single_corr = -1.0
        best_single_sign = 1
        for u in range(n_act):
            s_val = z_single[u]
            if abs(s_val) > max_single_corr:
                max_single_corr = abs(s_val)
                best_single = active_nodes[u]
                best_single_sign = 1 if s_val >= 0 else -1

        if max_edge_corr >= max_single_corr and best_edge is not None:
            u_node, v_node = best_edge
            constraints.append((u_node, v_node, best_sign))
            for neighbor in list(current_g.neighbors(u_node)):
                if neighbor != v_node:
                    w_un = current_g[u_node][neighbor].get("weight", 0.0)
                    new_w = best_sign * float(w_un)
                    if current_g.has_edge(v_node, neighbor):
                        current_g[v_node][neighbor]["weight"] += new_w
                    else:
                        current_g.add_edge(v_node, neighbor, weight=new_w)
            current_h[v_node] = current_h.get(v_node, 0.0) + best_sign * current_h.get(u_node, 0.0)
            current_g.remove_node(u_node)
            current_h.pop(u_node, None)
            current_warm.pop(u_node, None)
        elif best_single is not None:
            u_node = best_single
            s_val = best_single_sign
            constraints.append((u_node, None, s_val))
            for neighbor in list(current_g.neighbors(u_node)):
                w_un = current_g[u_node][neighbor].get("weight", 0.0)
                current_h[neighbor] = current_h.get(neighbor, 0.0) + 0.5 * s_val * float(w_un)
            current_g.remove_node(u_node)
            current_h.pop(u_node, None)
            current_warm.pop(u_node, None)
        else:
            break

    base_sol = _exact_ising_base(current_g, current_h)
    final_sol = dict(base_sol)
    for u_node, v_node, sign in reversed(constraints):
        if v_node is not None:
            v_bit = final_sol.get(v_node, 0)
            v_spin = 1 if v_bit == 0 else -1
            u_spin = sign * v_spin
            final_sol[u_node] = 0 if u_spin >= 0 else 1
        else:
            final_sol[u_node] = 0 if sign >= 0 else 1

    return final_sol


def solve(block: BlockQUBO, meta: SubproblemMeta, config, seed: int) -> BlockResult:
    graph, boundary_fields, warm_start = block_to_ising_graph(block)
    bits_by_idx = _solve_rqaoa_on_graph(
        graph, boundary_fields, warm_start,
        cutoff_nodes=config.rqaoa_cutoff_nodes, p=config.rqaoa_p, maxiter=config.rqaoa_maxiter,
    )
    x_bits = [bits_by_idx.get(i, 0) for i in range(block.n)]

    return BlockResult(
        x_bits=x_bits,
        solver_name=name,
        strategy_name=block.strategy_name,
        meta={"cutoff_nodes": config.rqaoa_cutoff_nodes},
    )
