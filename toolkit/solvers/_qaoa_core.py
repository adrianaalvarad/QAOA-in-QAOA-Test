"""Nucleo compartido del ansatz QAOA con warm-start (Egger et al. 2021),
usado tanto por direct_qaoa.py (sobre el QUBO de un bloque) como por
qaoa_square.py (sobre los clusters y el super-grafo contraido). Centralizar
esto en un solo lugar evita tener dos circuitos con convenciones de escala
distintas -- ver toolkit/solvers/_ising_adapter.py para por que esa clase de
inconsistencia es exactamente el bug que hay que evitar al portar codigo
cuantico de un repo a otro.
"""

from __future__ import annotations

import math
from typing import Callable

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from scipy.optimize import minimize


def ising_energy(bits: list, h: list, j_coeffs: dict) -> float:
    """h*z + J*z_u*z_v, con z_i = 1 - 2*bits[i] (bits[i]=0 -> z=+1)."""
    z = [1.0 - 2.0 * b for b in bits]
    energy = sum(h[i] * z[i] for i in range(len(bits)))
    energy += sum(coeff * z[i] * z[j] for (i, j), coeff in j_coeffs.items())
    return energy


def build_cost_observable(h: list, j_coeffs: dict, n: int):
    sparse_list = [("Z", [i], coeff) for i, coeff in enumerate(h) if abs(coeff) > 1e-12]
    sparse_list += [
        ("ZZ", [i, j], coeff) for (i, j), coeff in j_coeffs.items() if abs(coeff) > 1e-12
    ]
    if not sparse_list:
        return None
    return SparsePauliOp.from_sparse_list(sparse_list, num_qubits=n)


def build_warm_start_circuit(n: int, h: list, j_coeffs: dict, p: int, epsilon: float):
    theta = 2 * math.asin(math.sqrt(epsilon))
    gammas = [Parameter(f"gamma{layer}") for layer in range(p)]
    betas = [Parameter(f"beta{layer}") for layer in range(p)]

    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(theta, i)

    for layer in range(p):
        gamma, beta = gammas[layer], betas[layer]
        for i, coeff in enumerate(h):
            if abs(coeff) > 1e-12:
                qc.rz(2 * gamma * coeff, i)
        for (i, j), coeff in j_coeffs.items():
            if abs(coeff) > 1e-12:
                qc.cx(i, j)
                qc.rz(2 * gamma * coeff, j)
                qc.cx(i, j)
        for i in range(n):
            qc.ry(-theta, i)
            qc.rz(-2 * beta, i)
            qc.ry(theta, i)

    return qc, gammas, betas


def _ordered_param_values(qc: QuantumCircuit, gammas: list, betas: list, x, p: int):
    values_by_param = {}
    k = 0
    for layer in range(p):
        values_by_param[gammas[layer]] = x[k]
        values_by_param[betas[layer]] = x[k + 1]
        k += 2
    return [values_by_param[param] for param in qc.parameters]


def _candidate_bitstrings(counts: dict, n: int, top_k: int) -> list:
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    candidates = [bitstring for bitstring, _ in ranked[:top_k]]
    keep_all = "0" * n
    if keep_all not in candidates:
        candidates.append(keep_all)
    return candidates


def _decode_bitstring(bitstring: str, n: int) -> list:
    reversed_bits = bitstring[::-1]
    return [int(reversed_bits[i]) for i in range(n)]


def run_warm_start_qaoa(
    n: int,
    h: list,
    j_coeffs: dict,
    p: int,
    shots: int,
    maxiter: int,
    epsilon: float,
    top_k: int,
    seed: int,
    evaluate_fn: Callable[[list], float],
) -> tuple[list, float]:
    """Optimiza y muestrea el ansatz warm-start; devuelve (mejores_bits, costo).

    `evaluate_fn` decide que significa "costo" para el llamador: direct_qaoa
    usa el QUBO del bloque (qubo_cost), qaoa_square usa la energia de Ising
    del cluster o del super-grafo (ising_energy) -- el nucleo del circuito es
    el mismo en los dos casos, solo cambia como se puntua cada candidato.
    """
    zero_bits = [0] * n
    best_bits = zero_bits
    best_cost = evaluate_fn(zero_bits)

    observable = build_cost_observable(h, j_coeffs, n)
    if observable is None:
        return best_bits, best_cost

    qc, gammas, betas = build_warm_start_circuit(n, h, j_coeffs, p, epsilon)
    estimator = AerEstimator()

    def expectation(x):
        values = _ordered_param_values(qc, gammas, betas, x, p)
        job = estimator.run([(qc, observable, [values])])
        return float(job.result()[0].data.evs[0])

    x0 = [0.3, 0.3] * p
    result = minimize(expectation, x0, method="COBYLA", options={"maxiter": maxiter})

    bound_values = {
        param: val
        for param, val in zip(qc.parameters, _ordered_param_values(qc, gammas, betas, result.x, p))
    }
    bound_circuit = qc.assign_parameters(bound_values)
    bound_circuit.measure_all()

    simulator = AerSimulator()
    transpiled = transpile(bound_circuit, simulator)
    counts = simulator.run(transpiled, shots=shots, seed_simulator=seed).result().get_counts()

    for bitstring in _candidate_bitstrings(counts, n, top_k):
        bits = _decode_bitstring(bitstring, n)
        cost = evaluate_fn(bits)
        if cost < best_cost:
            best_cost = cost
            best_bits = bits

    return best_bits, best_cost
