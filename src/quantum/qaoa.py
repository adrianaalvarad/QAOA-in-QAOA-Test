"""
Qiskit Implementation of Standard and Warm-Start QAOA for MaxCut/Interference Minimization.
Uses native Qiskit parameterized QuantumCircuit gates (RZZ, RZ, RX, RY) and exact Statevector evaluation.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize


def build_qaoa_circuit(
    num_qubits: int,
    edges: List[Tuple[int, int, float]],
    linear_fields: Dict[int, float],
    gamma: List[float],
    beta: List[float],
    warm_start_bits: Optional[List[int]] = None,
    warm_start_bias: float = 0.15,
) -> QuantumCircuit:
    """
    Builds a QAOA circuit for interference minimization with optional warm-start.
    
    Args:
        num_qubits: Number of qubits (subgraph vertices).
        edges: List of tuples (u, v, weight) representing mutual interference coupling.
        linear_fields: Dict mapping qubit index u -> external linear bias h_u.
        gamma: QAOA cost layer rotation angles (length p).
        beta: QAOA mixer layer rotation angles (length p).
        warm_start_bits: Optional binary list in {0, 1}^n from classical baseline solution.
        warm_start_bias: Parameter epsilon controlling bias towards classical solution.
        
    Returns:
        Parameterized Qiskit QuantumCircuit.
    """
    p = len(gamma)
    qc = QuantumCircuit(num_qubits)
    
    # State preparation: Warm-start biased initialization or uniform |+> superposition
    if warm_start_bits is not None:
        for i in range(num_qubits):
            b_i = warm_start_bits[i]
            # Ry rotation: if b_i == 0 bias to |0>, if b_i == 1 bias to |1>
            theta_i = (
                2.0 * np.arcsin(np.sqrt(warm_start_bias))
                if b_i == 0
                else np.pi - 2.0 * np.arcsin(np.sqrt(warm_start_bias))
            )
            qc.ry(theta_i, i)
    else:
        # Standard uniform superposition |+>
        for i in range(num_qubits):
            qc.h(i)
            
    # Alternating Cost and Mixer Unitaries
    for l in range(p):
        g = gamma[l]
        b = beta[l]
        
        # Cost unitary e^{-i g H_C}: RZZ for quadratic terms, RZ for boundary linear terms
        for u, v, w in edges:
            qc.rzz(2.0 * g * float(w), u, v)
        for u, h in linear_fields.items():
            qc.rz(2.0 * g * float(h), u)
            
        # Mixer unitary e^{-i b H_M}: RX rotations
        for i in range(num_qubits):
            qc.rx(2.0 * b, i)
            
    return qc


def compute_bitstring_cost(
    bitstring: List[int],
    edges: List[Tuple[int, int, float]],
    linear_fields: Dict[int, float]
) -> float:
    """Computes exact interference cost for a classical bitstring configuration."""
    cost = 0.0
    for u, v, w in edges:
        if bitstring[u] == bitstring[v]:
            cost += float(w)
    for u, h in linear_fields.items():
        # Spin mapping: bit 0 -> +1, bit 1 -> -1
        z_u = 1.0 if bitstring[u] == 0 else -1.0
        cost += float(h) * z_u
    return cost


def run_qaoa_subgraph(
    num_qubits: int,
    edges: List[Tuple[int, int, float]],
    linear_fields: Optional[Dict[int, float]] = None,
    warm_start_bits: Optional[List[int]] = None,
    p: int = 1,
    maxiter: int = 25,
) -> Tuple[float, List[int]]:
    """
    Optimizes and solves QAOA on a local subgraph using exact Qiskit simulation.
    
    Args:
        num_qubits: Number of qubits (<= 12 for efficient simulation).
        edges: Subgraph edge list (u, v, weight).
        linear_fields: Dict mapping node index -> linear bias from external boundary.
        warm_start_bits: Classical solution bitstring for warm start.
        p: Number of QAOA layers.
        maxiter: Maximum optimization iterations.
        
    Returns:
        Tuple of (best_objective_value, best_bitstring).
    """
    if linear_fields is None:
        linear_fields = {}
        
    if num_qubits == 0:
        return 0.0, []
    if num_qubits == 1:
        # Trivial single-qubit problem
        c0 = compute_bitstring_cost([0], edges, linear_fields)
        c1 = compute_bitstring_cost([1], edges, linear_fields)
        return (c0, [0]) if c0 <= c1 else (c1, [1])

    # Objective function: expectation value of cost Hamiltonian over statevector
    def objective(params: np.ndarray) -> float:
        gammas = list(params[:p])
        betas = list(params[p:])
        qc = build_qaoa_circuit(
            num_qubits, edges, linear_fields, gammas, betas, warm_start_bits=warm_start_bits
        )
        sv = Statevector(qc)
        probs = sv.probabilities()
        
        expval = 0.0
        for idx, prob in enumerate(probs):
            if prob > 1e-5:
                # Bitstring extraction (big-endian bit decoding)
                bits = [(idx >> (num_qubits - 1 - i)) & 1 for i in range(num_qubits)]
                expval += prob * compute_bitstring_cost(bits, edges, linear_fields)
        return float(expval)

    # Initial parameter guess
    init_params = np.array([0.4] * p + [0.4] * p)
    res = minimize(objective, init_params, method="COBYLA", options={"maxiter": maxiter})
    
    # Extract optimal state and find lowest-cost bitstring sampled
    opt_params = res.x
    qc_opt = build_qaoa_circuit(
        num_qubits, edges, linear_fields, list(opt_params[:p]), list(opt_params[p:]),
        warm_start_bits=warm_start_bits
    )
    sv_opt = Statevector(qc_opt)
    probs_opt = sv_opt.probabilities()
    
    # Candidate bitstrings evaluated directly
    best_cost = float("inf")
    best_bits = warm_start_bits if warm_start_bits is not None else [0] * num_qubits
    
    # Include warm_start_bits in evaluation
    if warm_start_bits is not None:
        c_warm = compute_bitstring_cost(warm_start_bits, edges, linear_fields)
        best_cost = c_warm
        best_bits = warm_start_bits[:]
        
    for idx, prob in enumerate(probs_opt):
        if prob > 1e-4:
            bits = [(idx >> (num_qubits - 1 - i)) & 1 for i in range(num_qubits)]
            c = compute_bitstring_cost(bits, edges, linear_fields)
            if c < best_cost:
                best_cost = c
                best_bits = bits
                
    return best_cost, best_bits
