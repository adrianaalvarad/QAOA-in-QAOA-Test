"""Estima que tan lejos esta un bloque real de correr en hardware cuantico
actual: transpila el circuito de direct_qaoa contra un mapa de conectividad
heavy-hex (la topologia de los chips superconductores IBM actuales) y
compara compuertas logicas vs. tras mapear, mas una estimacion gruesa de
fidelidad. Port parametrizado por n_colors de
poc-qaoa-agnostic-refinement/src/hardware_feasibility.py -- confirma si el
hallazgo original ("la densidad de acoplamientos domina sobre el numero de
qubits") se sostiene para otros valores de k, no solo k=6.
"""

from __future__ import annotations

from qiskit import transpile
from qiskit.transpiler import CouplingMap

from toolkit.qubo import build_alt_color_strategies, build_qubo, qubo_to_ising
from toolkit.solvers._qaoa_core import build_warm_start_circuit

COUPLING = CouplingMap.from_heavy_hex(5, bidirectional=True)
BASIS_GATES = ["ecr", "rz", "sx", "x"]
PER_2Q_ERROR = 0.005
PER_1Q_ERROR = 0.0003


def analyze_block(graph, coloring, block_nodes, n_colors, label, p=2):
    strategies = build_alt_color_strategies(graph, block_nodes, coloring, n_colors)
    rows = []
    for strategy_name, alt_colors in strategies.items():
        linear, quad = build_qubo(graph, block_nodes, coloring, alt_colors)
        n = len(block_nodes)
        h, j_coeffs = qubo_to_ising(linear, quad, n)
        qc, gammas, betas = build_warm_start_circuit(n, h, j_coeffs, p, epsilon=0.25)
        bound = qc.assign_parameters({param: 0.3 for param in qc.parameters})

        logical_2q = sum(1 for instr in bound.data if instr.operation.num_qubits == 2)
        transpiled = transpile(
            bound, coupling_map=COUPLING, basis_gates=BASIS_GATES,
            optimization_level=3, seed_transpiler=7,
        )
        counts = transpiled.count_ops()
        n_2q = counts.get("ecr", 0)
        n_1q = sum(v for k, v in counts.items() if k != "ecr")
        fidelity = ((1 - PER_2Q_ERROR) ** n_2q) * ((1 - PER_1Q_ERROR) ** n_1q)

        rows.append({
            "label": f"{label} / {strategy_name} (n={n}, k={n_colors})",
            "n_colors": n_colors, "n_qubits": n,
            "logical_2q": logical_2q, "transpiled_2q": n_2q,
            "depth": transpiled.depth(), "fidelity_pct": fidelity * 100,
        })
    return rows


def print_rows(rows):
    for r in rows:
        print(
            f"{r['label']}: 2q logico={r['logical_2q']} -> transpilado={r['transpiled_2q']}, "
            f"profundidad={r['depth']}, fidelidad~{r['fidelity_pct']:.1f}%"
        )
