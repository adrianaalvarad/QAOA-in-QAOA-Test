"""
Exploration script for interactive inspection of QAOA-in-QAOA refinement.
Can be converted to an IPython Notebook (.ipynb) or run directly.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data.generate_graph import load_graph_scenario
from baselines.greedy_coloring import greedy_coloring
from baselines.dsatur import dsatur
from evaluation.cost_function import compute_interference_cost, compute_node_residuals
from quantum.qaoa_square import refine


def run_demo_instance(filepath: str, name: str):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}. Run src/data/generate_graph.py first.")
        return
        
    g = load_graph_scenario(filepath)
    print(f"\n{'='*70}\n[DEMO] {name}: {g.number_of_nodes()} celdas, {g.number_of_edges()} enlaces.")
    
    # 1. Run Greedy baseline
    sol_greedy = greedy_coloring(g)
    cost_greedy = compute_interference_cost(g, sol_greedy)
    print(f"Greedy Baseline Cost: {cost_greedy:.2f} puntos de interferencia.")
    
    # 2. Apply Adaptive Multi-Subgraph Quantum Refinement
    print("Aplicando capa cuántica adaptativa (Múltiples subgrafos + R-QAOA / QAOA^2)...")
    refined_sol, info = refine(
        g,
        sol_greedy,
        use_multi_subgraphs=True,
        threshold_percentile=70.0,
        max_subgraphs=3,
        max_sub_size=8,
        density_threshold=0.15,
    )
    
    print("\n--- Resumen del Refinamiento Cuántico ---")
    print(f"Estado Global: {info['status']}")
    print(f"Costo Clásico Inicial: {info['initial_cost']:.2f}")
    print(f"Costo Final Híbrido: {info['final_cost']:.2f}")
    print(f"Delta Obtenido: -{info['delta_abs']:.2f} pts (-{info['delta_pct']:.2f}%)")
    print(f"Subgrafos Residuales Refinados: {info['num_subgraphs']}")
    
    for sub in info.get("subgraphs", []):
        print(
            f"  * Subgrafo {sub['subgraph_index']}: {sub['num_nodes']} nodos, {sub['num_edges']} aristas "
            f"(densidad={sub['density']:.3f}) -> Router: [{sub['solver']}] -> {sub['status']} "
            f"(Delta local: -{sub['delta_local']:.2f} pts)"
        )


def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    
    # Demo 1: Sparse 100 instance (routes sparse components to R-QAOA)
    run_demo_instance(os.path.join(data_dir, "sparse_100_hora_pico.json"), "Instancia Dispersa (sparse_100)")
    
    # Demo 2: Dense 100 instance (routes dense components to QAOA^2)
    run_demo_instance(os.path.join(data_dir, "dense_100_hora_pico.json"), "Instancia Densa (dense_100)")


if __name__ == "__main__":
    main()

