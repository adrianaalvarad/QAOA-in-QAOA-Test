"""
Synthetic Network Graph Generator for Urbania Telecom.
Grounded on OSIPTEL reference documentation for mobile network cell topologies in Lima, Peru.

Generates 3 network instances with dual temporal scenarios ("hora_pico" and "hora_valle"):
  1. sparse_100: 100 cells, ~450 edges (Zhou et al. 2023 benchmark scale, suburban/macrocells)
  2. dense_100: 100 cells, 3500-4000 edges (dense urban microcell core, high interference)
  3. realistic_metropolitan: 600 cells, ~4000 edges (realistic Lima metropolitan multi-district deployment)
"""

import json
import math
import os
from typing import Dict, List, Tuple
import networkx as nx
import numpy as np


# District centers approximating Lima's geography (in km relative to city center)
DISTRICT_CENTERS = {
    "San_Isidro": (0.0, 0.0),       # High-density financial center (microcells)
    "Miraflores": (1.5, -2.0),      # Dense commercial/residential tourist area
    "Cercado_Lima": (-1.0, 4.0),    # Historic center, mixed traffic
    "Santiago_Surco": (4.0, -1.5),  # Extensive suburban/residential area
    "San_Borja": (2.5, 1.0),        # Residential and commercial hub
}


def create_cell_positions(
    num_cells: int,
    dense_mode: bool = False,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Distributes cell sites according to OSIPTEL demographic and traffic density logic.
    
    Returns:
        positions: (N, 2) array of coordinates in km.
        radii: (N,) array of coverage radii in km.
        cell_types: List of 'microcell' or 'macrocell'.
    """
    rng = np.random.default_rng(seed)
    districts = list(DISTRICT_CENTERS.keys())
    
    positions = []
    radii = []
    cell_types = []
    
    for i in range(num_cells):
        # District assignment
        d_name = districts[i % len(districts)]
        cx, cy = DISTRICT_CENTERS[d_name]
        
        # High traffic districts have higher microcell proportion
        is_micro = dense_mode or (d_name in ["San_Isidro", "Miraflores"] and rng.random() < 0.7) or (rng.random() < 0.35)
        
        if is_micro:
            # Microcell: compact radius (0.8 - 1.5 km), tightly clustered
            std = 0.8 if not dense_mode else 0.45
            x = cx + rng.normal(0, std)
            y = cy + rng.normal(0, std)
            r = rng.uniform(0.8, 1.5)
            c_type = "microcell"
        else:
            # Macrocell: wide radius (2.5 - 4.5 km), widely distributed
            std = 2.0
            x = cx + rng.normal(0, std)
            y = cy + rng.normal(0, std)
            r = rng.uniform(2.5, 4.5)
            c_type = "macrocell"
            
        positions.append((x, y))
        radii.append(r)
        cell_types.append(c_type)
        
    return np.array(positions), np.array(radii), cell_types


def build_network_topology(
    num_cells: int,
    target_edges: int,
    dense_mode: bool = False,
    seed: int = 42
) -> nx.Graph:
    """
    Builds the structural graph topology connecting cells that have potential mutual interference.
    Edges are formed where coverage circles overlap or proximity causes channel interference.
    """
    rng = np.random.default_rng(seed)
    positions, radii, cell_types = create_cell_positions(num_cells, dense_mode=dense_mode, seed=seed)
    
    # Calculate all pairwise distances and RF overlap
    candidate_edges = []
    for u in range(num_cells):
        for v in range(u + 1, num_cells):
            dist = math.hypot(positions[u, 0] - positions[v, 0], positions[u, 1] - positions[v, 1])
            combined_range = radii[u] + radii[v]
            
            # Interference potential: inversely related to distance, directly to overlap
            if dist < 1e-4:
                dist = 1e-4
            overlap_score = max(0.0, (combined_range - dist) / combined_range)
            candidate_edges.append((overlap_score, dist, u, v))
            
    # Sort candidate edges by highest potential interference (overlap score descending, distance ascending)
    candidate_edges.sort(key=lambda x: (-x[0], x[1]))
    
    # Select edges to match target_edges while ensuring connected network
    G = nx.Graph()
    for i in range(num_cells):
        G.add_node(i, pos=positions[i].tolist(), radius=float(radii[i]), type=cell_types[i])
        
    # First, build a minimum spanning tree on Euclidean distances to guarantee connectivity
    complete_dist_g = nx.Graph()
    for _, dist, u, v in candidate_edges:
        complete_dist_g.add_edge(u, v, dist=dist)
    mst = nx.minimum_spanning_tree(complete_dist_g, weight="dist")
    for u, v in mst.edges():
        G.add_edge(u, v)
        
    # Add remaining edges based on highest interference potential
    for _, _, u, v in candidate_edges:
        if G.number_of_edges() >= target_edges:
            break
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            
    return G


def compute_scenario_weights(
    G: nx.Graph,
    scenario: str,
    seed: int = 42
) -> nx.Graph:
    """
    Applies realistic edge weights ('puntos de interferencia') for a temporal scenario:
      - 'hora_pico': Peak congestion, heavy spectral load, higher interference penalty (+60-80%).
      - 'hora_valle': Low traffic hours, lower spectral interference penalty (-30-40%).
      
    Both scenarios share EXACTLY the same nodes and edges.
    """
    rng = np.random.default_rng(seed + (100 if scenario == "hora_pico" else 200))
    G_scenario = G.copy()
    
    positions = nx.get_node_attributes(G_scenario, "pos")
    radii = nx.get_node_attributes(G_scenario, "radius")
    
    for u, v in G_scenario.edges():
        pos_u = np.array(positions[u])
        pos_v = np.array(positions[v])
        dist = float(np.linalg.norm(pos_u - pos_v))
        dist = max(dist, 0.1)
        r_sum = float(radii[u] + radii[v])
        
        # Base geometric RF interference score (1 to 10 scale)
        geom_factor = max(0.1, (r_sum - dist) / r_sum) if dist < r_sum else (r_sum / (2.0 * dist))
        base_cost = 10.0 * geom_factor
        
        if scenario == "hora_pico":
            # Peak hours: higher active user density, frequency reuse stress, traffic noise
            traffic_mult = rng.uniform(1.4, 1.8)
            noise = rng.normal(0.0, 0.5)
            weight = max(1.0, base_cost * traffic_mult + noise)
        else:  # hora_valle
            # Valley hours: fewer active channels in air, reduced collision probability
            traffic_mult = rng.uniform(0.6, 0.8)
            noise = rng.normal(0.0, 0.2)
            weight = max(0.5, base_cost * traffic_mult + noise)
            
        G_scenario[u][v]["weight"] = round(float(weight), 3)
        G_scenario[u][v]["distance_km"] = round(dist, 3)
        
    return G_scenario


def generate_synthetic_network(config_name: str, seed: int = 42) -> Tuple[nx.Graph, nx.Graph]:
    """
    Generates the pair of temporal scenarios (hora_pico, hora_valle) for a named configuration:
      - 'sparse_100': 100 cells, ~450 edges (Zhou et al. benchmark scale)
      - 'dense_100': 100 cells, ~3800 edges (dense urban microcells, 3500-4000 edges)
      - 'realistic_metropolitan': 600 cells, ~4000 edges (Lima metropolitan multi-district)
    """
    if config_name == "sparse_100":
        num_cells = 100
        target_edges = 450
        dense_mode = False
    elif config_name == "dense_100":
        num_cells = 100
        target_edges = 3800  # user requested 3500-4000 edges for dense graph
        dense_mode = True
    elif config_name == "realistic_metropolitan":
        num_cells = 600     # user requested 500-1000 vertices
        target_edges = 4200
        dense_mode = False
    else:
        raise ValueError(f"Unknown configuration name: {config_name}")
        
    topology = build_network_topology(num_cells, target_edges, dense_mode=dense_mode, seed=seed)
    
    g_pico = compute_scenario_weights(topology, "hora_pico", seed=seed)
    g_valle = compute_scenario_weights(topology, "hora_valle", seed=seed)
    
    return g_pico, g_valle


def save_graph_scenario(graph: nx.Graph, filepath: str) -> None:
    """Serializes a network graph to JSON format compatible with QAOA_in_QAQA data format."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    edges = []
    for u, v, data in graph.edges(data=True):
        edges.append([int(u), int(v), float(data.get("weight", 1.0))])
        
    data = {
        "n_v": graph.number_of_nodes(),
        "n_e": graph.number_of_edges(),
        "edges": edges,
        "nodes": {
            str(n): {
                "pos": data.get("pos", [0.0, 0.0]),
                "radius": data.get("radius", 1.0),
                "type": data.get("type", "macrocell")
            }
            for n, data in graph.nodes(data=True)
        }
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_graph_scenario(filepath: str) -> nx.Graph:
    """Loads a network graph from JSON format."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    G = nx.Graph()
    for n_str, n_data in data.get("nodes", {}).items():
        G.add_node(int(n_str), **n_data)
        
    for edge in data["edges"]:
        u, v, w = edge
        G.add_edge(int(u), int(v), weight=float(w))
        
    return G


if __name__ == "__main__":
    # Generate and save all 3 configurations for both scenarios
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    for name in ["sparse_100", "dense_100", "realistic_metropolitan"]:
        print(f"Generating {name}...")
        g_pico, g_valle = generate_synthetic_network(name, seed=42)
        save_graph_scenario(g_pico, os.path.join(out_dir, f"{name}_hora_pico.json"))
        save_graph_scenario(g_valle, os.path.join(out_dir, f"{name}_hora_valle.json"))
        print(f"  {name}: Nodes={g_pico.number_of_nodes()}, Edges={g_pico.number_of_edges()}")
    print("Graph generation completed.")
