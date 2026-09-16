"""
DSATUR (Degree of Saturation) Graph Coloring Baseline for Frequency Channel Allocation.
More sophisticated classical heuristic, widely used in mobile network frequency planning.
"""

from typing import Dict, Set
import networkx as nx


def dsatur(graph: nx.Graph, num_channels: int = 2) -> Dict[int, int]:
    """
    Computes a frequency channel assignment using the DSATUR algorithm.
    
    Selects uncolored vertices prioritizing maximal saturation degree
    (count of distinct colors used by adjacent neighbors), with ties broken
    by vertex degree.
    
    Args:
        graph: NetworkX graph representing cell network with edge 'weight'.
        num_channels: Number of available frequency channels (default 2).
        
    Returns:
        Dict mapping node ID -> assigned channel index in {0, ..., num_channels - 1}.
    """
    uncolored: Set[int] = set(graph.nodes())
    assignment: Dict[int, int] = {}
    neighbor_colors: Dict[int, Set[int]] = {n: set() for n in graph.nodes()}
    degrees = dict(graph.degree(weight="weight"))
    
    while uncolored:
        # Saturation degree is the number of distinct colors assigned to neighbors
        # Select vertex with max saturation; tie-break with weighted degree
        best_node = max(
            uncolored,
            key=lambda n: (len(neighbor_colors[n]), degrees[n])
        )
        
        # Calculate interference for each candidate channel
        channel_costs = [0.0] * num_channels
        for neighbor in graph.neighbors(best_node):
            if neighbor in assignment:
                c_neigh = assignment[neighbor]
                w = graph[best_node][neighbor].get("weight", 1.0)
                channel_costs[c_neigh] += float(w)
                
        # Pick channel with minimal conflict
        best_channel = min(range(num_channels), key=lambda c: channel_costs[c])
        assignment[best_node] = best_channel
        uncolored.remove(best_node)
        
        # Update neighbor color sets
        for neighbor in graph.neighbors(best_node):
            if neighbor in uncolored:
                neighbor_colors[neighbor].add(best_channel)
                
    return assignment
