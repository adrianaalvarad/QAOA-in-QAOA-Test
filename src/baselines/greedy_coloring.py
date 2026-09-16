"""
Greedy Graph Coloring Baseline for Frequency Channel Allocation.
Industry-standard reference heuristic (Welsh-Powell / Largest-Degree-First).
"""

from typing import Dict
import networkx as nx


def greedy_coloring(graph: nx.Graph, num_channels: int = 2) -> Dict[int, int]:
    """
    Computes a frequency channel assignment using greedy largest-degree-first heuristic.
    
    Args:
        graph: NetworkX graph representing cell network with edge 'weight'.
        num_channels: Number of available frequency channels (default 2).
        
    Returns:
        Dict mapping node ID -> assigned channel index in {0, ..., num_channels - 1}.
    """
    # Order nodes by weighted degree descending
    degrees = dict(graph.degree(weight="weight"))
    sorted_nodes = sorted(graph.nodes(), key=lambda n: degrees[n], reverse=True)
    
    assignment: Dict[int, int] = {}
    
    for node in sorted_nodes:
        # Calculate interference for each candidate channel
        channel_costs = [0.0] * num_channels
        for neighbor in graph.neighbors(node):
            if neighbor in assignment:
                neigh_color = assignment[neighbor]
                w = graph[node][neighbor].get("weight", 1.0)
                channel_costs[neigh_color] += float(w)
                
        # Pick channel with minimal conflict
        best_channel = min(range(num_channels), key=lambda c: channel_costs[c])
        assignment[node] = best_channel
        
    return assignment
