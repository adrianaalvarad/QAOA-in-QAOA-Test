"""Adapta un BlockQUBO (linear, quad sobre x en {0,1}) al grafo de Ising con
pesos que espera el codigo de R-QAOA/QAOA2 portado de QAOA-in-QAOA-Test.

Detalle critico (verificado algebraicamente, no asumido): el circuito de
R-QAOA (`qc.rzz(gamma*w, u, v)`) y su calculo de energia por correlaciones
(`0.5*w*<Z_u Z_v>`) asumen que el peso de arista `w` es el DOBLE del
acoplamiento de Ising `J_ij`, no J_ij directo.

Por que: qiskit define RZZ(theta) = exp(-i*theta/2 * Z@Z). Para reproducir
exp(-i*gamma*J*Z@Z) (el unitario de costo estandar de un termino J*Z_i*Z_j),
se necesita theta = 2*gamma*J. El codigo llama `rzz(gamma*w, u, v)`, o sea
theta=gamma*w -- para que theta == 2*gamma*J, hace falta w = 2*J.
Consistente con la formula de energia: `0.5*w*z_corr = 0.5*(2J)*z_corr =
J*z_corr`, que es la energia de Ising correcta. Sin este factor, la
eliminacion por correlacion de R-QAOA decidiria mal que variable eliminar
en cada ronda.

El campo lineal h SI se pasa sin reescalar -- el circuito ya usa
`rz(2*gamma*h, u)`, que es exactamente exp(-i*gamma*h*Z_u).
"""

from __future__ import annotations

import networkx as nx

from toolkit.qubo import qubo_to_ising


def block_to_ising_graph(block) -> tuple[nx.Graph, dict, dict]:
    """Devuelve (grafo con peso=2*J_ij por arista, {idx: h_i}, {idx: 0} warm-start).

    Los indices de nodo del grafo devuelto son los indices LOCALES 0..n-1 de
    block.block_nodes, no los nodos originales del grafo grande -- igual que
    hace toolkit/qubo.py internamente.
    """
    n = block.n
    h, j_coeffs = qubo_to_ising(block.linear, block.quad, n)

    g = nx.Graph()
    g.add_nodes_from(range(n))
    for (i, j), j_ij in j_coeffs.items():
        if abs(j_ij) > 1e-12:
            g.add_edge(i, j, weight=2.0 * j_ij)

    boundary_fields = {i: h[i] for i in range(n)}
    warm_start = {i: 0 for i in range(n)}  # x_i=0 ("mantener") es el sesgo correcto siempre
    return g, boundary_fields, warm_start
