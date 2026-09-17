"""Interfaz comun a los 4 solvers (exact, direct_qaoa, r_qaoa, qaoa_square).

Los cuatro reciben el MISMO QUBO por-nodo (BlockQUBO) -- nunca una
representacion distinta cada uno. pipeline.py es el unico lugar que
decodifica bits -> coloreo y evalua costo exacto sobre el grafo completo;
ningun solver toca la funcion de costo del grafo directamente, solo el QUBO
local que se le entrega.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Protocol


@dataclass(frozen=True)
class BlockQUBO:
    block_nodes: list  # el orden define el indice local 0..n-1
    linear: list
    quad: dict
    alt_colors: dict  # candidato c_i' por nodo, si x_i=1
    strategy_name: str  # "individual" | "swap"

    @property
    def n(self) -> int:
        return len(self.block_nodes)


@dataclass
class SubproblemMeta:
    n: int
    n_edges_internal: int
    density: float
    hotspot_index: int
    global_density: float
    route: str = ""


@dataclass
class BlockResult:
    x_bits: list  # decision x_i por indice local, en el mismo orden que block_nodes
    solver_name: str
    strategy_name: str
    meta: dict = field(default_factory=dict)


class SubproblemSolver(Protocol):
    name: str

    def solve(self, block: BlockQUBO, meta: SubproblemMeta, config, seed: int) -> BlockResult: ...
