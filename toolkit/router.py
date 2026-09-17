"""El router: decide que herramienta usar para cada hotspot y para cada
bloque, de forma transparente -- cada decision queda registrada (no solo
tomada) para que el reporte final pueda mostrar, por region real, que ruta
se siguio y por que. Ver toolkit/docs/ARCHITECTURE.md para el diagrama de
flujo completo; cada funcion de aca corresponde 1:1 a un rombo del diagrama.
"""

from __future__ import annotations

from dataclasses import dataclass

from toolkit.solvers import direct_qaoa, exact, qaoa_square, r_qaoa

HOTSPOT_ROUTE_BLOCK_LEVEL = "block_level"
HOTSPOT_ROUTE_RQAOA_DIRECT = "r_qaoa_direct_on_hotspot"
HOTSPOT_ROUTE_PARTITION_RECURSE = "partition_then_block_router"
HOTSPOT_ROUTE_QAOA_SQUARE = "qaoa_square_hierarchical"

BLOCK_SOLVER_EXACT = "exact"
BLOCK_SOLVER_RQAOA = "r_qaoa"
BLOCK_SOLVER_DIRECT_QAOA = "direct_qaoa"

_SOLVER_MODULES = {
    BLOCK_SOLVER_EXACT: exact,
    BLOCK_SOLVER_RQAOA: r_qaoa,
    BLOCK_SOLVER_DIRECT_QAOA: direct_qaoa,
    "qaoa_square": qaoa_square,  # solo se invoca directo a nivel-hotspot, no via decide_block_solver
}


@dataclass
class RouteDecision:
    route: str
    reason: str
    n: int
    density: float


def decide_hotspot_route(n: int, density: float, config) -> RouteDecision:
    """Nivel-hotspot: ¿el hotspot completo cabe directo en un bloque, o
    necesita particion (dispersa -> R-QAOA directo o recursion; densa y
    grande -> QAOA² jerarquico)?
    """
    cap = config.max_qubits_for(density)
    if n <= cap:
        return RouteDecision(
            HOTSPOT_ROUTE_BLOCK_LEVEL,
            f"tamano {n} <= cap para densidad {density:.2f} ({cap})",
            n, density,
        )

    is_sparse = density < config.density_threshold
    if is_sparse and n <= config.max_qubits_rqaoa_direct:
        return RouteDecision(
            HOTSPOT_ROUTE_RQAOA_DIRECT,
            f"disperso (densidad {density:.2f} < {config.density_threshold}) y "
            f"tamano {n} <= cap R-QAOA directo ({config.max_qubits_rqaoa_direct})",
            n, density,
        )
    if is_sparse:
        return RouteDecision(
            HOTSPOT_ROUTE_PARTITION_RECURSE,
            f"disperso pero tamano {n} excede incluso el cap de R-QAOA directo "
            f"({config.max_qubits_rqaoa_direct}) -- se particiona y se re-enruta cada bloque",
            n, density,
        )
    return RouteDecision(
        HOTSPOT_ROUTE_QAOA_SQUARE,
        f"denso (densidad {density:.2f} >= {config.density_threshold}) y tamano {n} > "
        f"cap denso ({config.max_qubits_dense}) -- particion en clusters + super-grafo",
        n, density,
    )


def decide_block_solver(n: int, density: float, config) -> RouteDecision:
    """Nivel-bloque: dado un bloque que ya cabe en un solo circuito, ¿con
    que solver se resuelve?
    """
    if n <= config.exact_threshold:
        return RouteDecision(
            BLOCK_SOLVER_EXACT,
            f"tamano {n} <= exact_threshold ({config.exact_threshold}): fuerza bruta, optimo garantizado",
            n, density,
        )
    if density < config.density_threshold:
        return RouteDecision(
            BLOCK_SOLVER_RQAOA,
            f"disperso (densidad {density:.2f} < {config.density_threshold})",
            n, density,
        )
    return RouteDecision(
        BLOCK_SOLVER_DIRECT_QAOA,
        f"denso (densidad {density:.2f} >= {config.density_threshold})",
        n, density,
    )


def get_solver(solver_name: str):
    return _SOLVER_MODULES[solver_name]
