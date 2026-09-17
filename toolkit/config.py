"""Configuracion central del kit. Cada umbral que decide que herramienta se
usa en que particion vive aca, en un solo lugar, con la razon documentada
al lado -- ver toolkit/docs/ARCHITECTURE.md para como se usan.
"""

from __future__ import annotations

from dataclasses import dataclass


# Tamanos de cluster validos del reuso hexagonal clasico de frecuencias:
# N = i^2 + i*j + j^2, i,j enteros >= 0  ->  N en {1, 3, 4, 7, 9, 12, 13, 16, 19, 21, ...}
# Referencia de ingenieria RF: T. Rappaport, "Wireless Communications: Principles
# and Practice", cap. "The Cellular Concept" -- estos son los valores que la
# literatura de planificacion celular 2G/3G cita como patrones estandar.
#
# SALVEDAD IMPORTANTE: redes LTE/5G modernas operan mayormente en reuse-1 con
# coordinacion de interferencia (ICIC/eICIC) o fractional frequency reuse (FFR),
# no en un reuse-N estricto. El modelo de "coloreo con k colores" es una
# abstraccion de planificacion clasica (2G/3G), util para demostrar el mecanismo
# de refinamiento cuantico -- no una descripcion literal de como opera una red
# 5G real hoy. No sobre-vender la analogia en material de cliente.
N_COLORS_REFERENCE: dict[int, str] = {
    3: "Reuse-3: patron hexagonal ajustado, alta capacidad, exige C/I alto.",
    4: "Reuse-4: patron citado en despliegues AMPS/GSM tempranos.",
    7: "Reuse-7: el ejemplo de manual del reuso hexagonal clasico.",
    9: "Reuse-9: GSM urbano denso con mayor tolerancia a interferencia.",
    12: "Reuse-12: baja reutilizacion, redes 2G de gran escala.",
}

DEFAULT_N_COLORS = 6  # heredado de poc-qaoa-agnostic-refinement: ni trivial ni degenerado


@dataclass
class ToolkitConfig:
    n_colors: int = DEFAULT_N_COLORS
    seed: int = 7

    # --- Extraccion multi-hotspot ---
    threshold_percentile: float = 70.0
    min_hotspot_size: int = 4
    max_hotspot_size: int = 24
    max_subgraphs: int = 4

    # --- Umbrales de ruteo (density_threshold justificado empiricamente en
    # hardware.py: bloques con densidad >= este umbral mostraron ~1.4-1.9x mas
    # compuertas de 2 qubits tras transpilar a heavy-hex que bloques dispersos
    # del mismo tamano -- ver toolkit/docs/ARCHITECTURE.md) ---
    density_threshold: float = 0.15
    exact_threshold: int = 6          # 2^6 = 64 combinaciones, optimo garantizado, gratis
    max_qubits_sparse: int = 14
    max_qubits_dense: int = 8
    max_qubits_rqaoa_direct: int = 18
    qaoa2_cluster_size: int = 8

    # --- Presupuesto de optimizacion por solver ---
    direct_qaoa_p: int = 2
    direct_qaoa_shots: int = 512
    direct_qaoa_maxiter: int = 60
    direct_qaoa_epsilon: float = 0.25
    direct_qaoa_top_k: int = 10

    rqaoa_p: int = 1
    rqaoa_maxiter: int = 15
    rqaoa_cutoff_nodes: int = 3

    qaoa2_p: int = 1
    qaoa2_maxiter: int = 20

    def max_qubits_for(self, density: float) -> int:
        return self.max_qubits_sparse if density < self.density_threshold else self.max_qubits_dense
