"""Entrypoint del kit. Corre baseline x escenario x k sobre los grafos ya
existentes en QAOA-in-QAOA-Test/data/, e imprime -- por cada hotspot real
que encontro esa corrida -- que ruta tomo el router y por que, mas el
resultado final. Uso:

    python -m toolkit.cli --graph sparse_100 --scenario hora_pico --baseline dsatur --k 6
"""

from __future__ import annotations

import argparse
import os

from toolkit import _bootstrap  # noqa: F401  (side effect: src/ en sys.path)
from baselines.dsatur import dsatur  # de QAOA-in-QAOA-Test/src
from baselines.greedy_coloring import greedy_coloring  # de QAOA-in-QAOA-Test/src
from data.generate_graph import load_graph_scenario  # de QAOA-in-QAOA-Test/src
from evaluation.cost_function import compute_interference_cost  # de QAOA-in-QAOA-Test/src

from toolkit.config import N_COLORS_REFERENCE, ToolkitConfig
from toolkit.pipeline import refine

BASELINES = {"greedy": greedy_coloring, "dsatur": dsatur}

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_REPO_ROOT, "data")


def print_report(report: dict, k: int) -> None:
    ref_note = N_COLORS_REFERENCE.get(k, "sin referencia de reuso clasico documentada para este k")
    print(f"k={k} colores -- {ref_note}")
    print(f"costo inicial: {report['initial_cost']:.2f}  |  densidad global del grafo: {report['global_density']:.3f}")
    print(f"hotspots detectados: {report['n_hotspots']}")

    for h in report["hotspots"]:
        estado = "ADOPTADO" if h["adopted"] else "descartado (no mejoro)"
        print(
            f"\n  hotspot {h['index']}: n={h['n']} densidad={h['density']} "
            f"-> ruta={h['route']} [{estado}, delta={h['delta']:.2f}]"
        )
        print(f"    razon de ruta: {h['reason']}")
        for b in h["blocks"]:
            tried_str = ", ".join(f"{t['strategy']}={t['cost']:.2f}" for t in b["tried"])
            print(f"    bloque n={b['n']} solver={b['solver']} ({b['reason']}) -> {tried_str}")

    print(
        f"\ncosto final: {report['final_cost']:.2f}  "
        f"delta={report['delta_abs']:.2f} ({report['delta_pct']:+.2f}%)  "
        f"nunca_empeora={report['never_worsens']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="sparse_100", choices=["sparse_100", "dense_100", "realistic_metropolitan"])
    parser.add_argument("--scenario", default="hora_pico", choices=["hora_pico", "hora_valle"])
    parser.add_argument("--baseline", default="dsatur", choices=list(BASELINES))
    parser.add_argument("--k", type=int, default=6, help="numero de canales/colores disponibles")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    json_path = os.path.join(_DATA_DIR, f"{args.graph}_{args.scenario}.json")
    graph = load_graph_scenario(json_path)

    baseline_fn = BASELINES[args.baseline]
    candidate = baseline_fn(graph, num_channels=args.k)
    baseline_cost = compute_interference_cost(graph, candidate)
    print(f"grafo: {args.graph}/{args.scenario}  |V|={graph.number_of_nodes()} |E|={graph.number_of_edges()}")
    print(f"baseline: {args.baseline}  costo={baseline_cost:.2f}\n")

    config = ToolkitConfig(n_colors=args.k, seed=args.seed)
    _, report = refine(graph, candidate, args.k, config)
    print_report(report, args.k)


if __name__ == "__main__":
    main()
