"""Pone QAOA-in-QAOA-Test/src en sys.path para poder reusar (importar, no
duplicar) los baselines, el cargador de grafos y la funcion de costo que ya
existen ahi -- son agnosticos a k y no hace falta portarlos. Se importa por
efecto secundario: `from toolkit import _bootstrap` antes de importar algo
de `data`, `baselines` o `evaluation` (los paquetes de src/, no los de
toolkit/). Sigue el mismo patron que ya usa src/run_experiment.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))
