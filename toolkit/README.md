# Toolkit — kit modular de refinamiento cuántico (k general)

Carpeta nueva y separada dentro de `QAOA-in-QAOA-Test`: no modifica nada de `src/`, `data/`,
`requirements.txt` ni `results/` ya existentes en el repo.

Combina las ideas de dos proyectos hermanos:
- `poc-qaoa-agnostic-refinement`: QAOA warm-start con dos estrategias de candidato por nodo
  ("individual", "swap"), sobre *k* general (no solo 2 colores).
- `QAOA-in-QAOA-Test` (`src/`, sin tocar): extracción multi-hotspot, R-QAOA (Bravyi et al. 2020) y
  QAOA² jerárquico (Zhou et al. 2023), originalmente restringidos a *k*=2.

Ver `docs/ARCHITECTURE.md` para el diagrama de flujo completo (qué decide el router en cada
punto, y por qué) y `docs/n_colors_reference.md` para los valores de referencia de *k* (reuso
celular clásico, con la salvedad de que LTE/5G moderno no opera así literalmente).

## Cómo correr

```powershell
# desde la raíz de QAOA-in-QAOA-Test, con un venv que tenga requirements.txt (raíz) +
# toolkit/requirements.txt (agrega qiskit-aer) instalados:
python -m toolkit.cli --graph sparse_100 --scenario hora_pico --baseline dsatur --k 6
```

Imprime, por cada hotspot real detectado en esa corrida: tamaño, densidad, qué ruta tomó el
router y por qué, qué solver resolvió cada bloque, el resultado de las dos estrategias de
candidato, y si se adoptó o no — más el delta final y la garantía de "nunca empeora".

## Tests

```powershell
python -m toolkit.tests.test_toolkit
```

## Estado de validación (honesto, no todo al mismo nivel)

| Pieza | Origen | Validación |
|---|---|---|
| `qubo.py` (QUBO/Ising, estrategias individual/swap) | Port de `poc-qaoa-agnostic-refinement` | Cruzado contra costo manual, para *k* general (`tests/test_toolkit.py`). |
| `solvers/exact.py` | Nuevo | Es el oráculo — óptimo por construcción (fuerza bruta). |
| `solvers/direct_qaoa.py` | Port de `poc-qaoa-agnostic-refinement` | Cruzado contra `exact` en bloques de juguete. |
| `solvers/r_qaoa.py` | Port + adaptación de escala (`_ising_adapter.py`) de `QAOA-in-QAOA-Test` | Cruzado contra `exact` en bloques de juguete (`<=10` nodos) — **no** validado todavía a escala real con la misma investigación por fuerza bruta que motivó la estrategia "swap" en el repo de origen. |
| `solvers/qaoa_square.py` | Port + reescritura del costo de hoja de `QAOA-in-QAOA-Test` | Mismo nivel que `r_qaoa`: validado en bloques de juguete, pendiente validación a escala real. |
| `evaluation/hotspots.py` | Port (sin boundary fields *k*=2-específicos) de `QAOA-in-QAOA-Test` | No tiene un test dedicado todavía — se ejerce indirectamente vía `cli.py`. |

**Siguiente paso natural:** correr `r_qaoa`/`qaoa_square` sobre los grafos reales de `data/` (no
solo bloques de juguete) y, si aparece algún caso sin mejora, investigarlo con la misma
disciplina de `poc-qaoa-agnostic-refinement/docs/Diagnostico_DSATUR.md` — fuerza bruta cuando el
tamaño lo permita, no solo aceptar "no mejoró" sin explicación.

## Idiomas y dependencias

100% Python (igual que los dos repos de origen). Dependencias nuevas respecto a la raíz del
repo: `qiskit-aer` (`toolkit/requirements.txt`) — todo lo demás (`qiskit`, `networkx`, `numpy`,
`scipy`) ya estaba en el `requirements.txt` de la raíz.
