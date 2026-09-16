# PoC — Capa de Refinamiento Cuántica Adaptativa Agnóstica
## (Multi-Subgrafo Residual + Enrutamiento Adaptativo R-QAOA / QAOA²)

> **NTT DATA Perú — Equipo de Quantum Computing**  
> **Caso Ficticio:** "Urbania Telecom" (Planificación de frecuencias y reducción de interferencias)  
> **Implementación:** 100% Qiskit (simulación exacta por Statevector y ansatz variacional con Warm-Start)  
> **Métodos Cuánticos:**  
>   * **R-QAOA (Recursive QAOA, Bravyi et al. 2020):** Para topologías y subgrafos dispersos (densidad < 15%).  
>   * **$\text{QAOA}^2$ (QAOA-in-QAOA, Zhou et al. 2023):** Para topologías y subgrafos densos (densidad $\ge 15\%$).

---

## 1. Contexto y Objetivos

Este proyecto demuestra empíricamente que una capa de optimización cuántica variacional adaptativa puede **insertarse de forma completamente agnóstica encima de cualquier heurística clásica de producción**, extrayendo múltiples focos de interferencia residual a partir de un umbral, enrutando subgrafos dispersos a R-QAOA y densos a $\text{QAOA}^2$, sin requerir reemplazar el software existente del cliente, y garantizando por diseño que el resultado **nunca empeore**.

A diferencia de proyectos de Feature Extraction (como DQFE para Komatsu, que se entrena una sola vez), este enfoque variacional está diseñado para **servicios recurrentes de re-optimización** (semanal o mensual) conforme evoluciona el tráfico y la topología de red.

---

## 2. Arquitectura del Pipeline

```
1. Grafo de Celdas Urbania Telecom (posiciones geográficas, solapamiento RF y distritos de Lima)
        │
2. Heurística Clásica en Producción (Greedy o DSATUR)
        │
3. Extracción de Múltiples Focos Residuales Críticos (Umbral de Interferencia)
        │  * Distribución de residuos por celda y corte por percentil (p70)
        │  * Detección de componentes conexos de conflicto (hotspots independientes)
        │  * Campos de acoplamiento estático de frontera h_u
        │
4. Enrutamiento Cuántico Adaptativo por Subgrafo:
        │  ¿Densidad del subgrafo / topología < 15%?
       ┌┴──────────────────────────────┐
       ▼                               ▼
 [Sí: R-QAOA (Bravyi et al.)]    [No: QAOA² (Zhou et al.)]
 * Eliminación recursiva de      * Particionamiento modular ≤ 8 qubits
   correlaciones <Z_u Z_v>       * Contracción a super-grafo
 * Óptimo en grafos dispersos    * Óptimo en comunidades densas
       │                               │
       └───────────────┬───────────────┘
                       ▼
5. Evaluación y Mecanismo de Fallback de Seguridad por Subregión
        │  ¿Costo(Híbrido) < Costo(Clásico)?
       ┌┴──────────────────────────────┐
       ▼                               ▼
 [Sí: Adoptar Híbrido]       [No: Fallback al Clásico]
       │                               │
       └───────────────┬───────────────┘
                       ▼
    Solución Final (Garantía Matemática: ΔCosto <= 0)
```


---

## 3. Ejemplos Evaluados (3 Topologías x 2 Escenarios Temporales)

El proyecto incluye 3 instancias de grafo generadas a partir de datos técnicos referenciales de **OSIPTEL** (macroceldas con radios amplios y microceldas urbanas densas en distritos como San Isidro, Miraflores, Surco, Cercado y San Borja):

1. **`sparse_100` (100 nodos, 450 aristas):**
   * Corresponde a la escala base del benchmark del paper original de Zhou et al. (2023) (`data/test.json`).
   * Topología de macroceldas suburbanas con grado promedio $\approx 9$.
2. **`dense_100` (100 nodos, 3800 aristas):**
   * Grafo densamente conectado con microceldas de alta densidad y solapamiento severo (densidad $\approx 77\%$).
   * Diseñado específicamente para poner a prueba la hipótesis del paper de que $\text{QAOA}^2$ captura ventajas notables en grafos densos.
3. **`realistic_metropolitan` (600 nodos, 4200 aristas):**
   * Despliegue a escala metropolitana real de Lima cubriendo 5 distritos con clusters urbanos.
   * Tamaño intratable para solvers exactos clásicos en tiempo de producción.

Para cada instancia se modelaron **dos escenarios temporales** sobre la misma estructura de grafo:
* **`hora_pico`:** Sobrecarga de tráfico (+60-80% penalización de interferencia, ruido de congestión).
* **`hora_valle`:** Tráfico moderado y baja interferencia.

---

## 4. Estructura del Repositorio

```
poc-qaoa-agnostic-refinement/
├── README.md                          # Documentación completa del PoC
├── requirements.txt                   # Dependencias (qiskit, networkx, numpy, scipy)
├── data/                              # Instancias de grafos generadas (JSON)
│   ├── sparse_100_hora_pico.json
│   ├── sparse_100_hora_valle.json
│   ├── dense_100_hora_pico.json
│   ├── dense_100_hora_valle.json
│   ├── realistic_metropolitan_hora_pico.json
│   └── realistic_metropolitan_hora_valle.json
├── src/
│   ├── data/
│   │   └── generate_graph.py          # Generador sintético basado en OSIPTEL
│   ├── baselines/
│   │   ├── greedy_coloring.py         # Heurística Welsh-Powell
│   │   └── dsatur.py                  # Heurística Degree of Saturation
│   ├── quantum/
│   │   ├── qaoa.py                    # QAOA en Qiskit nativo con warm-start
│   │   ├── qaoa_square.py             # QAOA² jerárquico y refine(graph, sol) adaptativo
│   │   └── r_qaoa.py                  # R-QAOA (Recursive QAOA) para grafos dispersos
│   ├── evaluation/
│   │   └── cost_function.py           # Función de costo y extracción multi-subgrafo
│   └── run_experiment.py              # Orquestador automatizado de las 12 pruebas
└── results/
    ├── comparison_report.md           # Reporte final con tabla comparativa y análisis
    └── experiment_data.json           # Métricas brutas en JSON
```

---

## 5. Instrucciones de Ejecución

### Prerrequisitos
Tener instalado Python 3.10+ con Qiskit y NetworkX:
```bash
pip install -r requirements.txt
```

### 1. Generar los Grafos Sintéticos
```bash
python src/data/generate_graph.py
```

### 2. Ejecutar la Batería Completa de Experimentos
```bash
python src/run_experiment.py
```

El script ejecutará automáticamente las 12 combinaciones y actualizará `results/comparison_report.md` y `results/experiment_data.json`.

---

## 6. Resumen de Resultados Principales

De las 12 ejecuciones evaluadas:
* **Criterio "Nunca Empeora": 100% de cumplimiento** (en ningún escenario la solución final tuvo mayor interferencia que el baseline clásico).
* **R-QAOA en Grafos Dispersos:** Resolvió con éxito la limitación teórica de QAOA² en `sparse_100` pico, logrando una reducción de **-27.06 puntos (-1.21%)** con Greedy (donde antes se activaba fallback) y **-27.06 puntos (-1.19%)** con DSATUR.
* **Refinamiento Multi-Subgrafo:** En la red metropolitana (600 nodos), el sistema identificó 4 hotspots independientes, alcanzando reducciones de hasta **-160.56 puntos** en hora pico.
* **Enrutamiento Inteligente:** Demostró la efectividad de combinar R-QAOA en macroceldas dispersas con $\text{QAOA}^2$ en microceldas densas.

Para más detalles y la tabla completa de métricas, consultar el [Reporte Comparativo](results/comparison_report.md).

