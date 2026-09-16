# Reporte de Resultados: Capa de Refinamiento Cuántica Adaptativa Agnóstica
## (Multi-Subgrafo Residual + Selección Adaptativa R-QAOA / QAOA²)

**Proyecto:** NTT DATA Perú — Quantum Computing Team  
**Caso de Estudio Ficticio:** Urbania Telecom (Planificación de frecuencias y mitigación de interferencias)  
**Framework Cuántico:** Qiskit 2.3 (Statevector exacto y ansatz variacional con warm-start)  
**Métodos Cuánticos:**
* **R-QAOA (Recursive QAOA, Bravyi et al. 2020):** Para topologías y subgrafos dispersos (densidad < 15%).
* **QAOA² (QAOA-in-QAOA, Zhou et al. 2023):** Para topologías y subgrafos densos (densidad >= 15%).
**Fecha:** 16 de septiembre de 2026  

---

## 1. Resumen Ejecutivo

El presente experimento evalúa empíricamente la hipótesis central de la PoC: **una capa de optimización cuántica variacional adaptativa puede insertarse de forma 100% agnóstica sobre cualquier heurística clásica de telecomunicaciones ya existente en producción, extrayendo múltiples focos de interferencia residual a partir de un umbral, enrutando subgrafos dispersos a R-QAOA y densos a QAOA², garantizando que el costo total nunca empeore.**

### Métricas Globales del PoC:
* **Total de ejecuciones evaluadas:** 12 (3 topologías de red × 2 escenarios temporales × 2 heurísticas clásicas)
* **Tasa de cumplimiento del criterio 'Nunca Empeora':** **100.0%** (Cumplido en todas las corridas).
* **Tasa de mejora activa:** **12/12 (100.0%)** de los escenarios lograron reducción neta de interferencia.
* **Reducción porcentual promedio (en corridas mejoradas):** **1.62%** sobre el costo global de la red.

---

## 2. Tabla Comparativa de Experimentos (12 Escenarios)

| Instancia de Red | Escenario | Heurística Clásica | Solvers Cuánticos | Subgrafos | Costo Clásico (pts) | Costo Híbrido (pts) | Delta Absoluto (pts) | Delta % | Estado | Criterio 'Nunca Empeora' |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `sparse_100` (100V, 450E) | Pico | Greedy (Welsh-Powell) | `R-QAOA` | 3 | 2238.62 | 2174.20 | **-64.43** | **-2.88%** | `IMPROVED` | ✓ CUMPLE |
| `sparse_100` (100V, 450E) | Pico | DSATUR | `R-QAOA` | 3 | 2276.63 | 2239.97 | **-36.66** | **-1.61%** | `IMPROVED` | ✓ CUMPLE |
| `sparse_100` (100V, 450E) | Valle | Greedy (Welsh-Powell) | `R-QAOA` | 3 | 1027.42 | 988.86 | **-38.56** | **-3.75%** | `IMPROVED` | ✓ CUMPLE |
| `sparse_100` (100V, 450E) | Valle | DSATUR | `R-QAOA` | 3 | 1031.71 | 971.76 | **-59.95** | **-5.81%** | `IMPROVED` | ✓ CUMPLE |
| `dense_100` (100V, 3800E) | Pico | Greedy (Welsh-Powell) | `QAOA^2` | 4 | 11885.74 | 11877.57 | **-8.17** | **-0.07%** | `IMPROVED` | ✓ CUMPLE |
| `dense_100` (100V, 3800E) | Pico | DSATUR | `QAOA^2` | 4 | 11885.74 | 11877.57 | **-8.17** | **-0.07%** | `IMPROVED` | ✓ CUMPLE |
| `dense_100` (100V, 3800E) | Valle | Greedy (Welsh-Powell) | `QAOA^2` | 4 | 5180.43 | 5127.74 | **-52.68** | **-1.02%** | `IMPROVED` | ✓ CUMPLE |
| `dense_100` (100V, 3800E) | Valle | DSATUR | `QAOA^2` | 4 | 5180.43 | 5127.74 | **-52.68** | **-1.02%** | `IMPROVED` | ✓ CUMPLE |
| `realistic_metropolitan` (600V, 4200E) | Pico | Greedy (Welsh-Powell) | `R-QAOA` | 4 | 25349.69 | 25028.74 | **-320.95** | **-1.27%** | `IMPROVED` | ✓ CUMPLE |
| `realistic_metropolitan` (600V, 4200E) | Pico | DSATUR | `R-QAOA` | 4 | 25463.02 | 25398.35 | **-64.67** | **-0.25%** | `IMPROVED` | ✓ CUMPLE |
| `realistic_metropolitan` (600V, 4200E) | Valle | Greedy (Welsh-Powell) | `R-QAOA` | 4 | 10963.90 | 10872.64 | **-91.26** | **-0.83%** | `IMPROVED` | ✓ CUMPLE |
| `realistic_metropolitan` (600V, 4200E) | Valle | DSATUR | `R-QAOA` | 4 | 11011.41 | 10911.97 | **-99.44** | **-0.90%** | `IMPROVED` | ✓ CUMPLE |

---

## 3. Análisis de Hallazgos Técnicos

### 3.1. Extracción de Múltiples Subgrafos Residuales mediante Umbral
En lugar de restringirse a un único subgrafo aislado, el sistema calcula la distribución de interferencia residual por nodo y extrae los componentes conexos de conflicto que superan el umbral (percentil 70%).
* En la red metropolitana (600 nodos), esto permite aislar y tratar focos de congestión independientes simultáneamente en distritos separados (ej. San Isidro y Miraflores).
* Cada subregión se optimiza secuencialmente actualizando los campos de frontera de los vecinos, asegurando convergencia coordinada.

### 3.2. Selección Adaptativa de Algoritmo Cuántico: R-QAOA vs. QAOA²
* **Para grafos dispersos (`sparse_100` con densidad ~9% y cadenas de macroceldas):**
  Como identificó Zhou et al. (2023), QAOA² pierde eficacia en grafos sin comunidades densas definidas. La incorporación de **R-QAOA (Bravyi et al., PRL 2020)** resuelve esta limitación: elimina variables iterativamente con base en las correlaciones cuánticas de dos puntos $\langle Z_u Z_v \rangle$, superando el horizonte de causalidad local de QAOA y desbloqueando mejoras donde QAOA² antes requería fallback.
* **Para grafos densos (`dense_100` con densidad ~77% y microceldas urbanas):**
  La alta modularidad permite que **QAOA²** agrupe celdas en bloques conexos de $\le 8$ qubits, contraiga interacciones efectivas $J_{ij}$ y resuelva la orientación global en el super-grafo.

### 3.3. Agnosticidad Estricta y Fallback Protector
La interfaz `refine(graph, candidate_solution)` opera de forma idéntica sin conocer el algoritmo generador (Greedy o DSATUR). Tanto a nivel local por subgrafo como a nivel global, se compara el costo antes y después de la propuesta cuántica. Si en alguna subregión la propuesta no supera a la clásica, se activa el fallback conservando la solución clásica.

---

## 4. Conclusión para la Conversación con el Cliente (Urbania Telecom)

> La incorporación de múltiples subgrafos residuales y el enrutamiento adaptativo entre **R-QAOA** (para macroceldas dispersas) y **QAOA²** (para microceldas densas) responde directamente a las particularidades físicas de la red. La capa cuántica actúa como un refinador inteligente y agnóstico que respeta la infraestructura clásica del cliente, ataca focos críticos en paralelo y ofrece cero riesgo de degradación operativa.