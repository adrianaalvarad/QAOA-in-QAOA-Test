# Cuántos colores (k) usar — valores de referencia, no arbitrarios

## Reuso hexagonal clásico

La teoría clásica de reuso de frecuencias celulares (geometría hexagonal de celdas) restringe
los tamaños de clúster válidos a:

$$N = i^2 + i j + j^2, \quad i, j \in \mathbb{Z}_{\ge 0}$$

lo que da $N \in \{1, 3, 4, 7, 9, 12, 13, 16, 19, 21, \dots\}$ — no cualquier entero sirve como
patrón de reuso hexagonal exacto. Los valores citados como referencia estándar en manuales de
ingeniería RF (p.ej. T. Rappaport, *Wireless Communications: Principles and Practice*, capítulo
"The Cellular Concept") son:

| N | Contexto histórico citado |
|---|---|
| 3 | Patrón ajustado, alta capacidad, exige buena relación portadora/interferencia (C/I). |
| 4 | Citado en despliegues AMPS/GSM tempranos. |
| 7 | El ejemplo "de manual" del reuso hexagonal — el más citado en literatura introductoria. |
| 9 | GSM urbano denso, mayor tolerancia a interferencia que N=7. |
| 12 | Baja reutilización, redes 2G de gran escala. |

Este proyecto usa por defecto **k=6** (`toolkit/config.py::DEFAULT_N_COLORS`), heredado de
`poc-qaoa-agnostic-refinement`: no es uno de los valores "clásicos" exactos de la tabla, se eligió
empíricamente como el punto donde el problema sintético deja de ser trivial (con k=12 el grafo se
resuelve sin conflicto) ni degenerado (con k=2 las dos heurísticas de baseline convergen casi a
la misma respuesta en grafos densos — ver comparación entre `poc-qaoa-agnostic-refinement` y
`QAOA-in-QAOA-Test` original). `toolkit/cli.py` acepta `--k` para correr con cualquier valor,
incluidos los de la tabla.

## La salvedad importante: esto es un modelo de planificación clásico, no una descripción de 5G

Redes LTE y 5G modernas **no** operan mayormente con reuso-N estricto. La práctica dominante hoy
es **reuse-1** (todas las celdas pueden usar todo el espectro) combinado con:

- **ICIC / eICIC** (Inter-Cell Interference Coordination / enhanced): coordinación dinámica entre
  celdas vecinas, no una asignación estática de "colores".
- **Fractional Frequency Reuse (FFR)**: solo el borde de celda (donde la interferencia es peor)
  se restringe a un subconjunto de frecuencias; el centro de celda usa todo el espectro.
- Scheduling dinámico por PRB (Physical Resource Block) en el dominio tiempo-frecuencia, no una
  asignación fija por celda.

El modelo de "coloreo de grafo con k colores" que usa este kit es una **abstracción de
planificación clásica (2G/3G)**, útil porque da un problema combinatorio bien definido para
demostrar el mecanismo de refinamiento cuántico — no una representación literal de cómo opera una
red 5G real hoy. Esta distinción debe mantenerse explícita en cualquier material dirigido a un
cliente: no vender la analogía "k colores" como si describiera su red 5G actual.
