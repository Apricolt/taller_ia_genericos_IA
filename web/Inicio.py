"""
Pagina de inicio del aplicativo web del Taller 1.

Ejecutar con:    streamlit run web/Inicio.py
"""

import _comun  # noqa: F401  (configura el sys.path del proyecto)
import streamlit as st

st.set_page_config(page_title="Taller 1 - Algoritmos Geneticos",
                   page_icon="ADN", layout="wide")

st.title("Taller 1 - Algoritmos Evolutivos / Algoritmos Geneticos")
st.caption("Inteligencia Artificial | Implementacion, experimentacion y analisis "
           "de algoritmos geneticos sobre cuatro problemas de optimizacion")

st.markdown(
    """
Este aplicativo despliega los cuatro problemas del taller. **Cada pagina del menu
lateral es una interfaz distinta**, con sus propios controles y su propia forma de
visualizar la solucion, pero todas ejecutan el **mismo motor generacional con
elitismo** (`ga/core.py`). Esa es justamente la idea de fondo del taller: lo que
cambia de un problema a otro no es el algoritmo, sino **como se representa una
solucion**, **como se mide su calidad** y **que operadores respetan esa representacion**.
"""
)

st.divider()
st.subheader("Los cuatro problemas")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
#### N-Reinas
Colocar N reinas sin que se ataquen.
- **Representacion:** permutacion de tamano N (indice = columna, valor = fila)
- **Aptitud:** minimizar los pares de reinas en conflicto (optimo = 0)
- **Operadores:** OX / PMX + intercambio
- **Visualizacion:** tablero de ajedrez

#### Agente viajero (TSP)
Visitar N ciudades una vez y volver al origen.
- **Representacion:** permutacion de ciudades, p. ej. `[0, 3, 1, 5, 2, 4]`
- **Aptitud:** `1 / (distancia + epsilon)` sobre el ciclo completo
- **Operadores:** OX / PMX + intercambio, inversion o insercion
- **Visualizacion:** mapa de la ruta
"""
    )

with col2:
    st.markdown(
        """
#### Asignacion de cursos a salas
Programar 8 cursos en 4 salas y 5 franjas.
- **Representacion:** vector entero, cada gen es un slot sala-franja
- **Aptitud:** minimizar penalizaciones duras (1000) y blandas (1 a 10)
- **Operadores:** cruce de un punto o uniforme + reasignacion
- **Visualizacion:** horario salas x franjas

#### Mochila (knapsack 0/1)
Maximizar el valor sin superar la capacidad.
- **Representacion:** cromosoma binario (1 = se lleva el objeto)
- **Aptitud:** valor total, con penalizacion o reparacion del exceso
- **Operadores:** cruce de un punto o uniforme + bit-flip
- **Visualizacion:** tabla de objetos y brecha frente al optimo exacto
"""
    )

st.divider()
st.subheader("Por que una permutacion no admite los mismos operadores que un binario")

st.markdown(
    """
En la mochila los genes son **independientes**: llevar el objeto 3 no impide llevar
el 7, asi que cualquier vector de ceros y unos representa una seleccion valida y el
cruce de un punto funciona sin problema.

En el TSP y en N-Reinas los genes estan **acoplados**: visitar una ciudad impide
volver a visitarla. Un cruce de un punto mezcla dos rutas y produce ciudades
repetidas y ciudades sin visitar, es decir, un cromosoma que ya no es una ruta.
Por eso hacen falta operadores que preserven la permutacion, como **OX** y **PMX**.
"""
)

col1, col2 = st.columns(2)
col1.code(
    "Padre 1  [5, 0, 1, 4, | 2, 6, 3, 7]\n"
    "Padre 2  [1, 6, 7, 2, | 3, 4, 5, 0]\n"
    "-----------------------------------\n"
    "Hijo     [5, 0, 1, 4,   3, 4, 5, 0]\n\n"
    "Repetidas   : 0, 4, 5\n"
    "No visitadas: 2, 6, 7   -> RUTA INVALIDA",
    language="text")
col2.markdown(
    """
El ejemplo de la izquierda es el resultado real de aplicar un cruce de un punto
sobre dos permutaciones (`experimentos/exp_tsp.py`). El hijo visita tres ciudades
dos veces y se olvida de otras tres.

Con **OX** se copia un segmento de un padre y se rellena el resto con el orden del
otro, saltando lo que ya esta; con **PMX** se intercambia un segmento y se usa el
mapeo que induce para deshacer los duplicados. Ambos devuelven **siempre** una
permutacion valida.
"""
)

st.divider()
st.info("Elige un problema en el menu lateral para ejecutar el algoritmo con tus "
        "propios parametros.")

_comun.pie_de_pagina()
