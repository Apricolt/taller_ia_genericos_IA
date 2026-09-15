"""Interfaz del problema base: N-Reinas."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import _comun
import streamlit as st

from problemas.nreinas import (resolver, conflictos, dibujar_tablero,
                               tablero_texto, matriz_tablero)

st.set_page_config(page_title="N-Reinas", page_icon="Q", layout="wide")

_comun.encabezado(
    "N-Reinas",
    "Colocar N reinas en un tablero de N x N sin que ninguna ataque a otra",
    "Permutacion de tamano N. El **indice** es la columna y el **valor** es la "
    "fila. Al ser permutacion, ninguna reina comparte fila ni columna: solo "
    "quedan por resolver las diagonales.",
    "Numero de pares de reinas que se atacan, **a minimizar**. El optimo vale 0, "
    "asi que la corrida se detiene apenas lo alcanza.",
)

# ------------------------------- Controles --------------------------------
st.sidebar.header("N-Reinas")
n = st.sidebar.slider("N (numero de reinas)", 4, 30, 8,
                      help="El enunciado pide probar N=6 y N=8")
cruce = st.sidebar.selectbox("Operador de cruce", ["OX", "PMX"],
                             help="Ambos preservan la permutacion")
mutacion = st.sidebar.selectbox("Operador de mutacion", ["swap", "inversion"])

params = _comun.controles_geneticos(tam_poblacion=50, generaciones=300,
                                    tasa_mutacion=0.1,
                                    ayuda_mutacion="Probabilidad de mutar cada hijo")

st.sidebar.caption(
    "Sugerencia: con poblaciones de 100 o mas, para N=6 y N=8 la solucion suele "
    "aparecer ya en la poblacion inicial. Baja la poblacion a 20 para ver "
    "trabajar al algoritmo."
)

# ------------------------------- Ejecucion --------------------------------
if st.sidebar.button("Ejecutar algoritmo genetico", type="primary",
                     use_container_width=True):
    with st.spinner(f"Evolucionando {params['tam_poblacion']} tableros..."):
        res = resolver(n, cruce=cruce, mutacion=mutacion, **params)
    st.session_state["nreinas"] = (res, n, cruce, mutacion)

if "nreinas" not in st.session_state:
    st.info("Ajusta los parametros en la barra lateral y pulsa "
            "**Ejecutar algoritmo genetico**.")
    _comun.pie_de_pagina()
    st.stop()

res, n, cruce, mutacion = st.session_state["nreinas"]
conflictos_finales = conflictos(res.mejor_individuo)

# ------------------------------- Resultados -------------------------------
st.divider()
if res.exito:
    st.success(f"Solucion perfecta encontrada en la generacion "
               f"**{res.generacion_mejor}** (0 conflictos).")
else:
    st.warning(f"No se alcanzo una solucion perfecta en "
               f"{res.generaciones_ejecutadas} generaciones. "
               f"Mejor resultado: {conflictos_finales} conflictos.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Conflictos", conflictos_finales)
c2.metric("Generacion de la mejor", res.generacion_mejor)
c3.metric("Generaciones ejecutadas", res.generaciones_ejecutadas)
c4.metric("Tiempo", f"{res.tiempo_seg:.3f} s")

col_tablero, col_grafica = st.columns([1, 1])

with col_tablero:
    st.subheader("Tablero")
    st.pyplot(dibujar_tablero(
        res.mejor_individuo,
        titulo=f"N={n} | {cruce} + {mutacion} | {conflictos_finales} conflictos"))

with col_grafica:
    st.subheader("Convergencia")
    st.pyplot(_comun.grafica_convergencia(
        res.historial_mejor, res.historial_promedio,
        ylabel="Conflictos (menor es mejor)",
        transformar=lambda v: -v,
        titulo="Conflictos por generacion"))

    st.markdown("**Cromosoma de la solucion**")
    st.code(str([int(g) for g in res.mejor_individuo]), language="python")
    st.caption("Cada posicion es una columna del tablero; el valor indica la fila "
               "donde se coloco la reina de esa columna.")

with st.expander("Ver el tablero en texto y como matriz"):
    st.code(tablero_texto(res.mejor_individuo), language="text")
    st.dataframe(matriz_tablero(res.mejor_individuo), use_container_width=True)

_comun.pie_de_pagina()
