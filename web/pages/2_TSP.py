"""Interfaz del Ejercicio 1: problema del agente viajero (TSP)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import _comun
import numpy as np
import pandas as pd
import streamlit as st

from problemas.tsp import (crear_instancia, resolver, distancia_ruta,
                           dibujar_ruta, ruta_vecino_mas_cercano,
                           optimo_exacto, aptitud_a_distancia)

st.set_page_config(page_title="Agente viajero (TSP)", page_icon="Ruta", layout="wide")

_comun.encabezado(
    "Ejercicio 1 - Agente viajero (TSP)",
    "Visitar N ciudades exactamente una vez y regresar a la ciudad inicial "
    "recorriendo la menor distancia posible",
    "Permutacion de las ciudades, por ejemplo `[0, 3, 1, 5, 2, 4]`. Se interpreta "
    "como un ciclo: al llegar a la ultima ciudad se vuelve a la primera.",
    "Distancia total del ciclo **incluido el regreso**. Como el motor maximiza, "
    "se usa `1 / (distancia + epsilon)`, que ademas es siempre positiva y por eso "
    "admite seleccion por ruleta.",
)

# ------------------------------- Controles --------------------------------
st.sidebar.header("Agente viajero")
n_ciudades = st.sidebar.slider("Numero de ciudades", 5, 50, 15,
                               help="El enunciado pide 8, 10 o 15 ciudades")
semilla_mapa = st.sidebar.number_input("Semilla del mapa", 0, 9999, 2024,
                                       help="Define las coordenadas de las ciudades")
cruce = st.sidebar.selectbox("Operador de cruce", ["OX", "PMX"])
mutacion = st.sidebar.selectbox("Operador de mutacion",
                                ["inversion", "swap", "insercion"],
                                help="La inversion equivale a un movimiento 2-opt")

params = _comun.controles_geneticos(tam_poblacion=100, generaciones=300,
                                    tasa_mutacion=0.2,
                                    ayuda_mutacion="Probabilidad de mutar cada hijo")

coordenadas, matriz = crear_instancia(n_ciudades, semilla=int(semilla_mapa))

# ------------------------------- Ejecucion --------------------------------
if st.sidebar.button("Ejecutar algoritmo genetico", type="primary",
                     use_container_width=True):
    with st.spinner(f"Buscando la mejor ruta entre {n_ciudades} ciudades..."):
        res = resolver(matriz, cruce=cruce, mutacion=mutacion, **params)
    st.session_state["tsp"] = (res, n_ciudades, int(semilla_mapa), cruce, mutacion)

if "tsp" not in st.session_state:
    st.info("Ajusta los parametros en la barra lateral y pulsa "
            "**Ejecutar algoritmo genetico**.")
    st.subheader("Mapa de ciudades")
    st.pyplot(dibujar_ruta(coordenadas, np.arange(n_ciudades),
                           titulo="Ruta inicial en orden 0, 1, 2, ... (sin optimizar)"))
    _comun.pie_de_pagina()
    st.stop()

res, n_ciudades, semilla_mapa, cruce, mutacion = st.session_state["tsp"]
coordenadas, matriz = crear_instancia(n_ciudades, semilla=semilla_mapa)
distancia = distancia_ruta(res.mejor_individuo, matriz)
voraz = distancia_ruta(ruta_vecino_mas_cercano(matriz), matriz)

# ------------------------------- Resultados -------------------------------
st.divider()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Distancia de la mejor ruta", f"{distancia:.2f}")
c2.metric("Vecino mas cercano (voraz)", f"{voraz:.2f}",
          delta=f"{distancia - voraz:+.2f}", delta_color="inverse",
          help="Heuristica simple usada como punto de comparacion")
c3.metric("Generacion de la mejor", res.generacion_mejor)
c4.metric("Tiempo", f"{res.tiempo_seg:.3f} s")

if n_ciudades <= 10:
    _, optima = optimo_exacto(matriz)
    brecha = 100 * (distancia - optima) / optima
    if brecha < 1e-9:
        st.success(f"El algoritmo alcanzo el **optimo exacto** ({optima:.2f}), "
                   f"verificado por fuerza bruta sobre las {n_ciudades - 1}! rutas.")
    else:
        st.warning(f"Optimo exacto por fuerza bruta: {optima:.2f}. "
                   f"Brecha del algoritmo genetico: {brecha:.2f} %.")
else:
    st.caption(f"Con {n_ciudades} ciudades el espacio tiene "
               f"({n_ciudades}-1)!/2 rutas posibles, demasiadas para calcular el "
               f"optimo exacto por fuerza bruta. Se compara contra la heuristica "
               f"del vecino mas cercano.")

col_mapa, col_grafica = st.columns([1, 1])

with col_mapa:
    st.subheader("Mejor ruta encontrada")
    st.pyplot(dibujar_ruta(
        coordenadas, res.mejor_individuo,
        titulo=f"{n_ciudades} ciudades | {cruce} + {mutacion} | "
               f"distancia {distancia:.2f}"))

with col_grafica:
    st.subheader("Convergencia")
    st.pyplot(_comun.grafica_convergencia(
        res.historial_mejor, res.historial_promedio,
        ylabel="Distancia de la ruta",
        transformar=aptitud_a_distancia,
        titulo="Distancia por generacion"))

    st.markdown("**Orden de visita**")
    orden = [int(c) for c in res.mejor_individuo]
    st.code(" -> ".join(f"C{c}" for c in orden) + f" -> C{orden[0]}", language="text")

with st.expander("Ver la matriz de distancias"):
    etiquetas = [f"C{i}" for i in range(n_ciudades)]
    st.dataframe(pd.DataFrame(matriz.round(2), index=etiquetas, columns=etiquetas),
                 use_container_width=True)

with st.expander("Ver el tramo a tramo de la ruta"):
    siguiente = np.roll(res.mejor_individuo, -1)
    st.dataframe(pd.DataFrame({
        "desde": [f"C{int(c)}" for c in res.mejor_individuo],
        "hasta": [f"C{int(c)}" for c in siguiente],
        "distancia": [round(float(matriz[a, b]), 2)
                      for a, b in zip(res.mejor_individuo, siguiente)],
    }), use_container_width=True, hide_index=True)

_comun.pie_de_pagina()
