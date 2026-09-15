"""
Utilidades compartidas por las paginas del aplicativo web.

Streamlit trata como pagina a cada archivo dentro de web/pages/, de modo que
este modulo (fuera de esa carpeta y con prefijo _) queda como codigo de apoyo.
"""

import os
import sys

# Se agrega la raiz del proyecto al path para poder importar ga/ y problemas/
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st


def encabezado(titulo, subtitulo, representacion, aptitud):
    """Cabecera comun: titulo del problema y su ficha tecnica."""
    st.title(titulo)
    st.caption(subtitulo)
    col1, col2 = st.columns(2)
    col1.info(f"**Representacion**\n\n{representacion}")
    col2.success(f"**Funcion de aptitud**\n\n{aptitud}")


def controles_geneticos(tam_poblacion=100, generaciones=300, tasa_mutacion=0.1,
                        ayuda_mutacion="Probabilidad de aplicar el operador de mutacion"):
    """Controles que comparten los cuatro problemas, en la barra lateral."""
    st.sidebar.subheader("Parametros del algoritmo genetico")
    params = {
        "tam_poblacion": st.sidebar.slider(
            "Tamano de la poblacion", 10, 300, tam_poblacion, step=10,
            help="Numero de soluciones candidatas que evolucionan en paralelo"),
        "generaciones": st.sidebar.slider(
            "Generaciones maximas", 10, 600, generaciones, step=10,
            help="Ciclos de evaluacion, seleccion, cruce y mutacion"),
        "tasa_mutacion": st.sidebar.slider(
            "Tasa de mutacion", 0.0, 1.0, tasa_mutacion, step=0.01,
            help=ayuda_mutacion),
        "elitismo": st.sidebar.slider(
            "Elitismo (individuos conservados)", 0, 10, 2,
            help="0 desactiva el elitismo: el mejor individuo puede perderse"),
        "seleccion": st.sidebar.selectbox(
            "Metodo de seleccion", ["torneo", "ruleta", "truncamiento"],
            help="Determina la presion selectiva sobre la poblacion"),
    }
    params["semilla"] = st.sidebar.number_input(
        "Semilla aleatoria", min_value=0, max_value=9999, value=42,
        help="Fija el azar para que la corrida sea reproducible")
    return params


def grafica_convergencia(historial_mejor, historial_promedio, ylabel,
                         transformar=None, titulo="Convergencia"):
    """Curva del mejor y del promedio de la poblacion por generacion."""
    if transformar:
        historial_mejor = [transformar(v) for v in historial_mejor]
        historial_promedio = [transformar(v) for v in historial_promedio]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(historial_mejor, label="Mejor individuo", linewidth=2, color="#4C72B0")
    ax.plot(historial_promedio, label="Promedio de la poblacion", linewidth=1.2,
            color="#C44E52", alpha=0.75)
    ax.set_xlabel("Generacion")
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def pie_de_pagina():
    st.divider()
    st.caption(
        "Taller 1 - Inteligencia Artificial | Algoritmos Evolutivos / Geneticos. "
        "Las cuatro paginas comparten el mismo motor generacional con elitismo "
        "(`ga/core.py`) y solo cambian la representacion, la funcion de aptitud "
        "y los operadores."
    )
