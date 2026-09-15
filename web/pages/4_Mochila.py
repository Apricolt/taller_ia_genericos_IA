"""Interfaz del Ejercicio 3: problema de la mochila (knapsack 0/1)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import _comun
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from problemas.mochila import (crear_objetos, resolver, optimo_exacto,
                               resumen_solucion, seleccion_dataframe,
                               lambda_penalizacion)

st.set_page_config(page_title="Mochila", page_icon="Mochila", layout="wide")

_comun.encabezado(
    "Ejercicio 3 - Problema de la mochila (knapsack 0/1)",
    "Seleccionar objetos que maximicen el valor total sin superar la capacidad",
    "Cromosoma **binario**: gen = 1 si el objeto se lleva, 0 si no. Funciona "
    "porque la decision sobre cada objeto es independiente de las demas, a "
    "diferencia del TSP, donde visitar una ciudad impide volver a visitarla.",
    "Valor total de los objetos elegidos, **a maximizar**. El exceso de peso se "
    "trata con **penalizacion** (`valor - lambda x exceso`) o con **reparacion** "
    "(se quitan objetos de peor relacion valor/peso hasta que cabe).",
)

# ------------------------------- Controles --------------------------------
st.sidebar.header("Mochila")
n_objetos = st.sidebar.slider("Numero de objetos", 10, 100, 15,
                              help="El enunciado pide al menos 15 objetos")
semilla_datos = st.sidebar.number_input("Semilla de los objetos", 0, 9999, 2024)

pesos, valores = crear_objetos(int(n_objetos), semilla=int(semilla_datos))
peso_total_objetos = int(pesos.sum())

fraccion = st.sidebar.slider(
    "Capacidad de la mochila (% del peso total)", 5, 95, 50, step=5,
    help=f"El peso de todos los objetos suma {peso_total_objetos}")
capacidad = int(round(peso_total_objetos * fraccion / 100))
st.sidebar.caption(f"Capacidad = **{capacidad}** de {peso_total_objetos}")

manejo = st.sidebar.radio(
    "Manejo del exceso de peso",
    ["reparacion", "penalizacion fuerte", "penalizacion debil"],
    help="La penalizacion debil usa lambda = max(valor/peso) y conserva el "
         "gradiente, pero puede dejar ganar a una solucion invalida. "
         "La fuerte usa lambda = suma de valores y lo impide.")
OPCIONES = {
    "reparacion": {"estrategia": "reparar"},
    "penalizacion fuerte": {"estrategia": "penalizar", "modo_penalizacion": "fuerte"},
    "penalizacion debil": {"estrategia": "penalizar",
                           "modo_penalizacion": "proporcional"},
}
cruce = st.sidebar.selectbox("Operador de cruce", ["un_punto", "uniforme"])

params = _comun.controles_geneticos(
    tam_poblacion=60, generaciones=120, tasa_mutacion=0.05,
    ayuda_mutacion="Probabilidad de invertir CADA gen (bit-flip por gen)")

# ------------------------------- Ejecucion --------------------------------
if st.sidebar.button("Ejecutar algoritmo genetico", type="primary",
                     use_container_width=True):
    with st.spinner(f"Seleccionando entre {n_objetos} objetos..."):
        res = resolver(pesos, valores, capacidad, cruce=cruce,
                       **OPCIONES[manejo], **params)
    st.session_state["mochila"] = (res, int(n_objetos), int(semilla_datos),
                                   capacidad, manejo, cruce)

if "mochila" not in st.session_state:
    st.info("Ajusta los parametros en la barra lateral y pulsa "
            "**Ejecutar algoritmo genetico**.")
    st.subheader("Objetos disponibles")
    st.dataframe(seleccion_dataframe(np.zeros(len(pesos), dtype=int), pesos, valores)
                 .drop(columns=["seleccionado"]),
                 use_container_width=True, hide_index=True)
    _comun.pie_de_pagina()
    st.stop()

res, n_objetos, semilla_datos, capacidad, manejo, cruce = st.session_state["mochila"]
pesos, valores = crear_objetos(n_objetos, semilla=semilla_datos)
r = resumen_solucion(res.mejor_individuo, pesos, valores, capacidad)
_, valor_optimo = optimo_exacto(pesos, valores, capacidad)
brecha = 100 * (valor_optimo - r["valor"]) / valor_optimo

# ------------------------------- Resultados -------------------------------
st.divider()
if not r["valido"]:
    st.error(f"**La mejor solucion NO cabe en la mochila**: pesa {r['peso']} y la "
             f"capacidad es {capacidad}. Esto es lo que ocurre con una "
             f"penalizacion demasiado debil: una solucion invalida puede superar "
             f"en aptitud a la mejor solucion valida.")
elif brecha < 1e-9:
    st.success(f"El algoritmo alcanzo el **optimo exacto** ({valor_optimo}), "
               f"verificado por programacion dinamica.")
else:
    st.warning(f"Optimo exacto (programacion dinamica): {valor_optimo}. "
               f"Brecha del algoritmo genetico: {brecha:.2f} %.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Valor obtenido", r["valor"], delta=f"{r['valor'] - valor_optimo:+d}",
          help="Comparado con el optimo exacto")
c2.metric("Peso", f"{r['peso']} / {capacidad}")
c3.metric("Objetos llevados", r["objetos"])
c4.metric("Generacion de la mejor", res.generacion_mejor)
c5.metric("Tiempo", f"{res.tiempo_seg:.3f} s")

# Barra de ocupacion de la mochila
st.subheader("Ocupacion de la mochila")
fig, ax = plt.subplots(figsize=(9, 1.3))
ax.barh([0], [capacidad], color="#dddddd", label="Capacidad")
ax.barh([0], [r["peso"]], color="#C44E52" if not r["valido"] else "#55A868",
        label="Peso llevado")
ax.set_xlim(0, max(capacidad, r["peso"]) * 1.05)
ax.set_yticks([])
ax.set_xlabel("Peso")
ax.legend(loc="upper right", ncol=2)
fig.tight_layout()
st.pyplot(fig)
st.caption(f"Holgura: {r['holgura']} unidades de peso sin usar. "
           f"Lambda usada por la penalizacion: "
           f"debil {lambda_penalizacion(pesos, valores, 'proporcional'):.2f} | "
           f"fuerte {lambda_penalizacion(pesos, valores, 'fuerte'):.0f}")

col_tabla, col_grafica = st.columns([1.2, 1])

with col_tabla:
    st.subheader("Objetos seleccionados")
    tabla = seleccion_dataframe(res.mejor_individuo, pesos, valores)
    st.dataframe(
        tabla.style.apply(
            lambda f: ["background-color: #d6f5d6" if f["seleccionado"] == "Si" else ""
                       for _ in f], axis=1),
        use_container_width=True, hide_index=True, height=420)

with col_grafica:
    st.subheader("Convergencia")
    st.pyplot(_comun.grafica_convergencia(
        res.historial_mejor, res.historial_promedio,
        ylabel="Aptitud (valor penalizado)",
        titulo="Aptitud por generacion"))

    st.markdown("**Cromosoma binario**")
    st.code("".join(str(int(g)) for g in res.mejor_individuo), language="text")
    st.caption(f"Estrategia: **{manejo}** | cruce **{cruce}** | "
               f"{r['objetos']} de {n_objetos} objetos seleccionados")

_comun.pie_de_pagina()
