"""Interfaz del Ejercicio 2: asignacion de cursos a salas de computo."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import _comun
import pandas as pd
import streamlit as st

from problemas import asignacion as A

st.set_page_config(page_title="Asignacion de cursos", page_icon="Horario",
                   layout="wide")

A.restaurar_instancia_base()

_comun.encabezado(
    "Ejercicio 2 - Asignacion de cursos a salas de computo",
    "Programar 8 cursos en 4 salas y 5 franjas horarias minimizando conflictos "
    "y penalizaciones",
    "Vector de 8 genes enteros, uno por curso. Cada gen es un **slot** "
    "`sala x 5 + franja` (0 a 19). No es una permutacion: dos cursos pueden "
    "caer en el mismo slot, y eso es justamente un choque de horario.",
    "Suma de penalizaciones, **a minimizar**. Las restricciones **duras** pesan "
    "1000 (choque, sobrecupo, falta de recursos, franja bloqueada) y las "
    "**blandas** entre 1 y 10 (desbalance de franjas, sillas desperdiciadas).",
)

# ------------------------------- Controles --------------------------------
st.sidebar.header("Asignacion de cursos")
estrategia = st.sidebar.radio(
    "Manejo de individuos invalidos", ["penalizar", "reparar"],
    help="Penalizar: entran a la poblacion y la aptitud los castiga. "
         "Reparar: se corrigen a un slot factible antes de evaluarlos.")
cruce = st.sidebar.selectbox("Operador de cruce", ["uniforme", "un_punto"])

st.sidebar.subheader("Pesos de las penalizaciones")
peso_duro = st.sidebar.slider("Peso de las restricciones duras", 100, 5000, 1000,
                              step=100)
peso_desbalance = st.sidebar.slider("Peso del desbalance entre franjas", 0, 50, 10)
peso_holgura = st.sidebar.slider("Peso de las sillas desperdiciadas", 0, 20, 1)

params = _comun.controles_geneticos(
    tam_poblacion=100, generaciones=300, tasa_mutacion=0.1,
    ayuda_mutacion="Probabilidad de reasignar CADA gen (curso) a otro slot")

# Los pesos se aplican sobre el modulo antes de ejecutar
for tipo in A.TIPOS_DUROS:
    A.PESOS[tipo] = peso_duro
A.PESOS["desbalance"] = peso_desbalance
A.PESOS["holgura"] = peso_holgura

# ------------------------------- Datos de entrada -------------------------
tab_cursos, tab_salas = st.tabs(["Tabla de cursos", "Tabla de salas"])
df_cursos, df_salas = A.tablas_dataframes()  # sin escribir en disco
tab_cursos.dataframe(df_cursos, use_container_width=True, hide_index=True)
tab_salas.dataframe(df_salas, use_container_width=True, hide_index=True)

# ------------------------------- Ejecucion --------------------------------
if st.sidebar.button("Ejecutar algoritmo genetico", type="primary",
                     use_container_width=True):
    with st.spinner("Buscando un horario factible..."):
        res = A.resolver(estrategia=estrategia, cruce=cruce, **params)
    st.session_state["asignacion"] = (res, estrategia, cruce)

if "asignacion" not in st.session_state:
    st.info("Ajusta los parametros en la barra lateral y pulsa "
            "**Ejecutar algoritmo genetico**.")
    _comun.pie_de_pagina()
    st.stop()

res, estrategia, cruce = st.session_state["asignacion"]
detalle = A.detalle_penalizaciones(res.mejor_individuo)

# ------------------------------- Resultados -------------------------------
st.divider()
if detalle["factible"]:
    st.success("**Horario factible**: ninguna restriccion dura fue violada "
               "(sin choques, sin sobrecupo, con los recursos requeridos y "
               "respetando las franjas bloqueadas).")
else:
    st.error(f"**Horario NO factible**: penalizacion dura de "
             f"{detalle['costo_duro']}. Revisa las violaciones mas abajo.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Penalizacion dura", detalle["costo_duro"],
          help="Debe ser 0 para que el horario sea utilizable")
c2.metric("Penalizacion blanda", round(detalle["costo_blando"], 1))
c3.metric("Penalizacion total", round(detalle["costo_total"], 1))
c4.metric("Generacion de la mejor", res.generacion_mejor)

st.subheader("Horario final")
st.caption("Cada celda muestra los cursos programados en esa sala y franja. "
           "Dos cursos en la misma celda son un choque de horario.")
st.dataframe(A.horario_dataframe(res.mejor_individuo), use_container_width=True)

col_detalle, col_grafica = st.columns([1.3, 1])

with col_detalle:
    st.subheader("Verificacion curso a curso")
    tabla = A.asignacion_dataframe(res.mejor_individuo)

    def resaltar(fila):
        malo = (fila["sobrecupo"] == "Si" or fila["recursos_ok"] == "No" or
                fila["franja_permitida"] == "No" or fila["comparte_slot"] == "Si")
        return ["background-color: #ffd6d6" if malo else "" for _ in fila]

    st.dataframe(tabla.style.apply(resaltar, axis=1),
                 use_container_width=True, hide_index=True)

with col_grafica:
    st.subheader("Convergencia")
    st.pyplot(_comun.grafica_convergencia(
        res.historial_mejor, res.historial_promedio,
        ylabel="Penalizacion total",
        transformar=lambda v: -v,
        titulo="Penalizacion por generacion"))

    st.subheader("Violaciones por tipo")
    st.dataframe(pd.DataFrame([
        {"restriccion": tipo,
         "tipo": "dura" if tipo in A.TIPOS_DUROS else "blanda",
         "violaciones": valor,
         "costo": round(valor * A.PESOS[tipo], 1)}
        for tipo, valor in detalle["violaciones"].items()
    ]), use_container_width=True, hide_index=True)

with st.expander("Cromosoma y su decodificacion"):
    st.code(str([int(g) for g in res.mejor_individuo]), language="python")
    st.dataframe(pd.DataFrame([
        {"curso": A.CURSOS[i]["id"], "slot": int(slot),
         "sala": A.SALAS[A.decodificar(int(slot))[0]]["nombre"],
         "franja": A.FRANJAS[A.decodificar(int(slot))[1]]}
        for i, slot in enumerate(res.mejor_individuo)
    ]), use_container_width=True, hide_index=True)

_comun.pie_de_pagina()
