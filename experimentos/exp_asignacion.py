"""
EXPERIMENTOS DE ASIGNACION DE CURSOS A SALAS (Ejercicio 2)

Responde las tres preguntas del enunciado con datos medidos:

    ESTUDIO A - Elitismo si / no y penalizar vs reparar (rejilla 2 x 2).
                Contesta "es mejor penalizar o reparar individuos invalidos?"
                y "compare una version con y sin elitismo".

    ESTUDIO B - Escalado: 8, 12, 16 y 20 cursos sobre las mismas 4 salas y
                5 franjas (20 slots). Contesta "como cambia el resultado al
                aumentar el numero de cursos?".

    ESTUDIO C - Cruce de un punto vs cruce uniforme.

Metricas por configuracion:
    factibles_%      - corridas que llegaron a penalizacion dura 0
    dura_promedio    - penalizacion dura promedio (0 es lo deseable)
    blanda_promedio  - calidad del horario una vez es factible
    gen_mejor        - generacion en que aparecio la mejor solucion
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from problemas import asignacion as A
from ga.estadisticas import (guardar_tabla, grafica_barras,
                             grafica_convergencia, ruta_resultado)

CORRIDAS = 30
GENERACIONES = 300
TAM_POBLACION = 100


def corridas(etiqueta, n_corridas=CORRIDAS, **kwargs):
    """Repite una configuracion y resume sus metricas."""
    duras, blandas, totales, generaciones, historiales = [], [], [], [], []
    factibles = 0
    mejor_resultado, mejor_costo = None, float("inf")

    for semilla in range(n_corridas):
        res = A.resolver(tam_poblacion=TAM_POBLACION, generaciones=GENERACIONES,
                         semilla=semilla, **kwargs)
        d = A.detalle_penalizaciones(res.mejor_individuo)

        duras.append(d["costo_duro"])
        blandas.append(d["costo_blando"])
        totales.append(d["costo_total"])
        generaciones.append(res.generacion_mejor)
        factibles += int(d["factible"])

        h = [-v for v in res.historial_mejor]
        h += [h[-1]] * (GENERACIONES - len(h))
        historiales.append(h)

        if d["costo_total"] < mejor_costo:
            mejor_costo, mejor_resultado = d["costo_total"], res

    fila = {
        "configuracion": etiqueta,
        "corridas": n_corridas,
        "factibles_%": round(100 * factibles / n_corridas, 1),
        "dura_promedio": round(float(np.mean(duras)), 1),
        "blanda_promedio": round(float(np.mean(blandas)), 1),
        "total_mejor": int(np.min(totales)),
        "total_peor": int(np.max(totales)),
        "total_promedio": round(float(np.mean(totales)), 1),
        "gen_mejor_promedio": round(float(np.mean(generaciones)), 1),
    }
    return fila, np.mean(historiales, axis=0), mejor_resultado


# ==========================================================================
# ESTUDIO A: elitismo x estrategia de manejo de invalidos
# ==========================================================================


def estudio_elitismo_y_estrategia(sufijo="",
                                  archivo_conv="asignacion_elitismo_estrategia.png",
                                  archivo_barras="asignacion_factibilidad.png"):
    filas, series, etiquetas = [], [], []
    mejor_global, mejor_costo = None, float("inf")

    for estrategia in ("penalizar", "reparar"):
        for elitismo, nombre_e in ((0, "sin elitismo"), (2, "con elitismo")):
            etiqueta = f"{estrategia} / {nombre_e}"
            fila, historial, mejor = corridas(etiqueta, estrategia=estrategia,
                                              elitismo=elitismo, tasa_mutacion=0.1,
                                              cruce="uniforme")
            filas.append(fila)
            series.append(historial)
            etiquetas.append(etiqueta)
            print(f"  {etiqueta:<28} factibles={fila['factibles_%']:>5}%  "
                  f"dura={fila['dura_promedio']:>7}  blanda={fila['blanda_promedio']:>6}  "
                  f"gen={fila['gen_mejor_promedio']}")

            costo = A.detalle_penalizaciones(mejor.mejor_individuo)["costo_total"]
            if costo < mejor_costo:
                mejor_costo, mejor_global = costo, mejor

    df = pd.DataFrame(filas)
    grafica_convergencia(
        series, etiquetas,
        titulo=f"Asignacion de cursos{sufijo}: elitismo y manejo de invalidos "
               f"(promedio de {CORRIDAS} corridas)",
        ylabel="Penalizacion total del mejor individuo",
        archivo=archivo_conv)
    grafica_barras(df["configuracion"].tolist(), df["factibles_%"].tolist(),
                   titulo=f"Asignacion de cursos{sufijo}: porcentaje de corridas "
                          f"que alcanzan un horario factible",
                   ylabel="Corridas factibles (%)",
                   archivo=archivo_barras)
    return df, mejor_global


# ==========================================================================
# ESTUDIO B: que pasa al aumentar el numero de cursos
# ==========================================================================


def estudio_escalado_cursos(tamanos=(8, 12, 16, 20)):
    """Mismas 4 salas y 5 franjas (20 slots), cada vez mas cursos.

    Las instancias se generan garantizando que existe al menos un horario
    factible, de modo que cualquier fallo sea atribuible al algoritmo.
    """
    filas = []
    for n_cursos in tamanos:
        cursos, salas, franjas = A.instancia_sintetica(n_cursos)
        A.configurar_instancia(cursos, salas, franjas)
        ocupacion = 100 * n_cursos / A.N_SLOTS

        for estrategia in ("penalizar", "reparar"):
            fila, _, _ = corridas(f"{n_cursos} cursos / {estrategia}",
                                  estrategia=estrategia, elitismo=2,
                                  tasa_mutacion=0.1, cruce="uniforme")
            fila["n_cursos"] = n_cursos
            fila["estrategia"] = estrategia
            fila["ocupacion_slots_%"] = round(ocupacion, 1)
            filas.append(fila)
            print(f"  {n_cursos:>2} cursos ({ocupacion:>5.1f}% de los slots) / "
                  f"{estrategia:<10} -> factibles={fila['factibles_%']:>5}%  "
                  f"dura={fila['dura_promedio']:>8}")

    A.restaurar_instancia_base()
    df = pd.DataFrame(filas)
    grafica_barras(df["configuracion"].tolist(), df["factibles_%"].tolist(),
                   titulo="Asignacion: efecto de aumentar el numero de cursos",
                   ylabel="Corridas factibles (%)",
                   archivo="asignacion_escalado.png")
    return df


# ==========================================================================
# ESTUDIO C: operador de cruce
# ==========================================================================


def estudio_cruces(estrategia="penalizar"):
    filas = []
    for cruce in ("un_punto", "uniforme"):
        fila, _, _ = corridas(f"cruce {cruce}", estrategia=estrategia,
                              elitismo=2, tasa_mutacion=0.1, cruce=cruce)
        filas.append(fila)
        print(f"  cruce {cruce:<10} -> factibles={fila['factibles_%']:>5}%  "
              f"total_promedio={fila['total_promedio']}")
    return pd.DataFrame(filas)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 42)

    A.guardar_tablas()
    print("Tablas de cursos y salas guardadas en datos/\n")

    print("ESTUDIO A: elitismo si/no x penalizar vs reparar")
    df_a, mejor = estudio_elitismo_y_estrategia()
    print(f"  -> {guardar_tabla(df_a, 'asignacion_elitismo_estrategia.csv')}\n")

    print("ESTUDIO B: aumento del numero de cursos (4 salas x 5 franjas = 20 slots)")
    df_b = estudio_escalado_cursos()
    print(f"  -> {guardar_tabla(df_b, 'asignacion_escalado.csv')}\n")

    print("ESTUDIO C: cruce de un punto vs cruce uniforme (instancia base)")
    df_c = estudio_cruces()
    print(f"  -> {guardar_tabla(df_c, 'asignacion_cruces.csv')}\n")

    # --- Los estudios A y C se repiten sobre la instancia APRETADA -----------
    # Con 20 cursos y 20 slots no sobra ni un espacio. Es el unico escenario en
    # el que las diferencias entre configuraciones se vuelven medibles: en la
    # instancia base (8 cursos, 40% de ocupacion) todas empatan al 100%.
    print("ESTUDIO D: misma rejilla sobre la instancia apretada (20 cursos / 20 slots)")
    A.configurar_instancia(*A.instancia_sintetica(20))
    df_d, _ = estudio_elitismo_y_estrategia(
        sufijo=" (20 cursos / 20 slots)",
        archivo_conv="asignacion_elitismo_estrategia_apretada.png",
        archivo_barras="asignacion_factibilidad_apretada.png")
    print(f"  -> {guardar_tabla(df_d, 'asignacion_elitismo_estrategia_apretada.csv')}\n")

    print("ESTUDIO E: cruces sobre la instancia apretada")
    df_e = estudio_cruces()
    print(f"  -> {guardar_tabla(df_e, 'asignacion_cruces_apretada.csv')}\n")
    A.restaurar_instancia_base()

    # --- Mejor horario encontrado en todo el experimento ---
    detalle = A.detalle_penalizaciones(mejor.mejor_individuo)
    print("MEJOR HORARIO ENCONTRADO")
    print(f"  Penalizacion dura {detalle['costo_duro']} / blanda {detalle['costo_blando']}"
          f" / total {detalle['costo_total']}")
    print(f"  Violaciones: {detalle['violaciones']}\n")

    horario = A.horario_dataframe(mejor.mejor_individuo)
    print(horario.to_string())
    guardar_tabla(horario.reset_index().rename(columns={"index": "sala"}),
                  "asignacion_horario_final.csv")
    guardar_tabla(A.asignacion_dataframe(mejor.mejor_individuo),
                  "asignacion_detalle_final.csv")

    print("\nExperimentos de asignacion completados.")
