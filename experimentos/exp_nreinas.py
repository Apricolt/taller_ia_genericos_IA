"""
EXPERIMENTOS N-REINAS

Barrido que pide el enunciado: N = 6 y 8, distintos tamanos de poblacion y
tasas de mutacion 0.05, 0.1 y 0.2. Cada configuracion se repite CORRIDAS veces
con semillas distintas porque el algoritmo es estocastico.

Metricas registradas por configuracion:
    tasa_exito        - porcentaje de corridas que llegaron a 0 conflictos
    gens_promedio     - generaciones necesarias (solo corridas exitosas)
    conflictos_prom   - conflictos finales promedio sobre todas las corridas
    tiempo_promedio   - segundos por corrida

Genera ademas un segundo estudio comparando OX vs PMX y elitismo vs sin
elitismo, que alimenta el analisis del informe.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from problemas.nreinas import resolver, conflictos
from ga.estadisticas import guardar_tabla, grafica_barras, grafica_convergencia

CORRIDAS = 30
CORRIDAS_OPERADORES = 50  # mas repeticiones: las diferencias aqui son pequenas
GENERACIONES = 300
VALORES_N = (6, 8)
TAMANOS_POBLACION = (20, 50, 100)
TASAS_MUTACION = (0.05, 0.1, 0.2)


def correr_configuracion(n, tam_poblacion, tasa_mutacion, elitismo=2, cruce="OX",
                         corridas=CORRIDAS):
    """Repite una configuracion 'corridas' veces y devuelve las metricas."""
    exitos, generaciones, conflictos_finales, tiempos = 0, [], [], []

    for semilla in range(corridas):
        res = resolver(n, tam_poblacion=tam_poblacion, generaciones=GENERACIONES,
                       tasa_mutacion=tasa_mutacion, elitismo=elitismo,
                       cruce=cruce, semilla=semilla)
        conflictos_finales.append(conflictos(res.mejor_individuo))
        tiempos.append(res.tiempo_seg)
        if res.exito:
            exitos += 1
            generaciones.append(res.generacion_mejor)

    return {
        "N": n,
        "poblacion": tam_poblacion,
        "tasa_mutacion": tasa_mutacion,
        "elitismo": elitismo,
        "cruce": cruce,
        "corridas": corridas,
        "tasa_exito_%": round(100 * exitos / corridas, 1),
        "gens_promedio": round(float(np.mean(generaciones)), 1) if generaciones else None,
        "gens_minimo": int(np.min(generaciones)) if generaciones else None,
        "gens_maximo": int(np.max(generaciones)) if generaciones else None,
        "conflictos_prom": round(float(np.mean(conflictos_finales)), 3),
        "tiempo_promedio_s": round(float(np.mean(tiempos)), 4),
    }


def barrido_principal():
    """N x poblacion x tasa de mutacion."""
    filas = []
    for n in VALORES_N:
        for poblacion in TAMANOS_POBLACION:
            for tasa in TASAS_MUTACION:
                fila = correr_configuracion(n, poblacion, tasa)
                filas.append(fila)
                print(f"  N={n} pob={poblacion:>3} mut={tasa:<5} "
                      f"exito={fila['tasa_exito_%']:>5}%  gens={fila['gens_promedio']}")
    return pd.DataFrame(filas)


def estudio_operadores():
    """OX vs PMX y elitismo vs sin elitismo, con poblacion y mutacion fijas."""
    filas = []
    for n in VALORES_N:
        for cruce in ("OX", "PMX"):
            for elitismo in (0, 2):
                fila = correr_configuracion(n, tam_poblacion=20, tasa_mutacion=0.1,
                                            elitismo=elitismo, cruce=cruce,
                                            corridas=CORRIDAS_OPERADORES)
                fila["configuracion"] = f"N={n} {cruce} " + ("con elitismo" if elitismo else "sin elitismo")
                filas.append(fila)
                print(f"  {fila['configuracion']:<28} exito={fila['tasa_exito_%']:>5}%  "
                      f"gens={fila['gens_promedio']}")
    return pd.DataFrame(filas)


def curvas_por_tasa_mutacion(n=8, tam_poblacion=20):
    """Curva de conflictos del mejor individuo para cada tasa de mutacion.

    Se promedia el historial de varias corridas para que la curva no dependa de
    una semilla afortunada.
    """
    series, etiquetas = [], []
    for tasa in TASAS_MUTACION:
        historiales = []
        for semilla in range(CORRIDAS):
            res = resolver(n, tam_poblacion=tam_poblacion, generaciones=GENERACIONES,
                           tasa_mutacion=tasa, elitismo=2, semilla=semilla)
            # Se rellena con el valor final para poder promediar corridas de
            # distinta duracion (las exitosas terminan antes).
            h = [-v for v in res.historial_mejor]
            h += [h[-1]] * (GENERACIONES - len(h))
            historiales.append(h)
        series.append(np.mean(historiales, axis=0))
        etiquetas.append(f"mutacion = {tasa}")

    return grafica_convergencia(
        series, etiquetas,
        titulo=f"N-Reinas (N={n}, poblacion={tam_poblacion}): efecto de la tasa de mutacion",
        ylabel="Conflictos del mejor individuo (promedio de 30 corridas)",
        archivo="nreinas_tasas_mutacion.png",
    )


if __name__ == "__main__":
    print("BARRIDO PRINCIPAL: N x poblacion x tasa de mutacion")
    df = barrido_principal()
    ruta = guardar_tabla(df, "nreinas_experimentos.csv")
    print(f"\nTabla guardada en {ruta}\n")

    print("ESTUDIO DE OPERADORES: OX vs PMX, con y sin elitismo")
    df_ops = estudio_operadores()
    ruta_ops = guardar_tabla(df_ops, "nreinas_operadores.csv")
    print(f"\nTabla guardada en {ruta_ops}\n")

    grafica_barras(
        df_ops["configuracion"].tolist(), df_ops["tasa_exito_%"].tolist(),
        titulo="N-Reinas: efecto del operador de cruce y del elitismo",
        ylabel="Tasa de exito (%)", archivo="nreinas_operadores.png")

    print("Generando curvas de convergencia por tasa de mutacion...")
    print(curvas_por_tasa_mutacion())
    print("\nExperimentos de N-Reinas completados.")
