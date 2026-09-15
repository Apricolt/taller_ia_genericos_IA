"""
EXPERIMENTOS DE LA MOCHILA (Ejercicio 3)

Cubre la actividad del enunciado:
    1. Instancia de 15 objetos con pesos y valores distintos.
    2. Dos capacidades de mochila (50% y 25% del peso total).
    3. Comparacion de PENALIZACION frente a REPARACION.
    4. Varias corridas registrando valor, peso y generacion de la mejor solucion.
    5. Referencia: optimo exacto por programacion dinamica (brecha real del GA).

NOTA METODOLOGICA
    Con 15 objetos el espacio tiene solo 2^15 = 32.768 combinaciones y el
    algoritmo alcanza el optimo exacto practicamente siempre, con cualquiera de
    las dos estrategias. Ese resultado se reporta (ESTUDIO A), pero para poder
    COMPARAR penalizacion y reparacion hacen falta instancias mas grandes:
    el ESTUDIO D escala a 30, 50 y 100 objetos manteniendo la referencia exacta
    de la programacion dinamica.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from problemas.mochila import (crear_objetos, capacidades, guardar_tabla_objetos,
                               resolver, optimo_exacto, resumen_solucion,
                               peso_total, seleccion_dataframe)
from ga.estadisticas import (guardar_tabla, grafica_convergencia,
                             grafica_barras, resumen_valores)

CORRIDAS = 30
BASE = {"tam_poblacion": 60, "generaciones": 120}
AJUSTADO = {"tam_poblacion": 30, "generaciones": 40}

# Las tres maneras de tratar a los individuos que no caben en la mochila.
# La "debil" usa lambda = max(valor/peso) y la "fuerte" lambda = suma de valores.
ESTRATEGIAS = [
    ("penalizacion debil", {"estrategia": "penalizar",
                            "modo_penalizacion": "proporcional"}),
    ("penalizacion fuerte", {"estrategia": "penalizar",
                             "modo_penalizacion": "fuerte"}),
    ("reparacion", {"estrategia": "reparar"}),
]


def corridas(pesos, valores, capacidad, etiqueta, presupuesto=BASE,
             n_corridas=CORRIDAS, valor_optimo=None, **kwargs):
    """Repite una configuracion y resume valor, peso y validez de la mejor solucion."""
    valores_finales, pesos_finales, generaciones = [], [], []
    historiales, validez_poblacion = [], []
    n_gen = presupuesto["generaciones"]
    optimos_alcanzados, mejores_invalidas = 0, 0

    for semilla in range(n_corridas):
        res = resolver(pesos, valores, capacidad, semilla=semilla,
                       **presupuesto, **kwargs)
        r = resumen_solucion(res.mejor_individuo, pesos, valores, capacidad)

        # Con reparacion y con penalizacion fuerte la mejor solucion es siempre
        # valida. Con penalizacion debil NO: se mide cuantas veces falla, que es
        # justamente el argumento a favor de reparar.
        mejores_invalidas += int(not r["valido"])

        valores_finales.append(r["valor"])
        pesos_finales.append(r["peso"])
        generaciones.append(res.generacion_mejor)
        if valor_optimo is not None and r["valor"] == valor_optimo:
            optimos_alcanzados += 1

        # Fraccion de la poblacion final que respeta la capacidad
        validos = sum(peso_total(ind, pesos) <= capacidad
                      for ind in res.poblacion_final)
        validez_poblacion.append(100 * validos / len(res.poblacion_final))

        h = list(res.historial_mejor)
        h += [h[-1]] * (n_gen - len(h))
        historiales.append(h)

    r = resumen_valores(valores_finales)
    fila = {
        "configuracion": etiqueta,
        "corridas": n_corridas,
        "valor_mejor": int(r["mejor"]),
        "valor_peor": int(r["peor"]),
        "valor_promedio": round(r["promedio"], 1),
        "desviacion": round(r["desviacion"], 2),
        "peso_mejor": int(pesos_finales[int(np.argmax(valores_finales))]),
        "capacidad": capacidad,
        "gen_mejor_promedio": round(float(np.mean(generaciones)), 1),
        "poblacion_valida_%": round(float(np.mean(validez_poblacion)), 1),
        "mejor_invalida_%": round(100 * mejores_invalidas / n_corridas, 1),
    }
    if valor_optimo is not None:
        fila["optimo_dp"] = valor_optimo
        fila["corridas_en_optimo_%"] = round(100 * optimos_alcanzados / n_corridas, 1)
        fila["brecha_promedio_%"] = round(
            100 * (valor_optimo - float(np.mean(valores_finales))) / valor_optimo, 3)

    return fila, np.mean(historiales, axis=0)


# ==========================================================================
# ESTUDIO A: instancia del enunciado, dos capacidades, penalizar vs reparar
# ==========================================================================


def estudio_capacidades(pesos, valores):
    filas, series, etiquetas = [], [], []
    for nombre, capacidad in capacidades(pesos).items():
        _, valor_optimo = optimo_exacto(pesos, valores, capacidad)
        for estrategia, opciones in ESTRATEGIAS:
            etiqueta = f"cap {capacidad} ({nombre}) / {estrategia}"
            fila, historial = corridas(pesos, valores, capacidad, etiqueta,
                                       valor_optimo=valor_optimo,
                                       tasa_mutacion=0.05, elitismo=2,
                                       cruce="un_punto", **opciones)
            fila["estrategia"] = estrategia
            filas.append(fila)
            series.append(historial)
            etiquetas.append(etiqueta)
            print(f"  {etiqueta:<48} valor={fila['valor_promedio']:>7}  "
                  f"optimo={valor_optimo:>5}  en_optimo={fila['corridas_en_optimo_%']:>5}%  "
                  f"pob_valida={fila['poblacion_valida_%']:>5}%  "
                  f"mejor_invalida={fila['mejor_invalida_%']:>5}%")

    grafica_convergencia(
        series, etiquetas,
        titulo=f"Mochila (15 objetos): convergencia por capacidad y estrategia "
               f"(promedio de {CORRIDAS} corridas)",
        ylabel="Aptitud del mejor individuo",
        archivo="mochila_capacidades.png")
    return pd.DataFrame(filas)


# ==========================================================================
# ESTUDIO B: efecto de la tasa de mutacion
# ==========================================================================


def estudio_tasas_mutacion(pesos, valores, tasas=(0.01, 0.05, 0.1, 0.3)):
    capacidad = list(capacidades(pesos).values())[0]
    _, valor_optimo = optimo_exacto(pesos, valores, capacidad)
    filas, series, etiquetas = [], [], []

    for tasa in tasas:
        fila, historial = corridas(pesos, valores, capacidad, f"mutacion = {tasa}",
                                   presupuesto=AJUSTADO, valor_optimo=valor_optimo,
                                   estrategia="penalizar", modo_penalizacion="fuerte",
                                   tasa_mutacion=tasa,
                                   elitismo=2, cruce="un_punto")
        filas.append(fila)
        series.append(historial)
        etiquetas.append(f"mutacion = {tasa}")
        print(f"  tasa {tasa:<5} -> valor_promedio={fila['valor_promedio']:>7}  "
              f"en_optimo={fila['corridas_en_optimo_%']:>5}%  "
              f"brecha={fila['brecha_promedio_%']:>6}%")

    grafica_convergencia(
        series, etiquetas,
        titulo="Mochila: efecto de la tasa de mutacion por gen "
               "(presupuesto ajustado)",
        ylabel="Aptitud del mejor individuo",
        archivo="mochila_tasas_mutacion.png")
    return pd.DataFrame(filas)


# ==========================================================================
# ESTUDIO C: elitismo y operador de cruce
# ==========================================================================


def estudio_elitismo_cruce(pesos, valores):
    capacidad = list(capacidades(pesos).values())[0]
    _, valor_optimo = optimo_exacto(pesos, valores, capacidad)
    filas = []

    for cruce in ("un_punto", "uniforme"):
        for elitismo, nombre in ((0, "sin elitismo"), (2, "con elitismo")):
            etiqueta = f"{cruce} / {nombre}"
            fila, _ = corridas(pesos, valores, capacidad, etiqueta,
                               presupuesto=AJUSTADO, valor_optimo=valor_optimo,
                               estrategia="penalizar", modo_penalizacion="fuerte",
                               tasa_mutacion=0.05, elitismo=elitismo, cruce=cruce)
            filas.append(fila)
            print(f"  {etiqueta:<26} valor_promedio={fila['valor_promedio']:>7}  "
                  f"en_optimo={fila['corridas_en_optimo_%']:>5}%")

    df = pd.DataFrame(filas)
    grafica_barras(df["configuracion"].tolist(), df["corridas_en_optimo_%"].tolist(),
                   titulo="Mochila: elitismo y operador de cruce "
                          "(presupuesto ajustado)",
                   ylabel="Corridas que alcanzan el optimo (%)",
                   archivo="mochila_elitismo_cruce.png")
    return df


# ==========================================================================
# ESTUDIO D: escalado del numero de objetos (penalizar vs reparar de verdad)
# ==========================================================================


def estudio_escalado(tamanos=(15, 30, 50, 100)):
    """Instancias cada vez mas grandes con el mismo presupuesto de computo.

    Es el escenario donde penalizacion y reparacion dejan de empatar.
    """
    filas = []
    for n in tamanos:
        pesos, valores = crear_objetos(n)
        capacidad = int(round(pesos.sum() * 0.5))
        _, valor_optimo = optimo_exacto(pesos, valores, capacidad)

        for estrategia, opciones in ESTRATEGIAS:
            etiqueta = f"{n} objetos / {estrategia}"
            fila, _ = corridas(pesos, valores, capacidad, etiqueta,
                               presupuesto=AJUSTADO, valor_optimo=valor_optimo,
                               tasa_mutacion=0.05, elitismo=2,
                               cruce="un_punto", **opciones)
            fila["n_objetos"] = n
            fila["estrategia"] = estrategia
            filas.append(fila)
            print(f"  {etiqueta:<36} brecha={fila['brecha_promedio_%']:>7}%  "
                  f"en_optimo={fila['corridas_en_optimo_%']:>5}%  "
                  f"pob_valida={fila['poblacion_valida_%']:>5}%  "
                  f"mejor_invalida={fila['mejor_invalida_%']:>5}%")

    df = pd.DataFrame(filas)
    grafica_barras(df["configuracion"].tolist(), df["brecha_promedio_%"].tolist(),
                   titulo="Mochila: brecha frente al optimo exacto al crecer la instancia",
                   ylabel="Brecha promedio respecto al optimo (%)",
                   archivo="mochila_escalado.png")
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 220)

    pesos, valores = crear_objetos()
    guardar_tabla_objetos(pesos, valores)
    print(f"Instancia base: {len(pesos)} objetos, peso total {int(pesos.sum())}, "
          f"valor total {int(valores.sum())}\n")

    print("ESTUDIO A: dos capacidades x penalizar vs reparar (15 objetos)")
    df_a = estudio_capacidades(pesos, valores)
    print(f"  -> {guardar_tabla(df_a, 'mochila_capacidades.csv')}\n")

    print("ESTUDIO B: efecto de la tasa de mutacion (presupuesto ajustado)")
    df_b = estudio_tasas_mutacion(pesos, valores)
    print(f"  -> {guardar_tabla(df_b, 'mochila_tasas_mutacion.csv')}\n")

    print("ESTUDIO C: elitismo y operador de cruce (presupuesto ajustado)")
    df_c = estudio_elitismo_cruce(pesos, valores)
    print(f"  -> {guardar_tabla(df_c, 'mochila_elitismo_cruce.csv')}\n")

    print("ESTUDIO D: escalado del numero de objetos")
    df_d = estudio_escalado()
    print(f"  -> {guardar_tabla(df_d, 'mochila_escalado.csv')}\n")

    # --- Mejor solucion de la instancia del enunciado, objeto por objeto ---
    capacidad = list(capacidades(pesos).values())[0]
    seleccion_optima, valor_optimo = optimo_exacto(pesos, valores, capacidad)
    res = resolver(pesos, valores, capacidad, estrategia="reparar", semilla=42, **BASE)
    r = resumen_solucion(res.mejor_individuo, pesos, valores, capacidad)

    print(f"SOLUCION FINAL (capacidad {capacidad})")
    print(f"  Algoritmo genetico: valor {r['valor']}, peso {r['peso']}/{capacidad}, "
          f"{r['objetos']} objetos, encontrada en la generacion {res.generacion_mejor}")
    print(f"  Optimo exacto (DP): valor {valor_optimo}")
    detalle = seleccion_dataframe(res.mejor_individuo, pesos, valores)
    print(detalle.to_string(index=False))
    guardar_tabla(detalle, "mochila_seleccion_final.csv")

    print("\nExperimentos de la mochila completados.")
