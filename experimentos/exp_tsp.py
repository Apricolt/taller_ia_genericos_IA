"""
EXPERIMENTOS TSP (Ejercicio 1)

Cubre el flujo que pide el enunciado:
    1. Matriz de distancias para 8, 10 y 15 ciudades.
    2. Poblacion inicial de rutas validas (permutaciones aleatorias).
    3. Seleccion, cruzamiento, mutacion y elitismo.
    4. Mejor ruta encontrada y su distancia.
    5. Al menos 10 ejecuciones: mejor, peor y promedio.
    6. Comparacion de dos tasas de mutacion.

NOTA METODOLOGICA IMPORTANTE
    Con un presupuesto amplio (poblacion 100, 300 generaciones) el algoritmo
    alcanza la misma ruta en las 10 corridas para 8, 10 y 15 ciudades: la
    desviacion es cero y todas las configuraciones empatan. Ese resultado se
    reporta tal cual (ESTUDIO A), pero no permite comparar operadores.

    Por eso las comparaciones (estudios B, C y D) se hacen con un PRESUPUESTO
    AJUSTADO (poblacion 30, 60 generaciones) sobre la instancia de 15 ciudades:
    al limitar los recursos, las diferencias entre operadores si se vuelven
    medibles. Y el ESTUDIO E amplia la instancia a 30 y 50 ciudades para
    responder la pregunta sobre que ocurre al aumentar el numero de ciudades.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from problemas.tsp import (crear_instancia, resolver, distancia_ruta,
                           ruta_vecino_mas_cercano, optimo_exacto,
                           aptitud_a_distancia, dibujar_ruta)
from ga.estadisticas import (guardar_tabla, grafica_convergencia,
                             grafica_barras, resumen_costos, ruta_resultado)
from ga.operadores import es_permutacion_valida

CORRIDAS = 10                 # minimo exigido por el enunciado
TAMANOS = (8, 10, 15)

# Presupuesto amplio: para demostrar que el GA resuelve las instancias pedidas
AMPLIO = {"tam_poblacion": 100, "generaciones": 300}
# Presupuesto ajustado: para que las comparaciones entre operadores sean visibles
AJUSTADO = {"tam_poblacion": 30, "generaciones": 60}

INSTANCIA_PRINCIPAL = 15


def corridas(matriz, presupuesto, n_corridas=CORRIDAS, **kwargs):
    """Repite una configuracion n_corridas veces con semillas 0..n-1.

    Devuelve las distancias finales, los historiales de convergencia (ya
    convertidos a distancia) y el mejor ResultadoGA obtenido.
    """
    distancias, historiales, generaciones_mejor = [], [], []
    mejor_resultado, mejor_distancia = None, float("inf")
    n_gen = presupuesto["generaciones"]

    for semilla in range(n_corridas):
        res = resolver(matriz, semilla=semilla, **presupuesto, **kwargs)
        assert es_permutacion_valida(res.mejor_individuo, len(matriz)), "ruta invalida"

        d = distancia_ruta(res.mejor_individuo, matriz)
        distancias.append(d)
        generaciones_mejor.append(res.generacion_mejor)

        # Se rellena el historial hasta n_gen para poder promediar corridas
        # que hayan terminado antes por estancamiento.
        h = [aptitud_a_distancia(v) for v in res.historial_mejor]
        h += [h[-1]] * (n_gen - len(h))
        historiales.append(h)

        if d < mejor_distancia:
            mejor_distancia, mejor_resultado = d, res

    return distancias, historiales, generaciones_mejor, mejor_resultado


def fila_resumen(etiqueta, distancias, generaciones=None, extra=None):
    """Convierte una lista de distancias en una fila de tabla del informe."""
    r = resumen_costos(distancias)
    fila = {
        "configuracion": etiqueta,
        "corridas": r["n_corridas"],
        "mejor": round(r["mejor"], 2),
        "peor": round(r["peor"], 2),
        "promedio": round(r["promedio"], 2),
        "desviacion": round(r["desviacion"], 2),
    }
    if generaciones is not None:
        fila["gen_mejor_promedio"] = round(float(np.mean(generaciones)), 1)
    if extra:
        fila.update(extra)
    return fila


# ==========================================================================
# ESTUDIO A: instancias pedidas por el enunciado (presupuesto amplio)
# ==========================================================================


def estudio_escalado():
    filas = []
    for n in TAMANOS:
        coordenadas, matriz = crear_instancia(n)
        distancias, _, gens, mejor = corridas(
            matriz, AMPLIO, tasa_mutacion=0.2, elitismo=2,
            cruce="OX", mutacion="inversion")

        voraz = distancia_ruta(ruta_vecino_mas_cercano(matriz), matriz)
        extra = {"vecino_mas_cercano": round(voraz, 2)}

        if n <= 10:  # el optimo exacto solo es calculable en instancias pequenas
            _, dist_optima = optimo_exacto(matriz)
            extra["optimo_exacto"] = round(dist_optima, 2)
            extra["brecha_promedio_%"] = round(
                100 * (float(np.mean(distancias)) - dist_optima) / dist_optima, 2)

        filas.append(fila_resumen(f"{n} ciudades", distancias, gens, extra))
        print(f"  {n:>2} ciudades -> mejor={min(distancias):8.2f}  "
              f"peor={max(distancias):8.2f}  promedio={np.mean(distancias):8.2f}  "
              f"desv={np.std(distancias):5.2f}")

        dibujar_ruta(coordenadas, mejor.mejor_individuo,
                     titulo=f"TSP {n} ciudades - mejor ruta de {CORRIDAS} corridas "
                            f"(distancia {distancia_ruta(mejor.mejor_individuo, matriz):.2f})",
                     archivo=ruta_resultado(f"tsp_mejor_ruta_{n}.png"))
    return pd.DataFrame(filas)


# ==========================================================================
# ESTUDIO B: comparacion de dos tasas de mutacion (lo pide el enunciado)
# ==========================================================================


def estudio_tasas_mutacion(n=INSTANCIA_PRINCIPAL, tasas=(0.05, 0.2, 0.9)):
    _, matriz = crear_instancia(n)
    filas, series, etiquetas = [], [], []

    for tasa in tasas:
        distancias, historiales, gens, _ = corridas(
            matriz, AJUSTADO, tasa_mutacion=tasa, elitismo=2,
            cruce="OX", mutacion="inversion")
        filas.append(fila_resumen(f"mutacion = {tasa}", distancias, gens))
        series.append(np.mean(historiales, axis=0))
        etiquetas.append(f"mutacion = {tasa}")
        print(f"  tasa {tasa:<5} -> mejor={min(distancias):8.2f}  "
              f"peor={max(distancias):8.2f}  promedio={np.mean(distancias):8.2f}")

    grafica_convergencia(
        series, etiquetas,
        titulo=f"TSP {n} ciudades: efecto de la tasa de mutacion "
               f"(presupuesto ajustado, promedio de {CORRIDAS} corridas)",
        ylabel="Distancia de la mejor ruta",
        archivo="tsp_tasas_mutacion.png")
    return pd.DataFrame(filas)


# ==========================================================================
# ESTUDIO C: operadores de cruce y de mutacion
# ==========================================================================


def estudio_operadores(n=INSTANCIA_PRINCIPAL):
    _, matriz = crear_instancia(n)
    filas = []
    for cruce in ("OX", "PMX"):
        for mutacion in ("swap", "inversion", "insercion"):
            distancias, _, gens, _ = corridas(
                matriz, AJUSTADO, tasa_mutacion=0.2, elitismo=2,
                cruce=cruce, mutacion=mutacion)
            filas.append(fila_resumen(f"{cruce} + {mutacion}", distancias, gens,
                                      {"cruce": cruce, "mutacion": mutacion}))
            print(f"  {cruce:<4} + {mutacion:<10} -> mejor={min(distancias):8.2f}  "
                  f"promedio={np.mean(distancias):8.2f}  desv={np.std(distancias):5.2f}")

    df = pd.DataFrame(filas)
    grafica_barras(df["configuracion"].tolist(), df["promedio"].tolist(),
                   titulo=f"TSP {n} ciudades: comparacion de operadores "
                          f"(presupuesto ajustado)",
                   ylabel="Distancia promedio de 10 corridas",
                   archivo="tsp_operadores.png",
                   errores=df["desviacion"].tolist())
    return df


# ==========================================================================
# ESTUDIO D: elitismo si / no
# ==========================================================================


def estudio_elitismo(n=INSTANCIA_PRINCIPAL):
    _, matriz = crear_instancia(n)
    filas, series, etiquetas = [], [], []

    for elitismo, nombre in ((0, "sin elitismo"), (2, "con elitismo (e=2)")):
        distancias, historiales, gens, _ = corridas(
            matriz, AJUSTADO, tasa_mutacion=0.2, elitismo=elitismo,
            cruce="OX", mutacion="inversion")
        filas.append(fila_resumen(nombre, distancias, gens))
        series.append(np.mean(historiales, axis=0))
        etiquetas.append(nombre)
        print(f"  {nombre:<20} -> mejor={min(distancias):8.2f}  "
              f"peor={max(distancias):8.2f}  promedio={np.mean(distancias):8.2f}")

    grafica_convergencia(
        series, etiquetas,
        titulo=f"TSP {n} ciudades: efecto del elitismo "
               f"(presupuesto ajustado, promedio de {CORRIDAS} corridas)",
        ylabel="Distancia de la mejor ruta",
        archivo="tsp_elitismo.png")
    return pd.DataFrame(filas)


# ==========================================================================
# ESTUDIO E: que pasa cuando aumenta el numero de ciudades
# ==========================================================================


def estudio_ciudades_crecientes(tamanos=(10, 15, 30, 50), presupuesto=AMPLIO):
    """Mismo presupuesto de computo, instancias cada vez mas grandes.

    Como referencia se usa la heuristica del vecino mas cercano: la columna
    'mejora_vs_voraz_%' muestra cuanto le gana el GA a una heuristica simple.
    """
    filas = []
    for n in tamanos:
        _, matriz = crear_instancia(n)
        distancias, _, gens, _ = corridas(
            matriz, presupuesto, tasa_mutacion=0.2, elitismo=2,
            cruce="OX", mutacion="inversion")
        voraz = distancia_ruta(ruta_vecino_mas_cercano(matriz), matriz)
        promedio = float(np.mean(distancias))

        filas.append(fila_resumen(
            f"{n} ciudades", distancias, gens,
            {"vecino_mas_cercano": round(voraz, 2),
             "mejora_vs_voraz_%": round(100 * (voraz - promedio) / voraz, 2),
             "dispersion_relativa_%": round(100 * np.std(distancias) / promedio, 2)}))
        print(f"  {n:>2} ciudades -> promedio={promedio:8.2f}  "
              f"voraz={voraz:8.2f}  desv={np.std(distancias):6.2f}  "
              f"dispersion={100 * np.std(distancias) / promedio:5.2f}%")
    return pd.DataFrame(filas)


# ==========================================================================
# DEMOSTRACION: por que un cruce binario simple rompe la permutacion
# ==========================================================================


def demostracion_cruce_invalido(n=8, semilla=1):
    """Aplica un cruce de un punto sobre dos rutas y muestra el destrozo.

    Responde la pregunta del enunciado sobre por que no puede usarse un
    cruzamiento binario simple sin controlar duplicados.
    """
    from ga.operadores import cruce_un_punto

    rng = np.random.default_rng(semilla)
    p1, p2 = rng.permutation(n), rng.permutation(n)
    h1, _ = cruce_un_punto(p1, p2, rng)

    conteo = np.bincount(h1, minlength=n)
    return {
        "padre_1": [int(c) for c in p1],
        "padre_2": [int(c) for c in p2],
        "hijo_cruce_un_punto": [int(c) for c in h1],
        "ciudades_repetidas": [int(c) for c in np.where(conteo > 1)[0]],
        "ciudades_no_visitadas": [int(c) for c in np.where(conteo == 0)[0]],
        "es_ruta_valida": es_permutacion_valida(h1, n),
    }


if __name__ == "__main__":
    print("ESTUDIO A: instancias del enunciado (presupuesto amplio: pob=100, gen=300)")
    df_a = estudio_escalado()
    print(f"  -> {guardar_tabla(df_a, 'tsp_escalado.csv')}\n")

    print("ESTUDIO B: dos tasas de mutacion (presupuesto ajustado: pob=30, gen=60)")
    df_b = estudio_tasas_mutacion()
    print(f"  -> {guardar_tabla(df_b, 'tsp_tasas_mutacion.csv')}\n")

    print("ESTUDIO C: operadores de cruce y mutacion (presupuesto ajustado)")
    df_c = estudio_operadores()
    print(f"  -> {guardar_tabla(df_c, 'tsp_operadores.csv')}\n")

    print("ESTUDIO D: efecto del elitismo (presupuesto ajustado)")
    df_d = estudio_elitismo()
    print(f"  -> {guardar_tabla(df_d, 'tsp_elitismo.csv')}\n")

    print("ESTUDIO E: aumento del numero de ciudades (presupuesto amplio)")
    df_e = estudio_ciudades_crecientes()
    print(f"  -> {guardar_tabla(df_e, 'tsp_ciudades_crecientes.csv')}\n")

    print("DEMOSTRACION: cruce de un punto sobre permutaciones")
    demo = demostracion_cruce_invalido()
    for clave, valor in demo.items():
        print(f"  {clave:<22}: {valor}")
    guardar_tabla(pd.DataFrame([{k: str(v) for k, v in demo.items()}]),
                  "tsp_demo_cruce_invalido.csv")

    print("\nExperimentos del TSP completados.")
