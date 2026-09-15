"""
EJERCICIO 1: PROBLEMA DEL AGENTE VIAJERO (TSP)

Una persona debe visitar N ciudades exactamente una vez y regresar a la ciudad
inicial, minimizando la distancia total recorrida.

REPRESENTACION
    Cromosoma = permutacion de las ciudades, por ejemplo [0, 3, 1, 5, 2, 4].
    La ruta se interpreta como un ciclo: se visita en ese orden y desde la
    ultima ciudad se vuelve a la primera.

    El espacio de busqueda tiene (N-1)!/2 rutas distintas (se fija la ciudad de
    partida y se descarta el sentido de recorrido). Para N=15 son ya mas de
    43.000 millones de rutas, imposible de enumerar: de ahi la necesidad de una
    metaheuristica.

FUNCION DE APTITUD
    distancia_ruta() suma las distancias del ciclo completo, incluido el regreso.
    Como el motor maximiza, se ofrecen dos transformaciones:
      - "inversa"  -> 1/(distancia + epsilon), la que sugiere el enunciado.
                      Es siempre positiva, que es lo que exige la seleccion por
                      ruleta.
      - "negativa" -> -distancia. Equivalente para la seleccion por torneo,
                      porque el torneo solo compara ordenes, no magnitudes.

OPERADORES
    Cruce   : OX (Order Crossover) o PMX.
    Mutacion: intercambio (swap), inversion o insercion.
    NO se puede usar un cruce de un punto sin reparar: mezclaria dos rutas
    produciendo ciudades repetidas y ciudades no visitadas.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from ga.core import ConfigGA, ejecutar_ga
from ga.operadores import (cruce_ox, cruce_pmx, mut_swap, mut_inversion,
                           mut_insercion)
from ga.estadisticas import ruta_dato

CRUCES = {"OX": cruce_ox, "PMX": cruce_pmx}
MUTACIONES = {"swap": mut_swap, "inversion": mut_inversion, "insercion": mut_insercion}

EPSILON = 1e-9          # evita la division por cero en la aptitud inversa
SEMILLA_DATOS = 2024    # fija la instancia: el mapa es el mismo en todas las corridas


# ==========================================================================
# INSTANCIA DEL PROBLEMA (matriz de distancias)
# ==========================================================================


def crear_instancia(n_ciudades, semilla=SEMILLA_DATOS, lado=100.0):
    """Genera N ciudades en el plano y su matriz de distancias euclidianas.

    Se usan coordenadas para poder DIBUJAR la ruta; la matriz resultante es
    simetrica y cumple la desigualdad triangular, como en un mapa real.
    """
    rng = np.random.default_rng(semilla)
    coordenadas = rng.uniform(0, lado, size=(n_ciudades, 2))
    diferencias = coordenadas[:, None, :] - coordenadas[None, :, :]
    matriz = np.sqrt((diferencias ** 2).sum(axis=2))
    return coordenadas, matriz


def guardar_instancia(n_ciudades, semilla=SEMILLA_DATOS):
    """Escribe coordenadas y matriz de distancias en datos/ (para el informe)."""
    coordenadas, matriz = crear_instancia(n_ciudades, semilla)
    etiquetas = [f"C{i}" for i in range(n_ciudades)]

    df_coord = pd.DataFrame(coordenadas, columns=["x", "y"], index=etiquetas)
    df_coord.index.name = "ciudad"
    df_coord.round(2).to_csv(ruta_dato(f"tsp_coordenadas_{n_ciudades}.csv"),
                             encoding="utf-8-sig")

    df_matriz = pd.DataFrame(matriz, index=etiquetas, columns=etiquetas)
    df_matriz.index.name = "ciudad"
    df_matriz.round(2).to_csv(ruta_dato(f"tsp_matriz_{n_ciudades}.csv"),
                              encoding="utf-8-sig")
    return coordenadas, matriz


# ==========================================================================
# FUNCION DE APTITUD
# ==========================================================================


def distancia_ruta(ruta, matriz):
    """Distancia total del ciclo, INCLUYENDO el regreso a la ciudad inicial.

    np.roll desplaza la ruta una posicion, de modo que matriz[ruta, siguiente]
    recorre todas las aristas del ciclo de una sola vez.
    """
    siguiente = np.roll(ruta, -1)
    return float(matriz[ruta, siguiente].sum())


def crear_aptitud(matriz, tipo="inversa"):
    """Devuelve la funcion de aptitud (a MAXIMIZAR) para una matriz dada."""
    if tipo == "inversa":
        return lambda ruta: 1.0 / (distancia_ruta(ruta, matriz) + EPSILON)
    if tipo == "negativa":
        return lambda ruta: -distancia_ruta(ruta, matriz)
    raise ValueError("Tipo de aptitud desconocido: " + str(tipo))


def aptitud_a_distancia(valor, tipo="inversa"):
    """Convierte un valor de aptitud de vuelta a distancia (para graficar)."""
    return (1.0 / valor - EPSILON) if tipo == "inversa" else -valor


# ==========================================================================
# CONFIGURACION DEL ALGORITMO GENETICO
# ==========================================================================


def construir_config(matriz, tam_poblacion=100, generaciones=300,
                     tasa_mutacion=0.2, elitismo=2, cruce="OX",
                     mutacion="inversion", seleccion="torneo",
                     tipo_aptitud="inversa", semilla=None):
    """Arma la ConfigGA para una instancia concreta del TSP."""
    n = len(matriz)
    return ConfigGA(
        crear_individuo=lambda rng: rng.permutation(n),
        aptitud=crear_aptitud(matriz, tipo_aptitud),
        cruce=CRUCES[cruce],
        mutacion=MUTACIONES[mutacion],
        tam_poblacion=tam_poblacion,
        generaciones=generaciones,
        tasa_mutacion=tasa_mutacion,
        elitismo=elitismo,
        seleccion=seleccion,
        semilla=semilla,
    )


def resolver(matriz, **kwargs):
    """Ejecuta una corrida del GA sobre la matriz de distancias dada."""
    return ejecutar_ga(construir_config(matriz, **kwargs))


# ==========================================================================
# REFERENCIAS PARA EVALUAR LA CALIDAD DE LA SOLUCION
# ==========================================================================


def ruta_vecino_mas_cercano(matriz, inicio=0):
    """Heuristica voraz del vecino mas cercano: punto de comparacion clasico."""
    n = len(matriz)
    pendientes = set(range(n)) - {inicio}
    ruta = [inicio]
    actual = inicio
    while pendientes:
        siguiente = min(pendientes, key=lambda c: matriz[actual, c])
        ruta.append(siguiente)
        pendientes.remove(siguiente)
        actual = siguiente
    return np.array(ruta)


def optimo_exacto(matriz):
    """Optimo real por fuerza bruta. Solo viable hasta unas 10 ciudades.

    Fija la ciudad 0 y permuta las restantes: (N-1)! rutas.
    """
    from itertools import permutations

    n = len(matriz)
    if n > 10:
        raise ValueError("La fuerza bruta solo es viable hasta 10 ciudades")
    mejor_ruta, mejor_dist = None, float("inf")
    for resto in permutations(range(1, n)):
        ruta = np.array((0,) + resto)
        d = distancia_ruta(ruta, matriz)
        if d < mejor_dist:
            mejor_dist, mejor_ruta = d, ruta
    return mejor_ruta, mejor_dist


# ==========================================================================
# VISUALIZACION
# ==========================================================================


def dibujar_ruta(coordenadas, ruta, titulo=None, archivo=None):
    """Dibuja el mapa de ciudades y la ruta como un ciclo cerrado."""
    import matplotlib.pyplot as plt

    ciclo = np.append(ruta, ruta[0])  # se repite la primera para cerrar el ciclo
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(coordenadas[ciclo, 0], coordenadas[ciclo, 1], "-o",
            color="#4C72B0", markersize=7, linewidth=1.6, zorder=1)
    ax.scatter(coordenadas[ruta[0], 0], coordenadas[ruta[0], 1],
               color="#C44E52", s=160, zorder=2, label="Ciudad inicial")

    for i, (x, y) in enumerate(coordenadas):
        ax.annotate(f"C{i}", (x, y), textcoords="offset points",
                    xytext=(7, 6), fontsize=9)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(titulo or "Mejor ruta encontrada")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if archivo:
        fig.savefig(archivo, dpi=140)
    return fig


# ==========================================================================
# EJECUCION DIRECTA
# ==========================================================================

if __name__ == "__main__":
    from ga.estadisticas import grafica_convergencia, ruta_resultado
    from ga.operadores import es_permutacion_valida

    for n in (8, 10, 15):
        guardar_instancia(n)
    print("Instancias (coordenadas y matrices de distancia) guardadas en datos/\n")

    N_CIUDADES = 10
    coordenadas, matriz = crear_instancia(N_CIUDADES)

    res = resolver(matriz, tam_poblacion=100, generaciones=300, tasa_mutacion=0.2,
                   elitismo=2, cruce="OX", mutacion="inversion", semilla=42)

    distancia = distancia_ruta(res.mejor_individuo, matriz)
    assert es_permutacion_valida(res.mejor_individuo, N_CIUDADES), "ruta invalida"

    print("=" * 62)
    print(f"TSP con {N_CIUDADES} ciudades")
    print("=" * 62)
    print(f"Mejor ruta   : {[int(c) for c in res.mejor_individuo]}")
    print(f"Distancia    : {distancia:.2f}")
    print(f"Generacion   : {res.generacion_mejor} de {res.generaciones_ejecutadas}")
    print(f"Tiempo       : {res.tiempo_seg:.3f} s")

    ruta_voraz = ruta_vecino_mas_cercano(matriz)
    print(f"\nVecino mas cercano (heuristica voraz): {distancia_ruta(ruta_voraz, matriz):.2f}")

    _, dist_optima = optimo_exacto(matriz)
    brecha = 100 * (distancia - dist_optima) / dist_optima
    print(f"Optimo exacto por fuerza bruta       : {dist_optima:.2f}")
    print(f"Brecha del algoritmo genetico        : {brecha:.2f} %")

    dibujar_ruta(coordenadas, res.mejor_individuo,
                 titulo=f"TSP {N_CIUDADES} ciudades - distancia {distancia:.2f}",
                 archivo=ruta_resultado(f"tsp_ruta_{N_CIUDADES}.png"))
    grafica_convergencia(
        [[aptitud_a_distancia(v) for v in res.historial_mejor]], [""],
        titulo=f"TSP {N_CIUDADES} ciudades: convergencia",
        ylabel="Distancia de la mejor ruta", archivo=f"tsp_convergencia_{N_CIUDADES}.png")
    print("\nGraficas guardadas en resultados/")
