"""
Motor generico de Algoritmo Genetico (modelo GENERACIONAL con elitismo).

Sigue el esquema de la Clase 4 (Luke, Essentials of Metaheuristics):

    1. Inicializar la poblacion con individuos aleatorios.
    2. Repetir hasta el criterio de parada:
       a. Evaluar la aptitud de todos los individuos.
       b. Almacenar el mejor individuo encontrado hasta el momento.
       c. Reproduccion: Seleccion de padres + Cruce + Mutacion (Tweak).
       d. Operador Join: reemplazo de la poblacion, conservando 'e' elites.

El motor es agnostico al problema: recibe por callback la forma de crear un
individuo, la funcion de aptitud, el cruce, la mutacion y (opcionalmente) un
reparador. Eso permite resolver N-Reinas, TSP, Asignacion y Mochila con
exactamente el mismo bucle evolutivo, que es lo que hace comparables los
resultados entre problemas.

CONVENCION IMPORTANTE: la aptitud SIEMPRE se MAXIMIZA. Los problemas de
minimizacion (conflictos, distancia, penalizacion) devuelven el negativo del
costo o 1/(costo+epsilon).
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, List

import numpy as np

from ga.seleccion import METODOS


@dataclass
class ConfigGA:
    """Parametros de una corrida del algoritmo genetico."""

    # --- Definicion del problema (callbacks obligatorios) ---
    crear_individuo: Callable          # (rng) -> np.ndarray
    aptitud: Callable                  # (individuo) -> float  (se MAXIMIZA)
    cruce: Callable                    # (p1, p2, rng) -> (h1, h2)
    mutacion: Callable                 # (individuo, tasa, rng) -> individuo

    # --- Parametros evolutivos ---
    tam_poblacion: int = 100           # mu: tamano de la muestra de soluciones
    generaciones: int = 200            # maximo de ciclos evaluacion-reproduccion
    prob_cruce: float = 0.9            # probabilidad de recombinar una pareja
    tasa_mutacion: float = 0.1         # intensidad del Tweak
    elitismo: int = 2                  # numero de elites; 0 = sin elitismo
    seleccion: str = "torneo"          # torneo | ruleta | truncamiento
    k_torneo: int = 3                  # presion selectiva del torneo

    # --- Manejo de individuos invalidos (opcional) ---
    reparar: Optional[Callable] = None  # (individuo) -> individuo factible

    # --- Criterios de parada adicionales ---
    aptitud_objetivo: Optional[float] = None  # parar si mejor >= objetivo
    max_estancamiento: Optional[int] = None   # parar si no mejora en K gens

    # --- Reproducibilidad ---
    semilla: Optional[int] = None


@dataclass
class ResultadoGA:
    """Resultado completo de una corrida, con el historial para graficar."""

    mejor_individuo: np.ndarray
    mejor_aptitud: float
    generacion_mejor: int              # generacion en que aparecio la mejor
    generaciones_ejecutadas: int
    exito: bool                        # True si se alcanzo aptitud_objetivo
    tiempo_seg: float
    historial_mejor: List[float] = field(default_factory=list)
    historial_promedio: List[float] = field(default_factory=list)
    historial_peor: List[float] = field(default_factory=list)
    # Poblacion tal como quedo al terminar. Permite medir, por ejemplo, que
    # fraccion de los individuos es todavia invalida bajo la estrategia de
    # penalizacion frente a la de reparacion.
    poblacion_final: List[np.ndarray] = field(default_factory=list)


def ejecutar_ga(cfg: ConfigGA) -> ResultadoGA:
    """Ejecuta una corrida completa del algoritmo genetico."""
    rng = np.random.default_rng(cfg.semilla)
    inicio = time.perf_counter()

    metodo_seleccion = METODOS[cfg.seleccion]
    kwargs_seleccion = {"k": cfg.k_torneo} if cfg.seleccion == "torneo" else {}

    # --- 1. Inicializacion de la poblacion ---
    poblacion = [_preparar(cfg.crear_individuo(rng), cfg) for _ in range(cfg.tam_poblacion)]

    mejor_individuo = None
    mejor_aptitud = -np.inf
    generacion_mejor = 0
    exito = False
    hist_mejor, hist_prom, hist_peor = [], [], []
    generacion = 0

    for generacion in range(1, cfg.generaciones + 1):
        # --- 2a. Evaluacion de la aptitud de toda la poblacion ---
        aptitudes = np.array([cfg.aptitud(ind) for ind in poblacion], dtype=float)

        hist_mejor.append(float(aptitudes.max()))
        hist_prom.append(float(aptitudes.mean()))
        hist_peor.append(float(aptitudes.min()))

        # --- 2b. Memoria del mejor individuo de todos los tiempos ---
        idx_mejor = int(aptitudes.argmax())
        if aptitudes[idx_mejor] > mejor_aptitud:
            mejor_aptitud = float(aptitudes[idx_mejor])
            mejor_individuo = poblacion[idx_mejor].copy()
            generacion_mejor = generacion

        # --- Criterios de parada ---
        if cfg.aptitud_objetivo is not None and mejor_aptitud >= cfg.aptitud_objetivo:
            exito = True
            break
        if cfg.max_estancamiento is not None and generacion - generacion_mejor >= cfg.max_estancamiento:
            break

        # --- 2c. Reproduccion: seleccion + cruce + mutacion ---
        # El elitismo se implementa en el operador Join: las 'e' mejores
        # soluciones pasan intactas a la siguiente generacion, de modo que la
        # calidad del mejor individuo nunca puede empeorar (equivale al
        # esquema (mu + lambda) de las Estrategias Evolutivas).
        nueva_poblacion = []
        if cfg.elitismo > 0:
            elites = np.argsort(aptitudes)[-cfg.elitismo:]
            nueva_poblacion.extend(poblacion[i].copy() for i in elites)

        faltantes = cfg.tam_poblacion - len(nueva_poblacion)
        padres_idx = metodo_seleccion(aptitudes, faltantes + 1, rng, **kwargs_seleccion)

        i = 0
        while len(nueva_poblacion) < cfg.tam_poblacion:
            p1 = poblacion[padres_idx[i % len(padres_idx)]]
            p2 = poblacion[padres_idx[(i + 1) % len(padres_idx)]]
            i += 2

            if rng.random() < cfg.prob_cruce:
                h1, h2 = cfg.cruce(p1, p2, rng)
            else:
                h1, h2 = p1.copy(), p2.copy()

            for hijo in (h1, h2):
                if len(nueva_poblacion) < cfg.tam_poblacion:
                    hijo = cfg.mutacion(hijo, cfg.tasa_mutacion, rng)
                    nueva_poblacion.append(_preparar(hijo, cfg))

        # --- 2d. Operador Join: la nueva poblacion reemplaza a la anterior ---
        poblacion = nueva_poblacion

    return ResultadoGA(
        mejor_individuo=mejor_individuo,
        mejor_aptitud=mejor_aptitud,
        generacion_mejor=generacion_mejor,
        generaciones_ejecutadas=generacion,
        exito=exito,
        tiempo_seg=time.perf_counter() - inicio,
        historial_mejor=hist_mejor,
        historial_promedio=hist_prom,
        historial_peor=hist_peor,
        poblacion_final=poblacion,
    )


def _preparar(individuo, cfg):
    """Aplica el reparador si el problema usa estrategia de REPARACION.

    Si cfg.reparar es None el individuo entra tal cual a la poblacion y la
    infactibilidad se castiga desde la funcion de aptitud (estrategia de
    PENALIZACION). Esta es la bifurcacion que comparan los ejercicios 2 y 3.
    """
    return cfg.reparar(individuo) if cfg.reparar is not None else individuo
