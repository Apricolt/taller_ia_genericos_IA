"""
Metodos de seleccion de padres.

Todos reciben el vector de aptitudes (siempre en clave de MAXIMIZAR), el
numero de padres a elegir y un generador aleatorio, y devuelven un arreglo de
INDICES de la poblacion. La presion selectiva de cada metodo determina el
equilibrio entre exploracion y explotacion descrito en la Clase 4.
"""

import numpy as np


def seleccion_torneo(aptitudes, n_padres, rng, k=3):
    """Seleccion por torneo: se sortean k individuos y gana el mas apto.

    Es el metodo por defecto porque su presion selectiva se ajusta con k
    (k=2 poca presion y mucha diversidad, k grande mucha explotacion) y porque
    no le afecta la escala ni el signo de la aptitud.
    """
    aptitudes = np.asarray(aptitudes, dtype=float)
    n = len(aptitudes)
    competidores = rng.integers(0, n, size=(n_padres, k))
    ganadores = competidores[np.arange(n_padres), aptitudes[competidores].argmax(axis=1)]
    return ganadores


def seleccion_ruleta(aptitudes, n_padres, rng):
    """Seleccion proporcional a la aptitud (ruleta).

    Cada individuo ocupa una porcion de la ruleta proporcional a su aptitud.
    Como exige valores no negativos, las aptitudes se desplazan restando el
    minimo. En el TSP se combina con la aptitud 1/(distancia+epsilon) que pide
    el enunciado, precisamente para que todos los valores sean positivos.
    """
    aptitudes = np.asarray(aptitudes, dtype=float)
    pesos = aptitudes - aptitudes.min()
    total = pesos.sum()
    if total <= 0:
        # Toda la poblacion tiene la misma aptitud: seleccion uniforme.
        return rng.integers(0, len(aptitudes), size=n_padres)
    return rng.choice(len(aptitudes), size=n_padres, replace=True, p=pesos / total)


def seleccion_truncamiento(aptitudes, n_padres, rng, fraccion=0.5):
    """Seleccion por truncamiento: solo la mejor fraccion puede reproducirse.

    Es la seleccion que usan las Estrategias Evolutivas (mu, lambda) de la
    Clase 4 y tambien la del codigo original de N-Reinas. Converge rapido pero
    pierde diversidad, con riesgo de convergencia prematura.
    """
    aptitudes = np.asarray(aptitudes, dtype=float)
    n = len(aptitudes)
    corte = max(2, int(round(n * fraccion)))
    mejores = np.argsort(aptitudes)[-corte:]
    return rng.choice(mejores, size=n_padres, replace=True)


# Registro para elegir el metodo por nombre desde la configuracion o la web.
METODOS = {
    "torneo": seleccion_torneo,
    "ruleta": seleccion_ruleta,
    "truncamiento": seleccion_truncamiento,
}
