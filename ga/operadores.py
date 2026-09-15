"""
Operadores geneticos de cruce (recombinacion) y mutacion (Tweak).

Se dividen en dos familias segun la representacion del cromosoma:

* PERMUTACION -> N-Reinas y TSP. El cromosoma es una permutacion de 0..n-1 y
  todo operador DEBE devolver otra permutacion valida (sin repetidos ni
  faltantes). Por eso se usan OX y PMX en vez de un cruce de un punto.
* BINARIA / ENTERA -> Mochila y Asignacion de cursos. Cada gen es independiente
  de los demas, asi que los cruces clasicos (un punto, uniforme) son validos.

Convencion: todo cruce recibe dos padres y devuelve DOS hijos, tal como se
describe en la Clase 4 (operador de recombinacion, reproduccion sexual).
Toda mutacion recibe un individuo y devuelve una copia mutada.
"""

import numpy as np

# ==========================================================================
# CRUCE PARA CROMOSOMAS DE PERMUTACION
# ==========================================================================


def cruce_ox(padre1, padre2, rng):
    """Order Crossover (OX).

    Copia un segmento contiguo del padre1 en el hijo y rellena las posiciones
    restantes con los genes del padre2 en el orden en que aparecen, saltando
    los que ya estan presentes. Preserva el ORDEN RELATIVO de las ciudades,
    que es justo la informacion util en el TSP.
    """
    n = len(padre1)
    # Dos puntos de corte aleatorios: el segmento copiado es [i, j)
    i, j = sorted(rng.choice(n + 1, size=2, replace=False))
    if i == j:
        j = min(i + 1, n)
    return _ox_un_hijo(padre1, padre2, i, j), _ox_un_hijo(padre2, padre1, i, j)


def _ox_un_hijo(donante, resto, i, j):
    """Construye un hijo OX: segmento [i, j) del donante y el resto en orden."""
    n = len(donante)
    hijo = np.full(n, -1, dtype=donante.dtype)
    hijo[i:j] = donante[i:j]
    ya_usados = set(donante[i:j].tolist())

    # Se recorre el otro padre a partir de j (de forma circular) colocando los
    # genes que faltan en las posiciones libres, tambien a partir de j.
    pos = j % n
    for k in range(n):
        gen = resto[(j + k) % n]
        if gen not in ya_usados:
            hijo[pos] = gen
            ya_usados.add(gen)
            pos = (pos + 1) % n
    return hijo


def cruce_pmx(padre1, padre2, rng):
    """Partially Mapped Crossover (PMX).

    Intercambia un segmento entre los padres y usa el mapeo gen-a-gen inducido
    por ese segmento para resolver los duplicados que quedan fuera. Preserva
    mejor la POSICION ABSOLUTA de los genes que OX.
    """
    n = len(padre1)
    i, j = sorted(rng.choice(n + 1, size=2, replace=False))
    if i == j:
        j = min(i + 1, n)
    return _pmx_un_hijo(padre1, padre2, i, j), _pmx_un_hijo(padre2, padre1, i, j)


def _pmx_un_hijo(donante, resto, i, j):
    """Construye un hijo PMX: segmento del donante y mapeo para el resto."""
    n = len(donante)
    hijo = np.full(n, -1, dtype=donante.dtype)
    hijo[i:j] = donante[i:j]

    # mapeo[valor del donante] = valor del otro padre en la misma posicion
    mapeo = {int(donante[p]): int(resto[p]) for p in range(i, j)}
    en_segmento = set(donante[i:j].tolist())

    for p in range(n):
        if i <= p < j:
            continue
        gen = int(resto[p])
        # Si el gen ya esta en el segmento copiado se sigue la cadena de
        # mapeos hasta encontrar un valor libre.
        visitados = 0
        while gen in en_segmento and visitados <= n:
            gen = mapeo[gen]
            visitados += 1
        hijo[p] = gen
    return hijo


# ==========================================================================
# MUTACION PARA CROMOSOMAS DE PERMUTACION
# ==========================================================================


def mut_swap(individuo, tasa, rng):
    """Mutacion por intercambio: permuta dos genes elegidos al azar.

    Es el Tweak mas pequeno posible sobre una permutacion: en el TSP cambia
    como maximo 4 aristas de la ruta.
    """
    hijo = individuo.copy()
    if rng.random() < tasa:
        i, j = rng.choice(len(hijo), size=2, replace=False)
        hijo[i], hijo[j] = hijo[j], hijo[i]
    return hijo


def mut_inversion(individuo, tasa, rng):
    """Mutacion por inversion: invierte un segmento contiguo del cromosoma.

    Equivale a un movimiento 2-opt del TSP: rompe exactamente 2 aristas y las
    reconecta, por lo que suele ser el operador mas efectivo en rutas.
    """
    hijo = individuo.copy()
    if rng.random() < tasa:
        i, j = sorted(rng.choice(len(hijo), size=2, replace=False))
        hijo[i:j + 1] = hijo[i:j + 1][::-1]
    return hijo


def mut_insercion(individuo, tasa, rng):
    """Mutacion por insercion: saca un gen y lo reinserta en otra posicion.

    Desplaza el resto de genes, por lo que altera mas el orden relativo que el
    intercambio pero menos que una reinicializacion completa.
    """
    hijo = individuo.copy()
    if rng.random() < tasa:
        origen, destino = rng.choice(len(hijo), size=2, replace=False)
        gen = hijo[origen]
        hijo = np.delete(hijo, origen)
        hijo = np.insert(hijo, destino, gen)
    return hijo


# ==========================================================================
# CRUCE PARA CROMOSOMAS BINARIOS Y ENTEROS
# ==========================================================================


def cruce_un_punto(padre1, padre2, rng):
    """Cruce de un punto: corta ambos padres en la misma posicion y los cruza.

    Solo es valido cuando los genes son INDEPENDIENTES entre si (mochila,
    asignacion de cursos). Sobre una permutacion produciria valores repetidos.
    """
    n = len(padre1)
    punto = int(rng.integers(1, n)) if n > 1 else 1
    hijo1 = np.concatenate((padre1[:punto], padre2[punto:]))
    hijo2 = np.concatenate((padre2[:punto], padre1[punto:]))
    return hijo1, hijo2


def cruce_uniforme(padre1, padre2, rng, prob=0.5):
    """Cruce uniforme: cada gen se hereda de un padre u otro segun una mascara.

    Mezcla mucho mas que el cruce de un punto porque no conserva bloques
    contiguos, lo que ayuda cuando no hay relacion de vecindad entre genes.
    """
    mascara = rng.random(len(padre1)) < prob
    hijo1 = np.where(mascara, padre1, padre2)
    hijo2 = np.where(mascara, padre2, padre1)
    return hijo1, hijo2


# ==========================================================================
# MUTACION PARA CROMOSOMAS BINARIOS Y ENTEROS
# ==========================================================================


def mut_bitflip(individuo, tasa, rng):
    """Mutacion bit-flip: invierte cada gen con probabilidad 'tasa'.

    Aqui la tasa es POR GEN, no por individuo: con 15 objetos y tasa 0.1 se
    esperan 1.5 bits cambiados en cada hijo.
    """
    hijo = individuo.copy()
    mascara = rng.random(len(hijo)) < tasa
    hijo[mascara] = 1 - hijo[mascara]
    return hijo


def mut_reasignacion(individuo, tasa, rng, n_valores):
    """Mutacion por reasignacion: cada gen toma un nuevo valor al azar dentro
    del rango [0, n_valores) con probabilidad 'tasa' (por gen).

    Se usa en la asignacion de cursos, donde cada gen es un slot sala-franja.
    """
    hijo = individuo.copy()
    mascara = rng.random(len(hijo)) < tasa
    n_cambios = int(mascara.sum())
    if n_cambios:
        hijo[mascara] = rng.integers(0, n_valores, size=n_cambios)
    return hijo


# ==========================================================================
# UTILIDADES DE VALIDACION (usadas en pruebas y en el informe)
# ==========================================================================


def es_permutacion_valida(individuo, n):
    """True si el individuo es una permutacion completa de 0..n-1."""
    return sorted(int(g) for g in individuo) == list(range(n))
