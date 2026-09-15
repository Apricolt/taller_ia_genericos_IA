"""
PROBLEMA BASE: N-REINAS

Colocar N reinas en un tablero de N x N sin que ninguna ataque a otra.

REPRESENTACION
    Cromosoma = permutacion de tamano N.
    El INDICE representa la columna y el VALOR representa la fila de la reina.
    Ejemplo para N=6: [2, 5, 1, 4, 0, 3] -> reina de la columna 0 en la fila 2.

    Al usar una permutacion, dos reinas nunca comparten fila (los valores no se
    repiten) ni columna (cada indice aparece una vez). El espacio de busqueda se
    reduce de N^N a N! y solo quedan por resolver los ataques en DIAGONAL.

FUNCION DE APTITUD
    conflictos(tablero) cuenta los pares de reinas que se atacan.
    Es un problema de MINIMIZACION con optimo conocido igual a 0, asi que el
    motor (que siempre maximiza) recibe aptitud = -conflictos.

OPERADORES
    Cruce: OX o PMX (operadores de permutacion).
    Mutacion: intercambio (swap).
    Un cruce de un punto clasico produciria filas repetidas y filas faltantes,
    es decir un cromosoma que ya no es una permutacion valida.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ga.core import ConfigGA, ejecutar_ga
from ga.operadores import cruce_ox, cruce_pmx, mut_swap, mut_inversion

CRUCES = {"OX": cruce_ox, "PMX": cruce_pmx}
MUTACIONES = {"swap": mut_swap, "inversion": mut_inversion}


# ==========================================================================
# FUNCION DE APTITUD
# ==========================================================================


def conflictos(tablero):
    """Numero de pares de reinas que se atacan entre si.

    Version vectorizada. Dos reinas se atacan en diagonal cuando coinciden en
    (fila - columna) o en (fila + columna). Se cuenta cuantas reinas caen en
    cada diagonal y se suman las combinaciones de a dos: c*(c-1)/2.
    """
    tablero = np.asarray(tablero)
    n = len(tablero)
    columnas = np.arange(n)

    # Diagonales que descienden hacia la derecha: fila - columna (desplazado a
    # indices positivos) y diagonales que ascienden: fila + columna.
    diag_desc = np.bincount(tablero - columnas + (n - 1), minlength=2 * n - 1)
    diag_asc = np.bincount(tablero + columnas, minlength=2 * n - 1)
    filas = np.bincount(tablero, minlength=n)  # siempre 0 si es permutacion

    def pares(cuentas):
        return int((cuentas * (cuentas - 1) // 2).sum())

    return pares(diag_desc) + pares(diag_asc) + pares(filas)


def conflictos_didactico(tablero):
    """Misma metrica con el doble bucle explicito del enunciado.

    Se conserva porque es la formulacion que aparece en el material de clase y
    sirve para verificar la version vectorizada.
    """
    n = len(tablero)
    ataques = 0
    for i in range(n):
        for j in range(i + 1, n):
            misma_fila = tablero[i] == tablero[j]
            misma_diagonal = abs(tablero[i] - tablero[j]) == abs(i - j)
            if misma_fila or misma_diagonal:
                ataques += 1
    return ataques


def aptitud(tablero):
    """Aptitud que maximiza el motor: el negativo del numero de conflictos.

    El optimo global vale 0 (ningun ataque), por eso la corrida puede detenerse
    en cuanto la aptitud alcanza ese valor.
    """
    return -conflictos(tablero)


# ==========================================================================
# CONFIGURACION DEL ALGORITMO GENETICO
# ==========================================================================


def construir_config(n, tam_poblacion=100, generaciones=500, tasa_mutacion=0.1,
                     elitismo=2, cruce="OX", mutacion="swap", seleccion="torneo",
                     semilla=None):
    """Arma la ConfigGA para un tablero de N reinas."""
    return ConfigGA(
        crear_individuo=lambda rng: rng.permutation(n),
        aptitud=aptitud,
        cruce=CRUCES[cruce],
        mutacion=MUTACIONES[mutacion],
        tam_poblacion=tam_poblacion,
        generaciones=generaciones,
        tasa_mutacion=tasa_mutacion,
        elitismo=elitismo,
        seleccion=seleccion,
        aptitud_objetivo=0.0,          # 0 conflictos = solucion perfecta
        semilla=semilla,
    )


def resolver(n, **kwargs):
    """Ejecuta una corrida del GA para N reinas y devuelve el ResultadoGA."""
    return ejecutar_ga(construir_config(n, **kwargs))


# ==========================================================================
# VISUALIZACION
# ==========================================================================


def tablero_texto(solucion):
    """Representacion en texto del tablero: Q para reina, . para casilla vacia."""
    n = len(solucion)
    filas = []
    for fila in range(n):
        filas.append(" ".join("Q" if solucion[col] == fila else "." for col in range(n)))
    return "\n".join(filas)


def matriz_tablero(solucion):
    """Matriz N x N con 1 en las casillas ocupadas por una reina."""
    n = len(solucion)
    m = np.zeros((n, n), dtype=int)
    for col in range(n):
        m[solucion[col], col] = 1
    return m


def dibujar_tablero(solucion, archivo=None, titulo=None):
    """Dibuja el tablero de ajedrez con las reinas. Devuelve la figura."""
    import matplotlib.pyplot as plt

    n = len(solucion)
    fig, ax = plt.subplots(figsize=(5, 5))

    # Patron de casillas claras y oscuras
    patron = np.add.outer(np.arange(n), np.arange(n)) % 2
    ax.imshow(patron, cmap="binary", alpha=0.25)

    for col in range(n):
        ax.text(col, solucion[col], "♛", ha="center", va="center", fontsize=28)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("Columna (indice del cromosoma)")
    ax.set_ylabel("Fila (valor del gen)")
    ax.set_title(titulo or f"Solucion para N={n} - conflictos: {conflictos(solucion)}")
    fig.tight_layout()

    if archivo:
        fig.savefig(archivo, dpi=140)
    return fig


# ==========================================================================
# EJECUCION DIRECTA
# ==========================================================================

if __name__ == "__main__":
    from ga.estadisticas import grafica_convergencia, ruta_resultado

    # Verificacion de que la aptitud vectorizada coincide con la didactica
    rng = np.random.default_rng(0)
    for _ in range(200):
        t = rng.permutation(8)
        assert conflictos(t) == conflictos_didactico(t)
    print("Verificacion de la funcion de aptitud: OK\n")

    series, etiquetas = [], []
    for n in (6, 8):
        print("=" * 62)
        print(f"N-REINAS con N={n}")
        print("=" * 62)
        # Se usa una poblacion pequena (20) a proposito: con poblaciones de
        # 100 individuos la solucion suele aparecer ya en la poblacion inicial
        # y la curva de convergencia no muestra el trabajo del algoritmo.
        res = resolver(n, tam_poblacion=20, generaciones=500,
                       tasa_mutacion=0.1, elitismo=2, cruce="OX", semilla=42)

        if res.exito:
            print(f"Solucion encontrada en la generacion {res.generacion_mejor}")
        else:
            print(f"Sin solucion perfecta en {res.generaciones_ejecutadas} generaciones")

        print(f"Cromosoma : {[int(g) for g in res.mejor_individuo]}")
        print(f"Conflictos: {conflictos(res.mejor_individuo)}")
        print(f"Tiempo    : {res.tiempo_seg:.3f} s\n")
        print(tablero_texto(res.mejor_individuo), "\n")

        dibujar_tablero(res.mejor_individuo,
                        archivo=ruta_resultado(f"nreinas_tablero_n{n}.png"))
        series.append([-v for v in res.historial_mejor])
        etiquetas.append(f"N={n}")

    destino = grafica_convergencia(
        series, etiquetas,
        titulo="N-Reinas: convergencia del mejor individuo",
        ylabel="Conflictos (menor es mejor)",
        archivo="nreinas_convergencia.png",
    )
    print(f"Graficas guardadas en: {destino}")
