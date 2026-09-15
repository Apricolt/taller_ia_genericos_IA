"""
Utilidades de experimentacion: resumen estadistico de multiples corridas y
graficas de convergencia.

Un algoritmo genetico es ESTOCASTICO, por lo que una sola corrida no dice nada.
El taller exige ejecutar al menos 10 veces y reportar mejor, peor y promedio;
estas funciones centralizan ese calculo para los cuatro problemas.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # backend sin ventana: permite guardar PNG desde scripts
import matplotlib.pyplot as plt

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_RESULTADOS = os.path.join(RAIZ, "resultados")
DIR_DATOS = os.path.join(RAIZ, "datos")


def ruta_resultado(nombre):
    """Ruta absoluta de un archivo dentro de resultados/ (creando la carpeta)."""
    os.makedirs(DIR_RESULTADOS, exist_ok=True)
    return os.path.join(DIR_RESULTADOS, nombre)


def ruta_dato(nombre):
    """Ruta absoluta de un archivo dentro de datos/ (creando la carpeta)."""
    os.makedirs(DIR_DATOS, exist_ok=True)
    return os.path.join(DIR_DATOS, nombre)


def resumen(valores):
    """Estadisticas de una lista de valores de N corridas independientes.

    Devuelve mejor (minimo o maximo segun el contexto se interpreta fuera),
    peor, promedio y desviacion estandar.
    """
    v = np.asarray(valores, dtype=float)
    return {
        "n_corridas": int(v.size),
        "minimo": float(v.min()),
        "maximo": float(v.max()),
        "promedio": float(v.mean()),
        "desviacion": float(v.std(ddof=0)),
    }


def resumen_costos(costos):
    """Resumen para problemas de MINIMIZACION (distancia, conflictos, penalizacion).

    Renombra minimo/maximo como mejor/peor para que la tabla del informe se lea
    sin ambiguedad.
    """
    r = resumen(costos)
    return {
        "n_corridas": r["n_corridas"],
        "mejor": r["minimo"],
        "peor": r["maximo"],
        "promedio": r["promedio"],
        "desviacion": r["desviacion"],
    }


def resumen_valores(valores):
    """Resumen para problemas de MAXIMIZACION (valor de la mochila)."""
    r = resumen(valores)
    return {
        "n_corridas": r["n_corridas"],
        "mejor": r["maximo"],
        "peor": r["minimo"],
        "promedio": r["promedio"],
        "desviacion": r["desviacion"],
    }


def grafica_convergencia(series, etiquetas, titulo, ylabel, archivo,
                         xlabel="Generacion"):
    """Dibuja una o varias curvas de convergencia y las guarda como PNG.

    'series' es una lista de listas (una por configuracion comparada).
    """
    plt.figure(figsize=(9, 5))
    for serie, etiqueta in zip(series, etiquetas):
        plt.plot(serie, linewidth=1.8, label=etiqueta)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titulo)
    plt.grid(alpha=0.3)
    if len(series) > 1 or etiquetas[0]:
        plt.legend()
    plt.tight_layout()
    destino = ruta_resultado(archivo)
    plt.savefig(destino, dpi=140)
    plt.close()
    return destino


def grafica_barras(categorias, valores, titulo, ylabel, archivo, errores=None):
    """Grafico de barras comparativo (por ejemplo, con y sin elitismo)."""
    plt.figure(figsize=(9, 5))
    posiciones = np.arange(len(categorias))
    plt.bar(posiciones, valores, yerr=errores, capsize=4, color="#4C72B0", alpha=0.9)
    plt.xticks(posiciones, categorias, rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(titulo)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    destino = ruta_resultado(archivo)
    plt.savefig(destino, dpi=140)
    plt.close()
    return destino


def guardar_tabla(df, archivo):
    """Guarda un DataFrame como CSV en resultados/ y lo devuelve."""
    destino = ruta_resultado(archivo)
    df.to_csv(destino, index=False, encoding="utf-8-sig")
    return destino
