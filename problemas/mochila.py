"""
EJERCICIO 3: PROBLEMA DE LA MOCHILA (KNAPSACK 0/1)

Hay 15 objetos con peso y valor. Se debe elegir una combinacion que MAXIMICE el
valor total sin superar la capacidad de la mochila.

REPRESENTACION
    Cromosoma BINARIO de 15 genes: gen = 1 si el objeto se lleva, 0 si no.

    Por que aqui si sirve un cromosoma binario y en el TSP no:
    la decision sobre cada objeto es INDEPENDIENTE de las demas (llevar el
    objeto 3 no impide llevar el 7), asi que cualquier vector de ceros y unos
    representa una seleccion con sentido. En el TSP, en cambio, los genes estan
    acoplados: visitar una ciudad impide volver a visitarla, y por eso alli hace
    falta una permutacion.

    El espacio de busqueda tiene 2^15 = 32.768 combinaciones, lo bastante
    pequeno como para calcular el OPTIMO EXACTO por programacion dinamica y
    medir asi la brecha real del algoritmo genetico.

FUNCION DE APTITUD
    Maximizar la suma de valores de los objetos seleccionados. El problema esta
    en que hacer cuando el peso supera la capacidad. Se comparan dos estrategias:

    PENALIZAR: aptitud = valor - lambda * exceso_de_peso
               Con lambda >= max(valor_i / peso_i) se garantiza que ninguna
               solucion invalida pueda superar a la mejor solucion valida.

    REPARAR  : antes de evaluar, se van quitando objetos con la peor relacion
               valor/peso hasta que la mochila cabe. Toda la poblacion es
               siempre factible.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from ga.core import ConfigGA, ejecutar_ga
from ga.operadores import cruce_un_punto, cruce_uniforme, mut_bitflip
from ga.estadisticas import ruta_dato

CRUCES = {"un_punto": cruce_un_punto, "uniforme": cruce_uniforme}

N_OBJETOS = 15
SEMILLA_DATOS = 2024


# ==========================================================================
# INSTANCIA DEL PROBLEMA
# ==========================================================================


def crear_objetos(n=N_OBJETOS, semilla=SEMILLA_DATOS):
    """Genera n objetos con pesos y valores enteros distintos entre si.

    Los valores NO son proporcionales a los pesos (se les suma un ruido), para
    que el problema no se resuelva con la simple regla voraz de mayor valor.
    """
    rng = np.random.default_rng(semilla)
    pesos = rng.integers(5, 40, size=n)
    valores = (pesos * rng.uniform(1.5, 4.0, size=n)).round().astype(int)
    return pesos, valores


def capacidades(pesos, fracciones=(0.5, 0.25)):
    """Capacidades a probar, expresadas como fraccion del peso total."""
    total = int(pesos.sum())
    return {f"{int(f * 100)}% del peso total": int(round(total * f)) for f in fracciones}


def guardar_tabla_objetos(pesos, valores):
    """Exporta la tabla de objetos a datos/ para incluirla en el informe."""
    df = pd.DataFrame({
        "objeto": [f"O{i + 1}" for i in range(len(pesos))],
        "peso": pesos,
        "valor": valores,
        "ratio_valor_peso": (valores / pesos).round(2),
    })
    df.to_csv(ruta_dato("objetos_mochila.csv"), index=False, encoding="utf-8-sig")
    return df


# ==========================================================================
# EVALUACION
# ==========================================================================


def peso_total(cromosoma, pesos):
    """Peso de la seleccion codificada en el cromosoma."""
    return int(np.dot(cromosoma, pesos))


def valor_total(cromosoma, valores):
    """Valor de la seleccion codificada en el cromosoma."""
    return int(np.dot(cromosoma, valores))


def lambda_penalizacion(pesos, valores, modo="fuerte"):
    """Coeficiente lambda de la penalizacion aptitud = valor - lambda * exceso.

    Su magnitud decide el equilibrio entre dos efectos opuestos:

    "proporcional" -> lambda = max(valor_i / peso_i).
        Es el valor que suele citarse: cada unidad de peso excedida cuesta lo
        que vale la unidad de peso mas rentable. Conserva el GRADIENTE cerca de
        la frontera (una solucion que se pasa por poco sigue siendo atractiva),
        pero NO garantiza que una solucion invalida no gane: como los objetos
        son discretos, reparar una solucion que se pasa por 1 unidad puede
        obligar a sacar un objeto entero. Es una penalizacion DEBIL.

    "fuerte" -> lambda = suma de todos los valores.
        Cualquier exceso, por minimo que sea, hunde la aptitud por debajo de
        cero. Garantiza que la mejor solucion de la corrida sea siempre valida,
        a costa de perder el gradiente: todas las invalidas se parecen entre si
        y el algoritmo deja de recibir informacion sobre cuanto se pasa.
    """
    if modo == "proporcional":
        return float(np.max(valores / pesos))
    if modo == "fuerte":
        return float(valores.sum())
    raise ValueError("modo debe ser 'proporcional' o 'fuerte'")


def crear_aptitud(pesos, valores, capacidad, estrategia="penalizar",
                  modo_penalizacion="fuerte"):
    """Funcion de aptitud (a MAXIMIZAR) segun la estrategia elegida.

    Con estrategia "reparar" la aptitud es simplemente el valor, porque el
    reparador ya garantiza que todo individuo evaluado es factible.
    """
    if estrategia == "reparar":
        return lambda c: float(valor_total(c, valores))

    lam = lambda_penalizacion(pesos, valores, modo_penalizacion)

    def aptitud(cromosoma):
        exceso = peso_total(cromosoma, pesos) - capacidad
        valor = valor_total(cromosoma, valores)
        return float(valor - lam * exceso) if exceso > 0 else float(valor)

    return aptitud


def crear_reparador(pesos, valores, capacidad):
    """Reparador: quita objetos de peor relacion valor/peso hasta que cabe."""
    orden_peor_primero = np.argsort(valores / pesos)  # ratio ascendente

    def reparar(cromosoma):
        cromosoma = np.asarray(cromosoma, dtype=int).copy()
        peso = peso_total(cromosoma, pesos)
        for i in orden_peor_primero:
            if peso <= capacidad:
                break
            if cromosoma[i] == 1:
                cromosoma[i] = 0
                peso -= int(pesos[i])
        return cromosoma

    return reparar


# ==========================================================================
# OPTIMO EXACTO POR PROGRAMACION DINAMICA
# ==========================================================================


def optimo_exacto(pesos, valores, capacidad):
    """Optimo real del knapsack 0/1 por programacion dinamica.

    Tabla clasica de (n+1) x (capacidad+1): tabla[i][w] es el mejor valor
    alcanzable con los primeros i objetos y capacidad w. Sirve como referencia
    para medir la BRECHA del algoritmo genetico, que es la evidencia mas fuerte
    que se puede presentar sobre su calidad.
    """
    n = len(pesos)
    tabla = np.zeros((n + 1, capacidad + 1), dtype=int)

    for i in range(1, n + 1):
        p, v = int(pesos[i - 1]), int(valores[i - 1])
        tabla[i, :p] = tabla[i - 1, :p]
        tabla[i, p:] = np.maximum(tabla[i - 1, p:], tabla[i - 1, :capacidad + 1 - p] + v)

    # Reconstruccion de los objetos elegidos
    seleccion = np.zeros(n, dtype=int)
    w = capacidad
    for i in range(n, 0, -1):
        if tabla[i, w] != tabla[i - 1, w]:
            seleccion[i - 1] = 1
            w -= int(pesos[i - 1])

    return seleccion, int(tabla[n, capacidad])


# ==========================================================================
# CONFIGURACION DEL ALGORITMO GENETICO
# ==========================================================================


def construir_config(pesos, valores, capacidad, tam_poblacion=60, generaciones=120,
                     tasa_mutacion=0.05, elitismo=2, cruce="un_punto",
                     seleccion="torneo", estrategia="penalizar",
                     modo_penalizacion="fuerte", semilla=None):
    """Arma la ConfigGA para una instancia y capacidad concretas."""
    if estrategia not in ("penalizar", "reparar"):
        raise ValueError("estrategia debe ser 'penalizar' o 'reparar'")

    n = len(pesos)
    return ConfigGA(
        crear_individuo=lambda rng: rng.integers(0, 2, size=n),
        aptitud=crear_aptitud(pesos, valores, capacidad, estrategia,
                              modo_penalizacion),
        cruce=CRUCES[cruce],
        mutacion=mut_bitflip,
        tam_poblacion=tam_poblacion,
        generaciones=generaciones,
        tasa_mutacion=tasa_mutacion,
        elitismo=elitismo,
        seleccion=seleccion,
        reparar=(crear_reparador(pesos, valores, capacidad)
                 if estrategia == "reparar" else None),
        semilla=semilla,
    )


def resolver(pesos, valores, capacidad, **kwargs):
    """Ejecuta una corrida del GA sobre una instancia de la mochila."""
    return ejecutar_ga(construir_config(pesos, valores, capacidad, **kwargs))


# ==========================================================================
# PRESENTACION
# ==========================================================================


def seleccion_dataframe(cromosoma, pesos, valores):
    """Detalle de los objetos seleccionados por una solucion."""
    filas = []
    for i, gen in enumerate(np.asarray(cromosoma, dtype=int)):
        filas.append({
            "objeto": f"O{i + 1}",
            "peso": int(pesos[i]),
            "valor": int(valores[i]),
            "ratio": round(float(valores[i] / pesos[i]), 2),
            "seleccionado": "Si" if gen else "No",
        })
    return pd.DataFrame(filas)


def resumen_solucion(cromosoma, pesos, valores, capacidad):
    """Metricas de una solucion: valor, peso, holgura y validez."""
    peso = peso_total(cromosoma, pesos)
    return {
        "valor": valor_total(cromosoma, valores),
        "peso": peso,
        "capacidad": capacidad,
        "holgura": capacidad - peso,
        "valido": peso <= capacidad,
        "objetos": int(np.sum(cromosoma)),
    }


# ==========================================================================
# EJECUCION DIRECTA
# ==========================================================================

if __name__ == "__main__":
    pd.set_option("display.width", 160)

    pesos, valores = crear_objetos()
    tabla = guardar_tabla_objetos(pesos, valores)

    print("=" * 70)
    print(f"MOCHILA 0/1 con {N_OBJETOS} objetos")
    print("=" * 70)
    print(tabla.to_string(index=False))
    print(f"\nPeso total de todos los objetos : {int(pesos.sum())}")
    print(f"Valor total de todos los objetos: {int(valores.sum())}")
    print(f"Lambda proporcional (max ratio valor/peso): "
          f"{lambda_penalizacion(pesos, valores, 'proporcional'):.3f}")
    print(f"Lambda fuerte (suma de valores)           : "
          f"{lambda_penalizacion(pesos, valores, 'fuerte'):.0f}\n")

    # Las tres formas de tratar a los individuos que no caben en la mochila
    ESTRATEGIAS = [
        ("penalizar", "proporcional", "penalizacion debil"),
        ("penalizar", "fuerte", "penalizacion fuerte"),
        ("reparar", "fuerte", "reparacion"),
    ]

    for nombre, capacidad in capacidades(pesos).items():
        print("-" * 78)
        print(f"CAPACIDAD: {capacidad}  ({nombre})")
        print("-" * 78)

        _, valor_optimo = optimo_exacto(pesos, valores, capacidad)

        for estrategia, modo, etiqueta in ESTRATEGIAS:
            res = resolver(pesos, valores, capacidad, estrategia=estrategia,
                           modo_penalizacion=modo, tam_poblacion=60,
                           generaciones=120, tasa_mutacion=0.05,
                           elitismo=2, cruce="un_punto", semilla=42)
            r = resumen_solucion(res.mejor_individuo, pesos, valores, capacidad)
            brecha = 100 * (valor_optimo - r["valor"]) / valor_optimo

            print(f"  {etiqueta:<20} valor={r['valor']:>5}  peso={r['peso']:>4}/{capacidad}"
                  f"  objetos={r['objetos']:>2}  valido={'Si' if r['valido'] else 'NO'}"
                  f"  gen_mejor={res.generacion_mejor:>3}  brecha={brecha:5.2f}%")

        print(f"  {'optimo exacto (DP)':<20} valor={valor_optimo:>5}\n")
