"""
EJERCICIO 2: ASIGNACION DE CURSOS A SALAS DE COMPUTO

Una institucion debe asignar 8 cursos a 4 salas y 5 franjas horarias
minimizando conflictos y penalizaciones.

REPRESENTACION
    Cromosoma = vector de 8 genes enteros (uno por curso).
    Cada gen codifica el SLOT asignado al curso:

        slot = sala * N_FRANJAS + franja        (valores 0..19)
        sala   = slot // N_FRANJAS
        franja = slot %  N_FRANJAS

    No es una permutacion: dos cursos PUEDEN recibir el mismo slot, y eso es
    justamente un choque de horario que la funcion de aptitud debe castigar.
    Como los genes son independientes, aqui si son validos los cruces clasicos
    (un punto, uniforme) y la mutacion por reasignacion.

FUNCION DE APTITUD
    Suma de penalizaciones (a MINIMIZAR). Se distinguen dos niveles:

    RESTRICCIONES DURAS (peso 1000) - hacen el horario inviable:
        * choque          : dos cursos en la misma sala y la misma franja
        * sobrecupo       : mas estudiantes que la capacidad de la sala
        * recurso         : la sala no tiene los computadores o el software
        * franja_bloqueada: el curso no puede dictarse en esa franja

    RESTRICCIONES BLANDAS (peso 1 a 10) - solo degradan la calidad:
        * desbalance : franjas con carga muy desigual
        * holgura    : sala mucho mas grande que el curso (desperdicio de recurso)

    El objetivo es llegar a penalizacion dura = 0 y luego minimizar las blandas.

DOS ESTRATEGIAS PARA LOS INDIVIDUOS INVALIDOS
    PENALIZAR : se dejan entrar a la poblacion y la aptitud los castiga.
    REPARAR   : antes de evaluarlos se los corrige moviendolos a un slot factible.
    El experimento compara ambas.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from ga.core import ConfigGA, ejecutar_ga
from ga.operadores import cruce_un_punto, cruce_uniforme, mut_reasignacion
from ga.estadisticas import ruta_dato

CRUCES = {"un_punto": cruce_un_punto, "uniforme": cruce_uniforme}


# ==========================================================================
# TABLA DE CURSOS Y TABLA DE SALAS
# ==========================================================================

FRANJAS = [
    "Lun-Mie 07:00-09:00",
    "Lun-Mie 09:00-11:00",
    "Mar-Jue 07:00-09:00",
    "Mar-Jue 09:00-11:00",
    "Vie 14:00-16:00",
]
N_FRANJAS = len(FRANJAS)

# franjas_bloqueadas: franjas en las que el curso NO puede programarse
# (docente no disponible, choque con otro programa, etc.)
CURSOS = [
    {"id": "C1", "nombre": "Inteligencia Artificial", "estudiantes": 28,
     "computadores": True,  "software": "Python",      "franjas_bloqueadas": [4]},
    {"id": "C2", "nombre": "Bases de Datos",          "estudiantes": 35,
     "computadores": True,  "software": "SQL Server",  "franjas_bloqueadas": []},
    {"id": "C3", "nombre": "Redes de Computadores",   "estudiantes": 22,
     "computadores": True,  "software": None,          "franjas_bloqueadas": [1]},
    {"id": "C4", "nombre": "Algoritmos",              "estudiantes": 40,
     "computadores": False, "software": None,          "franjas_bloqueadas": [0]},
    {"id": "C5", "nombre": "Ingenieria de Software",  "estudiantes": 30,
     "computadores": False, "software": None,          "franjas_bloqueadas": []},
    {"id": "C6", "nombre": "Machine Learning",        "estudiantes": 18,
     "computadores": True,  "software": "Python",      "franjas_bloqueadas": []},
    {"id": "C7", "nombre": "Sistemas Operativos",     "estudiantes": 25,
     "computadores": True,  "software": "Linux",       "franjas_bloqueadas": []},
    {"id": "C8", "nombre": "Calculo Diferencial",     "estudiantes": 45,
     "computadores": False, "software": None,          "franjas_bloqueadas": [3, 4]},
]

SALAS = [
    {"id": "S1", "nombre": "Sala de Computo A", "capacidad": 36, "computadores": True,
     "software": ["Python", "SQL Server", "Linux"]},
    {"id": "S2", "nombre": "Sala de Computo B", "capacidad": 28, "computadores": True,
     "software": ["Python", "Linux"]},
    {"id": "S3", "nombre": "Aula 101",          "capacidad": 45, "computadores": False,
     "software": []},
    {"id": "S4", "nombre": "Aula 202",          "capacidad": 50, "computadores": False,
     "software": []},
]

N_CURSOS = len(CURSOS)
N_SALAS = len(SALAS)
N_SLOTS = N_SALAS * N_FRANJAS

# Pesos de penalizacion: las duras valen dos ordenes de magnitud mas que las
# blandas, de modo que el algoritmo nunca sacrifica factibilidad por comodidad.
PESOS = {
    "choque": 1000,
    "sobrecupo": 1000,
    "recurso": 1000,
    "franja_bloqueada": 1000,
    "desbalance": 10,
    "holgura": 1,
}
TIPOS_DUROS = ("choque", "sobrecupo", "recurso", "franja_bloqueada")
TIPOS_BLANDOS = ("desbalance", "holgura")

HOLGURA_TOLERADA = 10  # sillas vacias que no se castigan


def tablas_dataframes():
    """Devuelve las tablas de cursos y salas como DataFrames, SIN escribir nada.

    Se separa de guardar_tablas() porque el aplicativo web solo necesita
    mostrarlas: escribir en disco en cada carga de pagina es innecesario y
    falla en plataformas de despliegue con sistema de archivos de solo lectura.
    """
    df_cursos = pd.DataFrame([{
        "id": c["id"], "curso": c["nombre"], "estudiantes": c["estudiantes"],
        "requiere_computadores": "Si" if c["computadores"] else "No",
        "software_requerido": c["software"] or "-",
        "franjas_bloqueadas": ", ".join(FRANJAS[f] for f in c["franjas_bloqueadas"]) or "-",
    } for c in CURSOS])

    df_salas = pd.DataFrame([{
        "id": s["id"], "sala": s["nombre"], "capacidad": s["capacidad"],
        "computadores": "Si" if s["computadores"] else "No",
        "software_disponible": ", ".join(s["software"]) or "-",
    } for s in SALAS])

    return df_cursos, df_salas


def guardar_tablas():
    """Exporta las tablas de cursos y salas a datos/ para incluirlas en el informe.

    La usan los scripts de terminal y los experimentos, no el aplicativo web.
    """
    df_cursos, df_salas = tablas_dataframes()
    df_cursos.to_csv(ruta_dato("cursos.csv"), index=False, encoding="utf-8-sig")
    df_salas.to_csv(ruta_dato("salas.csv"), index=False, encoding="utf-8-sig")
    return df_cursos, df_salas


# ==========================================================================
# DECODIFICACION Y FACTIBILIDAD
# ==========================================================================


def decodificar(slot):
    """slot -> (indice de sala, indice de franja)."""
    return slot // N_FRANJAS, slot % N_FRANJAS


def codificar(sala, franja):
    """(sala, franja) -> slot."""
    return sala * N_FRANJAS + franja


def slot_compatible(curso_idx, slot):
    """True si el slot no viola ninguna restriccion dura INDIVIDUAL del curso.

    No mira los choques con otros cursos: eso depende del cromosoma completo.
    """
    curso = CURSOS[curso_idx]
    sala_idx, franja = decodificar(slot)
    sala = SALAS[sala_idx]

    if franja in curso["franjas_bloqueadas"]:
        return False
    if curso["estudiantes"] > sala["capacidad"]:
        return False
    if curso["computadores"] and not sala["computadores"]:
        return False
    if curso["software"] and curso["software"] not in sala["software"]:
        return False
    return True


SLOTS_COMPATIBLES = [
    [s for s in range(N_SLOTS) if slot_compatible(i, s)] for i in range(N_CURSOS)
]


# ==========================================================================
# INSTANCIAS PARAMETRIZABLES (para el estudio de escalado)
# ==========================================================================

# Copia de la instancia base del enunciado, para poder restaurarla despues.
_BASE = {"cursos": list(CURSOS), "salas": list(SALAS), "franjas": list(FRANJAS)}


def configurar_instancia(cursos, salas, franjas):
    """Reemplaza la instancia activa y recalcula todas las tablas derivadas.

    Permite estudiar que ocurre al aumentar el numero de cursos sin duplicar
    el codigo del problema.
    """
    global CURSOS, SALAS, FRANJAS, N_CURSOS, N_SALAS, N_FRANJAS, N_SLOTS
    global SLOTS_COMPATIBLES
    CURSOS, SALAS, FRANJAS = list(cursos), list(salas), list(franjas)
    N_CURSOS, N_SALAS, N_FRANJAS = len(CURSOS), len(SALAS), len(FRANJAS)
    N_SLOTS = N_SALAS * N_FRANJAS
    SLOTS_COMPATIBLES = [
        [s for s in range(N_SLOTS) if slot_compatible(i, s)] for i in range(N_CURSOS)
    ]


def restaurar_instancia_base():
    """Vuelve a la instancia de 8 cursos, 4 salas y 5 franjas del enunciado."""
    configurar_instancia(_BASE["cursos"], _BASE["salas"], _BASE["franjas"])


def instancia_sintetica(n_cursos, semilla=7):
    """Genera n_cursos SOBRE LAS MISMAS 4 salas y 5 franjas, garantizando que
    existe al menos un horario factible.

    Se construye al reves: primero se sortea un horario valido (n_cursos slots
    distintos) y despues se inventan los requisitos de cada curso de forma que
    la sala que le toco los satisfaga. Asi, si el algoritmo no encuentra un
    horario factible, es limitacion del algoritmo y no de la instancia.
    """
    if n_cursos > len(_BASE["salas"]) * len(_BASE["franjas"]):
        raise ValueError("No hay slots suficientes para tantos cursos")

    rng = np.random.default_rng(semilla)
    salas, franjas = _BASE["salas"], _BASE["franjas"]
    n_franjas = len(franjas)

    slots = rng.choice(len(salas) * n_franjas, size=n_cursos, replace=False)
    cursos = []
    for i, slot in enumerate(slots):
        sala = salas[int(slot) // n_franjas]
        franja = int(slot) % n_franjas

        # Requisitos compatibles con la sala que le correspondio en el horario
        # de referencia, para no crear instancias imposibles.
        estudiantes = int(rng.integers(max(10, sala["capacidad"] - 20),
                                       sala["capacidad"] + 1))
        computadores = bool(sala["computadores"] and rng.random() < 0.7)
        software = (str(rng.choice(sala["software"]))
                    if computadores and sala["software"] and rng.random() < 0.6
                    else None)
        bloqueadas = [f for f in range(n_franjas)
                      if f != franja and rng.random() < 0.2]

        cursos.append({
            "id": f"C{i + 1}", "nombre": f"Curso {i + 1}",
            "estudiantes": estudiantes, "computadores": computadores,
            "software": software, "franjas_bloqueadas": bloqueadas,
        })

    return cursos, salas, franjas


# ==========================================================================
# FUNCION DE APTITUD
# ==========================================================================


def detalle_penalizaciones(cromosoma):
    """Descompone la penalizacion por tipo de restriccion.

    Devuelve un diccionario con el numero de violaciones de cada tipo y el
    costo total, separando duras de blandas.
    """
    cromosoma = np.asarray(cromosoma, dtype=int)
    conteo = {tipo: 0 for tipo in TIPOS_DUROS + TIPOS_BLANDOS}
    # (el desbalance se mide como una cantidad continua, no como un conteo)

    # --- Restricciones duras que dependen de cada curso por separado ---
    for i, slot in enumerate(cromosoma):
        curso = CURSOS[i]
        sala_idx, franja = decodificar(int(slot))
        sala = SALAS[sala_idx]

        if franja in curso["franjas_bloqueadas"]:
            conteo["franja_bloqueada"] += 1
        if curso["estudiantes"] > sala["capacidad"]:
            conteo["sobrecupo"] += 1
        if curso["computadores"] and not sala["computadores"]:
            conteo["recurso"] += 1
        elif curso["software"] and curso["software"] not in sala["software"]:
            conteo["recurso"] += 1

        # --- Restriccion blanda: sillas desperdiciadas ---
        holgura = sala["capacidad"] - curso["estudiantes"]
        if holgura > HOLGURA_TOLERADA:
            conteo["holgura"] += holgura - HOLGURA_TOLERADA

    # --- Restriccion dura: dos cursos en la misma sala y franja ---
    ocupacion = np.bincount(cromosoma, minlength=N_SLOTS)
    conteo["choque"] = int((ocupacion * (ocupacion - 1) // 2).sum())

    # --- Restriccion blanda: distribucion desigual entre franjas ---
    # Se mide como la desviacion respecto a la carga media por franja.
    carga_franja = np.bincount(cromosoma % N_FRANJAS, minlength=N_FRANJAS)
    conteo["desbalance"] = round(float(np.abs(carga_franja - N_CURSOS / N_FRANJAS).sum()), 2)

    costo_duro = sum(conteo[t] * PESOS[t] for t in TIPOS_DUROS)
    costo_blando = sum(conteo[t] * PESOS[t] for t in TIPOS_BLANDOS)

    return {
        "violaciones": conteo,
        "costo_duro": costo_duro,
        "costo_blando": costo_blando,
        "costo_total": costo_duro + costo_blando,
        "factible": costo_duro == 0,
    }


def penalizacion(cromosoma):
    """Penalizacion total del horario (valor a MINIMIZAR)."""
    return detalle_penalizaciones(cromosoma)["costo_total"]


def aptitud(cromosoma):
    """Aptitud que maximiza el motor: el negativo de la penalizacion total."""
    return -penalizacion(cromosoma)


# ==========================================================================
# ESTRATEGIA DE REPARACION
# ==========================================================================


def reparar(cromosoma):
    """Convierte un horario invalido en uno factible.

    Recorre los cursos y, cuando el slot asignado es incompatible o ya esta
    ocupado, lo mueve al primer slot libre que si cumple sus requisitos. Si
    ninguno esta libre, deja el curso donde estaba y la aptitud lo penaliza.
    """
    cromosoma = np.asarray(cromosoma, dtype=int).copy()
    ocupados = set()

    # Se atienden primero los cursos mas restringidos (menos slots compatibles),
    # que es la heuristica habitual para no dejarlos sin opciones al final.
    orden = sorted(range(N_CURSOS), key=lambda i: len(SLOTS_COMPATIBLES[i]))

    for i in orden:
        slot = int(cromosoma[i])
        if slot in SLOTS_COMPATIBLES[i] and slot not in ocupados:
            ocupados.add(slot)
            continue
        for candidato in SLOTS_COMPATIBLES[i]:
            if candidato not in ocupados:
                cromosoma[i] = candidato
                ocupados.add(candidato)
                break
        else:
            ocupados.add(slot)
    return cromosoma


# ==========================================================================
# CONFIGURACION DEL ALGORITMO GENETICO
# ==========================================================================


def construir_config(tam_poblacion=100, generaciones=300, tasa_mutacion=0.1,
                     elitismo=2, cruce="uniforme", seleccion="torneo",
                     estrategia="penalizar", semilla=None):
    """Arma la ConfigGA para la asignacion de cursos.

    estrategia: "penalizar" (los invalidos entran y se castigan) o
                "reparar"   (se corrigen antes de evaluarlos).
    """
    if estrategia not in ("penalizar", "reparar"):
        raise ValueError("estrategia debe ser 'penalizar' o 'reparar'")

    return ConfigGA(
        crear_individuo=lambda rng: rng.integers(0, N_SLOTS, size=N_CURSOS),
        aptitud=aptitud,
        cruce=CRUCES[cruce],
        mutacion=lambda ind, tasa, rng: mut_reasignacion(ind, tasa, rng, N_SLOTS),
        tam_poblacion=tam_poblacion,
        generaciones=generaciones,
        tasa_mutacion=tasa_mutacion,
        elitismo=elitismo,
        seleccion=seleccion,
        reparar=reparar if estrategia == "reparar" else None,
        max_estancamiento=80,
        semilla=semilla,
    )


def resolver(**kwargs):
    """Ejecuta una corrida del GA para la asignacion de cursos."""
    return ejecutar_ga(construir_config(**kwargs))


# ==========================================================================
# PRESENTACION DEL HORARIO
# ==========================================================================


def horario_dataframe(cromosoma):
    """Horario final como tabla salas x franjas."""
    tabla = pd.DataFrame("", index=[s["nombre"] for s in SALAS], columns=FRANJAS)
    for i, slot in enumerate(np.asarray(cromosoma, dtype=int)):
        sala_idx, franja = decodificar(int(slot))
        curso = CURSOS[i]
        celda = f"{curso['id']} {curso['nombre']} ({curso['estudiantes']})"
        actual = tabla.iloc[sala_idx, franja]
        tabla.iloc[sala_idx, franja] = f"{actual} // {celda}" if actual else celda
    return tabla


def asignacion_dataframe(cromosoma):
    """Detalle curso a curso de la asignacion, con el chequeo de cada restriccion."""
    filas = []
    ocupacion = np.bincount(np.asarray(cromosoma, dtype=int), minlength=N_SLOTS)
    for i, slot in enumerate(np.asarray(cromosoma, dtype=int)):
        sala_idx, franja = decodificar(int(slot))
        curso, sala = CURSOS[i], SALAS[sala_idx]
        filas.append({
            "curso": f"{curso['id']} - {curso['nombre']}",
            "estudiantes": curso["estudiantes"],
            "sala": sala["nombre"],
            "capacidad": sala["capacidad"],
            "franja": FRANJAS[franja],
            "sobrecupo": "Si" if curso["estudiantes"] > sala["capacidad"] else "No",
            "recursos_ok": "No" if (
                (curso["computadores"] and not sala["computadores"]) or
                (curso["software"] and curso["software"] not in sala["software"])
            ) else "Si",
            "franja_permitida": "No" if franja in curso["franjas_bloqueadas"] else "Si",
            "comparte_slot": "Si" if ocupacion[slot] > 1 else "No",
        })
    return pd.DataFrame(filas)


# ==========================================================================
# EJECUCION DIRECTA
# ==========================================================================

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 40)

    guardar_tablas()
    print("Tablas de cursos y salas guardadas en datos/\n")

    res = resolver(tam_poblacion=100, generaciones=300, tasa_mutacion=0.1,
                   elitismo=2, cruce="uniforme", estrategia="penalizar", semilla=42)
    detalle = detalle_penalizaciones(res.mejor_individuo)

    print("=" * 78)
    print("ASIGNACION DE CURSOS A SALAS")
    print("=" * 78)
    print(f"Cromosoma        : {[int(g) for g in res.mejor_individuo]}")
    print(f"Penalizacion dura: {detalle['costo_duro']}  (0 = horario factible)")
    print(f"Penalizacion blanda: {detalle['costo_blando']}")
    print(f"Penalizacion total : {detalle['costo_total']}")
    print(f"Horario factible   : {'SI' if detalle['factible'] else 'NO'}")
    print(f"Generacion de la mejor solucion: {res.generacion_mejor}")
    print(f"\nViolaciones por tipo: {detalle['violaciones']}\n")

    print("DETALLE DE LA ASIGNACION")
    print(asignacion_dataframe(res.mejor_individuo).to_string(index=False))

    print("\nHORARIO FINAL (salas x franjas)")
    print(horario_dataframe(res.mejor_individuo).to_string())
