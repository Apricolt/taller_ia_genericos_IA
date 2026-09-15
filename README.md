# Taller 1 - Algoritmos Evolutivos / Algoritmos Geneticos


# Integrantes
- Javier Alejandro Ordoñez
- River Alejandro Bonilla
- Diego Alejandro Ocampo
- Tomas Benavides
- Dylan Santiago Rodriguez

Inteligencia Artificial. Implementacion, experimentacion y analisis de algoritmos
geneticos sobre cuatro problemas de optimizacion combinatoria.

## Que hay aqui

Un **unico motor genetico generacional con elitismo** (`ga/core.py`) resuelve los
cuatro problemas del taller. Lo unico que cambia entre ellos es la
**representacion** del cromosoma, la **funcion de aptitud** y los **operadores**
compatibles con esa representacion; el bucle evolutivo es identico. Esa decision
es la que hace comparables los resultados entre problemas.

| Problema | Representacion | Aptitud | Operadores |
|---|---|---|---|
| N-Reinas (base) | Permutacion de N | Minimizar conflictos (optimo 0) | OX / PMX + intercambio |
| Agente viajero (Ej. 1) | Permutacion de ciudades | `1/(distancia+eps)` del ciclo | OX / PMX + inversion, intercambio, insercion |
| Asignacion de cursos (Ej. 2) | Vector entero (slot sala-franja) | Minimizar penalizacion dura + blanda | Un punto / uniforme + reasignacion |
| Mochila (Ej. 3) | Vector binario | Maximizar valor, con penalizacion o reparacion | Un punto / uniforme + bit-flip |

## Instalacion

```
pip install -r requirements.txt
```

## Uso

Cada problema es ejecutable por separado:

```
python problemas/nreinas.py       # N=6 y N=8: tablero, convergencia
python problemas/tsp.py           # 10 ciudades: ruta y brecha vs optimo exacto
python problemas/asignacion.py    # horario factible con verificacion curso a curso
python problemas/mochila.py       # 2 capacidades x 3 estrategias vs optimo por DP
```

Los barridos experimentales que generan las tablas y graficas del informe:

```
python experimentos/correr_todo.py    # los cuatro, ~5,5 minutos
```

o uno a uno: `exp_nreinas.py`, `exp_tsp.py`, `exp_asignacion.py`, `exp_mochila.py`.

Aplicativo web (una interfaz distinta por problema):

```
streamlit run web/Inicio.py
```

## Estructura

```
ga/            Motor compartido: core, seleccion, operadores, estadisticas
problemas/     Representacion + aptitud de cada problema (ejecutables sueltos)
experimentos/  Barridos que producen la evidencia
datos/         Instancias con semilla fija (matrices de distancia, cursos, salas, objetos)
resultados/    20 CSV y 22 PNG generados por los experimentos
informe/       Informe del taller
web/           Aplicativo Streamlit: Inicio.py + pages/ (una pagina por problema)
```

## Reproducibilidad

Todas las instancias usan semilla fija (`SEMILLA_DATOS = 2024`) y la corrida *i*
de cada experimento usa la semilla *i*. Volver a ejecutar `correr_todo.py`
reproduce exactamente las tablas del informe.

## Despliegue

El aplicativo esta hecho con Streamlit, asi que la via mas directa es
**Streamlit Community Cloud**: subir el proyecto a un repositorio de GitHub,
entrar en share.streamlit.io, conectar el repositorio y apuntar a
`web/Inicio.py` como archivo principal. `requirements.txt` ya esta en la raiz,
que es donde lo busca la plataforma.

La carpeta `despliegue/` contiene notas de una alternativa distinta
(frontend en Vercel + API Flask en Render). Esa ruta exige escribir un backend
HTTP que no forma parte de este proyecto; no esta en uso y ningun archivo del
codigo la referencia.
