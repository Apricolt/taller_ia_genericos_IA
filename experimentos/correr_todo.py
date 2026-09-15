"""Ejecuta los cuatro conjuntos de experimentos y deja la evidencia en resultados/.

    python experimentos/correr_todo.py
"""

import os
import runpy
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

GUIONES = ["exp_nreinas.py", "exp_tsp.py", "exp_asignacion.py", "exp_mochila.py"]

if __name__ == "__main__":
    inicio = time.perf_counter()
    for guion in GUIONES:
        print("\n" + "#" * 78)
        print(f"# {guion}")
        print("#" * 78)
        runpy.run_path(os.path.join(RAIZ, "experimentos", guion), run_name="__main__")
    print(f"\nTodos los experimentos completados en "
          f"{time.perf_counter() - inicio:.1f} segundos.")
