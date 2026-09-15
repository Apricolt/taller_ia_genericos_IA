"""
Genera informe/informe_taller1.docx a partir de informe/informe_taller1.md.

    python informe/construir_docx.py

Usa pandoc, que convierte las tablas de Markdown en tablas NATIVAS de Word e
incrusta las imagenes referenciadas. Se ejecuta desde la carpeta informe/ para
que las rutas relativas (../resultados/...) de las figuras resuelvan bien.
"""

import os
import shutil
import subprocess
import sys

INFORME = os.path.dirname(os.path.abspath(__file__))
FUENTE = "informe_taller1.md"
DESTINO = "informe_taller1.docx"


def main():
    if shutil.which("pandoc") is None:
        sys.exit("pandoc no esta instalado o no esta en el PATH.")

    comando = [
        "pandoc", FUENTE,
        "-o", DESTINO,
        "--from", "markdown+pipe_tables+tex_math_dollars",
        "--toc",                    # tabla de contenido
        "--toc-depth=2",
        "-V", "lang=es-CO",
        "--resource-path=.:..",     # para resolver ../resultados/*.png
    ]

    resultado = subprocess.run(comando, cwd=INFORME, capture_output=True, text=True)
    if resultado.returncode != 0:
        sys.exit(f"pandoc fallo:\n{resultado.stderr}")
    if resultado.stderr.strip():
        print("Avisos de pandoc:")
        print(resultado.stderr.strip())

    ruta = os.path.join(INFORME, DESTINO)
    print(f"Generado: {ruta}  ({os.path.getsize(ruta) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
