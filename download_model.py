"""
download_model.py — Script de descarga única del modelo ColPali

Ejecutar UNA SOLA VEZ antes de arrancar el servicio por primera vez:

    python download_model.py

El modelo quedará guardado en la carpeta `models/` (o en MODEL_LOCAL_DIR
si tienes esa variable de entorno configurada). Desde ese momento el servicio
carga desde disco en cada arranque sin necesidad de internet ni de re-descargar.

Opciones:
    --force     Fuerza la re-descarga aunque el modelo ya exista en disco.
    --dir PATH  Carpeta destino (sobreescribe MODEL_LOCAL_DIR).
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga el modelo ColPali a disco.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-descarga aunque el modelo ya exista en disco.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Carpeta destino (por defecto: models/ o MODEL_LOCAL_DIR).",
    )
    args = parser.parse_args()

    # Sobreescribir directorio si se pasa por argumento
    if args.dir:
        os.environ["MODEL_LOCAL_DIR"] = args.dir

    # Importar core después de ajustar la variable de entorno
    import core

    if args.dir:
        core.MODEL_LOCAL_DIR = Path(args.dir)

    logger.info("=" * 60)
    logger.info("Destino: %s", core.MODEL_LOCAL_DIR.resolve())
    logger.info("Modelo:  %s", core.COLPALI_MODEL_NAME)
    logger.info("Processor: %s", core.COLPALI_PROCESSOR_NAME)
    logger.info("=" * 60)

    try:
        core.download_model(force=args.force)
        logger.info("✅  Descarga completada. Ya puedes iniciar el servicio con:")
        logger.info("    python app.py")
        logger.info("    (o: uvicorn app:app --host 0.0.0.0 --port 8000)")
    except KeyboardInterrupt:
        logger.warning("Descarga cancelada por el usuario.")
        sys.exit(1)
    except Exception as exc:
        logger.error("Error durante la descarga: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
