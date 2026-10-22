"""Prueba imagenes locales con el modelo final de 8 clases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


CLASES = [
    "bosque",
    "cascada",
    "desierto",
    "lago",
    "montana",
    "nevado",
    "playa",
    "volcan",
]
CARPETA_IMAGENES = Path(__file__).resolve().parent / "imagenes_prueba"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def buscar_imagenes(argumento: Path | None) -> list[Path]:
    """Resuelve una imagen indicada o todas las disponibles en la carpeta."""
    if argumento is None:
        imagenes = sorted(
            archivo
            for archivo in CARPETA_IMAGENES.iterdir()
            if archivo.is_file() and archivo.suffix.lower() in EXTENSIONES
        )
        if not imagenes:
            raise FileNotFoundError(
                f"No hay imagenes para probar en: {CARPETA_IMAGENES}"
            )
        return imagenes

    if argumento.is_absolute():
        ruta = argumento
    else:
        ruta_carpeta = CARPETA_IMAGENES / argumento
        ruta = ruta_carpeta if ruta_carpeta.is_file() else argumento
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No existe la imagen: {ruta}. Colocala en {CARPETA_IMAGENES}"
        )
    return [ruta]


def clasificar(
    modelo: keras.Model, ruta: Path, top: int, umbral_revision: float
) -> dict:
    with Image.open(ruta) as original:
        imagen = original.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
    entrada = preprocess_input(np.asarray(imagen, dtype=np.float32))[None, ...]
    probabilidades = np.asarray(modelo.predict(entrada, verbose=0)[0], dtype=float)
    orden = np.argsort(probabilidades)[::-1][: max(1, min(top, len(CLASES)))]
    return {
        "imagen": str(ruta.resolve()),
        "clase_predicha": CLASES[int(orden[0])],
        "confianza": round(float(probabilidades[orden[0]]) * 100, 2),
        "top_predicciones": [
            {
                "clase": CLASES[int(indice)],
                "probabilidad_pct": round(float(probabilidades[indice]) * 100, 2),
            }
            for indice in orden
        ],
        "requiere_revision_por_umbral": bool(
            probabilidades[orden[0]] < umbral_revision
        ),
        "umbral_revision": umbral_revision,
        "nota_confianza": (
            "La probabilidad del softmax no garantiza que la prediccion sea correcta; "
            "el test externo mostro sobreconfianza."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clasifica imagenes locales.")
    parser.add_argument(
        "imagen",
        type=Path,
        nargs="?",
        help=(
            "Nombre dentro de imagenes_prueba o ruta completa. "
            "Si se omite, procesa todas las imagenes de la carpeta."
        ),
    )
    parser.add_argument(
        "--modelo",
        type=Path,
        default=Path(
            r"D:\proyecto_turismo_ia\entrenamiento_mobilenetv2_v2"
            r"\modelo_mobilenetv2_limpio_final.keras"
        ),
    )
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument(
        "--umbral-revision",
        type=float,
        default=0.90,
        help="Marca para revision predicciones con confianza menor a este valor.",
    )
    args = parser.parse_args()

    if not args.modelo.is_file():
        raise FileNotFoundError(f"No existe el modelo: {args.modelo}")

    imagenes = buscar_imagenes(args.imagen)
    modelo = keras.models.load_model(args.modelo)
    resultados = [
        clasificar(modelo, ruta, args.top, args.umbral_revision) for ruta in imagenes
    ]
    salida = resultados[0] if len(resultados) == 1 else {
        "carpeta": str(CARPETA_IMAGENES),
        "cantidad": len(resultados),
        "resultados": resultados,
    }
    print(json.dumps(salida, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
