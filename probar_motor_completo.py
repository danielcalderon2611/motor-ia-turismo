"""Prueba local del motor CNN + embeddings + Gemma + GIS sin iniciar Uvicorn."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


CARPETA_IMAGENES = Path(__file__).resolve().parent / "imagenes_prueba"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prueba el motor multimodal completo.")
    parser.add_argument(
        "imagen",
        type=Path,
        nargs="?",
        help="Ruta o nombre dentro de imagenes_prueba. Si se omite debe existir una sola imagen.",
    )
    parser.add_argument("--lat", type=float, required=True, help="Latitud GPS")
    parser.add_argument("--lon", type=float, required=True, help="Longitud GPS")
    parser.add_argument("--alt", type=float, required=True, help="Altitud en metros")
    parser.add_argument(
        "--sin-gemma",
        action="store_true",
        help="Prueba CNN, embeddings y GIS sin consultar Ollama.",
    )
    return parser.parse_args()


def resolver_imagen(argumento: Path | None) -> Path:
    if argumento is None:
        candidatas = sorted(
            ruta
            for ruta in CARPETA_IMAGENES.iterdir()
            if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES
        )
        if len(candidatas) != 1:
            raise ValueError(
                f"Debe haber exactamente una imagen en {CARPETA_IMAGENES}; "
                f"se encontraron {len(candidatas)}"
            )
        return candidatas[0]
    if argumento.is_absolute():
        ruta = argumento
    else:
        candidata = CARPETA_IMAGENES / argumento
        ruta = candidata if candidata.is_file() else argumento
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe la imagen: {ruta}")
    return ruta


def main() -> None:
    args = argumentos()
    ruta = resolver_imagen(args.imagen)
    if args.sin_gemma:
        os.environ["GEMMA_ENABLED"] = "false"

    # La importación carga el modelo, embeddings y GIS después de configurar Gemma.
    import api

    resultado = api.procesar_imagen(
        ruta.read_bytes(), ruta.name, args.lat, args.lon, args.alt
    )
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
