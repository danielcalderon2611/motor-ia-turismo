"""Genera una base vectorial compatible con el modelo CNN de ocho clases."""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
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
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def argumentos() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Regenera embeddings del motor turístico.")
    parser.add_argument(
        "--modelo",
        type=Path,
        default=base / "models" / "modelo_mobilenetv2_limpio_final.keras",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"D:\proyecto_turismo_ia\dataset_limpio_v2\train"),
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=base / "fewshot_db" / "embeddings_v2.pkl",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def inventario(dataset: Path) -> tuple[list[Path], list[str], list[str]]:
    rutas: list[Path] = []
    etiquetas: list[str] = []
    relativas: list[str] = []
    encontradas = sorted(p.name for p in dataset.iterdir() if p.is_dir())
    if encontradas != sorted(CLASES):
        raise ValueError(
            f"Clases incompatibles. Esperadas={sorted(CLASES)}, encontradas={encontradas}"
        )
    for clase in CLASES:
        for ruta in sorted((dataset / clase).rglob("*")):
            if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES:
                rutas.append(ruta)
                etiquetas.append(clase)
                relativas.append(str(ruta.relative_to(dataset)))
    if not rutas:
        raise ValueError(f"No se encontraron imágenes en {dataset}")
    return rutas, etiquetas, relativas


def cargar(ruta: tf.Tensor) -> tf.Tensor:
    contenido = tf.io.read_file(ruta)
    imagen = tf.io.decode_image(contenido, channels=3, expand_animations=False)
    imagen.set_shape([None, None, 3])
    imagen = tf.image.resize(imagen, (224, 224), antialias=True)
    return preprocess_input(tf.cast(imagen, tf.float32))


def main() -> None:
    args = argumentos()
    if not args.modelo.is_file():
        raise FileNotFoundError(f"No existe el modelo: {args.modelo}")
    if not args.dataset.is_dir():
        raise FileNotFoundError(f"No existe el dataset: {args.dataset}")

    rutas, etiquetas, relativas = inventario(args.dataset)
    print(f"Imágenes encontradas: {len(rutas)}")
    modelo = keras.models.load_model(args.modelo)
    try:
        capa = modelo.get_layer("embedding_dense")
    except ValueError as error:
        raise ValueError("El modelo no contiene la capa embedding_dense") from error
    extractor = keras.Model(modelo.inputs, capa.output, name="extractor_embeddings")

    dataset = tf.data.Dataset.from_tensor_slices([str(ruta) for ruta in rutas])
    dataset = dataset.map(cargar, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    vectores = np.asarray(extractor.predict(dataset, verbose=1), dtype=np.float32)
    normas = np.linalg.norm(vectores, axis=1, keepdims=True)
    vectores = vectores / np.maximum(normas, 1e-12)

    paquete = {
        "version": 2,
        "clases": CLASES,
        "capa": "embedding_dense",
        "dimension": int(vectores.shape[1]),
        "modelo_sha256": sha256(args.modelo),
        "dataset_origen": str(args.dataset),
        "embeddings": vectores,
        "labels": etiquetas,
        "files": [ruta.name for ruta in rutas],
        "paths": relativas,
    }
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    temporal = args.salida.with_suffix(args.salida.suffix + ".tmp")
    with temporal.open("wb") as archivo:
        pickle.dump(paquete, archivo, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporal, args.salida)
    print(f"Embeddings: {vectores.shape}")
    print(f"Guardado en: {args.salida}")
    print(f"SHA256 modelo: {paquete['modelo_sha256']}")


if __name__ == "__main__":
    main()
