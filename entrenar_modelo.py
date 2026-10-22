"""Entrenamiento reproducible de MobileNetV2 para el motor de turismo.

Este script utiliza solamente ``train`` y ``val``. Los conjuntos ``test`` y
``test_externo_places365`` quedan reservados para la evaluación final.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input


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
    parser = argparse.ArgumentParser(
        description="Entrena MobileNetV2 con el dataset limpio sin tocar los tests finales."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"D:\proyecto_turismo_ia\dataset_limpio_v2"),
        help="Carpeta que contiene train/ y val/.",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=Path(r"D:\proyecto_turismo_ia\entrenamiento_mobilenetv2_v2"),
        help="Carpeta donde se guardarán modelos, métricas y logs.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epocas-cabeza", type=int, default=12)
    parser.add_argument("--epocas-finetuning", type=int, default=25)
    parser.add_argument("--capas-descongeladas", type=int, default=40)
    parser.add_argument("--lr-cabeza", type=float, default=1e-3)
    parser.add_argument("--lr-finetuning", type=float, default=1e-5)
    parser.add_argument("--paciencia", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sin-pesos-imagenet",
        action="store_true",
        help="Inicializa MobileNetV2 al azar. No se recomienda para este dataset.",
    )
    parser.add_argument(
        "--solo-verificar",
        action="store_true",
        help="Valida carpetas, clases y conteos sin construir ni entrenar el modelo.",
    )
    return parser.parse_args()


def contar_imagenes(carpeta: Path) -> dict[str, int]:
    return {
        clase: sum(
            1
            for archivo in (carpeta / clase).rglob("*")
            if archivo.is_file() and archivo.suffix.lower() in EXTENSIONES
        )
        for clase in CLASES
    }


def validar_dataset(dataset: Path) -> dict[str, dict[str, int]]:
    if not dataset.is_dir():
        raise FileNotFoundError(f"No existe el dataset: {dataset}")

    conteos: dict[str, dict[str, int]] = {}
    for split in ("train", "val"):
        carpeta = dataset / split
        if not carpeta.is_dir():
            raise FileNotFoundError(f"Falta la carpeta requerida: {carpeta}")
        encontradas = sorted(ruta.name for ruta in carpeta.iterdir() if ruta.is_dir())
        faltantes = sorted(set(CLASES) - set(encontradas))
        inesperadas = sorted(set(encontradas) - set(CLASES))
        if faltantes or inesperadas:
            raise ValueError(
                f"Clases inválidas en {split}. Faltantes={faltantes}; inesperadas={inesperadas}"
            )
        conteos[split] = contar_imagenes(carpeta)
        vacias = [clase for clase, cantidad in conteos[split].items() if cantidad == 0]
        if vacias:
            raise ValueError(f"Clases vacías en {split}: {vacias}")
    return conteos


def configurar_entorno(seed: int) -> list[str]:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    dispositivos = [dispositivo.name for dispositivo in tf.config.list_physical_devices("GPU")]
    if dispositivos:
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        keras.mixed_precision.set_global_policy("mixed_float16")
    return dispositivos


def crear_datasets(dataset: Path, batch_size: int, seed: int):
    comunes = {
        "labels": "inferred",
        "label_mode": "int",
        "class_names": CLASES,
        "image_size": (224, 224),
        "batch_size": batch_size,
    }
    train = keras.utils.image_dataset_from_directory(
        dataset / "train", shuffle=True, seed=seed, **comunes
    )
    val = keras.utils.image_dataset_from_directory(
        dataset / "val", shuffle=False, **comunes
    )

    aumento = keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(0.05, fill_mode="reflect", seed=seed + 1),
            layers.RandomZoom(0.15, fill_mode="reflect", seed=seed + 2),
            layers.RandomTranslation(0.08, 0.08, fill_mode="reflect", seed=seed + 3),
            layers.RandomContrast(0.15, seed=seed + 4),
        ],
        name="aumento_datos",
    )

    autotune = tf.data.AUTOTUNE

    def preparar_train(imagenes, etiquetas):
        imagenes = aumento(imagenes, training=True)
        return preprocess_input(imagenes), etiquetas

    def preparar_val(imagenes, etiquetas):
        return preprocess_input(imagenes), etiquetas

    train = train.map(preparar_train, num_parallel_calls=autotune).prefetch(autotune)
    val = val.map(preparar_val, num_parallel_calls=autotune).prefetch(autotune)
    return train, val


def pesos_clase(conteos_train: dict[str, int]) -> dict[int, float]:
    etiquetas = np.concatenate(
        [np.full(conteos_train[clase], indice) for indice, clase in enumerate(CLASES)]
    )
    pesos = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(CLASES)), y=etiquetas
    )
    return {indice: float(peso) for indice, peso in enumerate(pesos)}


def crear_modelo(pesos_imagenet: bool):
    base = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet" if pesos_imagenet else None,
    )
    base.trainable = False

    entradas = keras.Input(shape=(224, 224, 3), name="imagen_preprocesada")
    x = base(entradas, training=False)
    x = layers.GlobalAveragePooling2D(name="embedding_pool")(x)
    x = layers.Dropout(0.35, name="dropout_pool")(x)
    x = layers.Dense(256, activation="relu", name="embedding_dense")(x)
    x = layers.Dropout(0.30, name="dropout_clasificador")(x)
    salidas = layers.Dense(
        len(CLASES), activation="softmax", dtype="float32", name="clasificador"
    )(x)
    return keras.Model(entradas, salidas, name="motor_turismo_mobilenetv2"), base


def compilar(modelo: keras.Model, learning_rate: float) -> None:
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def callbacks(salida: Path, fase: str, paciencia: int):
    lista = [
        keras.callbacks.ModelCheckpoint(
            salida / f"mejor_{fase}.keras",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=paciencia,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.3,
            patience=max(2, paciencia // 2),
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(salida / f"historial_{fase}.csv"),
        keras.callbacks.TerminateOnNaN(),
    ]
    if importlib.util.find_spec("tensorboard") is not None:
        lista.insert(
            -1,
            keras.callbacks.TensorBoard(log_dir=salida / "tensorboard" / fase),
        )
    else:
        print("TensorBoard no está instalado; se conservarán historiales CSV y checkpoints.")
    return lista


def descongelar(base: keras.Model, ultimas_capas: int) -> None:
    base.trainable = True
    limite = max(0, len(base.layers) - ultimas_capas)
    for indice, capa in enumerate(base.layers):
        capa.trainable = indice >= limite and not isinstance(capa, layers.BatchNormalization)


def evaluar(modelo_path: Path, val) -> dict[str, float]:
    modelo = keras.models.load_model(modelo_path)
    valores = modelo.evaluate(val, verbose=0, return_dict=True)
    return {clave: float(valor) for clave, valor in valores.items()}


def main() -> None:
    args = argumentos()
    dataset = args.dataset.resolve()
    salida = args.salida.resolve()
    conteos = validar_dataset(dataset)

    print("Clases (orden de salida):", CLASES)
    print("Conteos:", json.dumps(conteos, ensure_ascii=False, indent=2))
    if args.solo_verificar:
        print("Verificación completada. No se entrenó ningún modelo.")
        return

    if args.epocas_cabeza < 1 or args.epocas_finetuning < 1:
        raise ValueError("Las épocas de ambas fases deben ser mayores que cero.")
    if not 1 <= args.capas_descongeladas <= 154:
        raise ValueError("--capas-descongeladas debe estar entre 1 y 154.")

    salida.mkdir(parents=True, exist_ok=True)
    dispositivos_gpu = configurar_entorno(args.seed)
    print("GPU detectada por TensorFlow:", dispositivos_gpu or "ninguna; se usará CPU")

    train, val = crear_datasets(dataset, args.batch_size, args.seed)
    class_weights = pesos_clase(conteos["train"])
    modelo, base = crear_modelo(not args.sin_pesos_imagenet)

    print("\nFASE 1: entrenamiento de la cabeza clasificadora")
    compilar(modelo, args.lr_cabeza)
    modelo.fit(
        train,
        validation_data=val,
        epochs=args.epocas_cabeza,
        class_weight=class_weights,
        callbacks=callbacks(salida, "cabeza", args.paciencia),
    )

    print("\nFASE 2: fine-tuning controlado")
    descongelar(base, args.capas_descongeladas)
    compilar(modelo, args.lr_finetuning)
    modelo.fit(
        train,
        validation_data=val,
        epochs=args.epocas_finetuning,
        class_weight=class_weights,
        callbacks=callbacks(salida, "finetuning", args.paciencia),
    )

    candidatos = [salida / "mejor_cabeza.keras", salida / "mejor_finetuning.keras"]
    metricas = {ruta.name: evaluar(ruta, val) for ruta in candidatos}
    ganador = max(candidatos, key=lambda ruta: metricas[ruta.name]["accuracy"])
    modelo_final = salida / "modelo_mobilenetv2_limpio_final.keras"
    shutil.copy2(ganador, modelo_final)

    metadatos = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "salida": str(salida),
        "clases": CLASES,
        "conteos": conteos,
        "pesos_clase": class_weights,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epocas_cabeza_max": args.epocas_cabeza,
        "epocas_finetuning_max": args.epocas_finetuning,
        "capas_descongeladas": args.capas_descongeladas,
        "lr_cabeza": args.lr_cabeza,
        "lr_finetuning": args.lr_finetuning,
        "tensorflow": tf.__version__,
        "gpu_tensorflow": dispositivos_gpu,
        "metricas_validacion_candidatos": metricas,
        "modelo_seleccionado": ganador.name,
        "modelo_final": modelo_final.name,
        "test_usado_durante_entrenamiento": False,
    }
    (salida / "metadatos_entrenamiento.json").write_text(
        json.dumps(metadatos, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (salida / "clases.json").write_text(
        json.dumps(CLASES, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nEntrenamiento terminado")
    print("Modelo seleccionado:", ganador)
    print("Modelo final:", modelo_final)
    print("Métricas de validación:", json.dumps(metricas, indent=2))
    print("Los conjuntos test no fueron utilizados.")


if __name__ == "__main__":
    main()
