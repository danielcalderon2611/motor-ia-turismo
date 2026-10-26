"""Entrenamiento version 2 de MobileNetV2 con validacion cruzada.

Este programa compara Stratified K-Fold con K=3 y K=5. Las carpetas
``train`` y ``val`` forman el conjunto de desarrollo. La carpeta ``test``
permanece separada hasta que se selecciona la configuracion mediante K-Fold.

No se aplica SMOTE: para las imagenes se mantienen aumento de datos y pesos
de clase. Cada fold comienza nuevamente desde los pesos de ImageNet.
"""

from __future__ import annotations

# ============================================================
# BLOQUE 1: Importacion de librerias
# ============================================================

import argparse
import csv
import gc
import json
import os
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input


# ============================================================
# BLOQUE 2: Configuracion general y parametros de terminal
# ============================================================

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
NOMBRES_GRAFICO = [
    "Bosque",
    "Cascada",
    "Desierto",
    "Lago",
    "Montaña",
    "Nevado",
    "Playa",
    "Volcán",
]
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
COLORES_ROC = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
]


def argumentos() -> argparse.Namespace:
    """Define todos los parametros configurables desde la terminal."""

    parser = argparse.ArgumentParser(
        description="Entrena MobileNetV2 con Stratified K-Fold y curvas ROC."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"D:\proyecto_turismo_ia\dataset_limpio_v2"),
        help="Carpeta que contiene train/, val/ y test/.",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=Path(r"D:\proyecto_turismo_ia\entrenamiento_mobilenetv2_kfold_v2"),
        help="Carpeta independiente para todos los resultados de la version 2.",
    )
    parser.add_argument("--k", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epocas-cabeza", type=int, default=12)
    parser.add_argument("--epocas-finetuning", type=int, default=25)
    parser.add_argument("--capas-descongeladas", type=int, default=40)
    parser.add_argument("--lr-cabeza", type=float, default=1e-3)
    parser.add_argument("--lr-finetuning", type=float, default=1e-5)
    parser.add_argument("--paciencia", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reanudar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reutiliza folds finalizados para continuar una ejecucion interrumpida.",
    )
    parser.add_argument(
        "--solo-verificar",
        action="store_true",
        help="Valida el dataset y la configuracion sin entrenar.",
    )
    parser.add_argument(
        "--sin-modelo-final",
        action="store_true",
        help="Ejecuta K-Fold, pero omite el entrenamiento final y el test interno.",
    )
    return parser.parse_args()


# ============================================================
# BLOQUE 3: Semillas, reproducibilidad y deteccion de GPU
# ============================================================


def configurar_entorno(seed: int) -> list[str]:
    """Fija semillas y activa opciones reproducibles y de GPU cuando existen."""

    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    dispositivos = [d.name for d in tf.config.list_physical_devices("GPU")]
    if dispositivos:
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        keras.mixed_precision.set_global_policy("float32")
    return dispositivos


# ============================================================
# BLOQUE 4: Lectura y validacion del dataset
# ============================================================


def listar_imagenes(carpeta: Path) -> tuple[list[Path], np.ndarray]:
    """Obtiene rutas y etiquetas siguiendo el orden oficial de las ocho clases."""

    if not carpeta.is_dir():
        raise FileNotFoundError(f"No existe la carpeta requerida: {carpeta}")
    encontradas = sorted(ruta.name for ruta in carpeta.iterdir() if ruta.is_dir())
    if encontradas != sorted(CLASES):
        raise ValueError(
            f"Clases invalidas en {carpeta}. "
            f"Esperadas={sorted(CLASES)}; encontradas={encontradas}"
        )

    archivos: list[Path] = []
    etiquetas: list[int] = []
    for indice, clase in enumerate(CLASES):
        encontrados = [
            ruta
            for ruta in sorted((carpeta / clase).rglob("*"))
            if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES
        ]
        if not encontrados:
            raise ValueError(f"La clase {clase} esta vacia en {carpeta}")
        archivos.extend(encontrados)
        etiquetas.extend([indice] * len(encontrados))
    return archivos, np.asarray(etiquetas, dtype=np.int64)


def preparar_inventario(dataset: Path) -> dict:
    """Combina train+val para K-Fold y conserva test de forma independiente."""

    if not dataset.is_dir():
        raise FileNotFoundError(f"No existe el dataset: {dataset}")
    train_x, train_y = listar_imagenes(dataset / "train")
    val_x, val_y = listar_imagenes(dataset / "val")
    test_x, test_y = listar_imagenes(dataset / "test")
    desarrollo_x = train_x + val_x
    desarrollo_y = np.concatenate([train_y, val_y])

    rutas_resueltas = [str(ruta.resolve()) for ruta in desarrollo_x + test_x]
    repetidas = [ruta for ruta, cantidad in Counter(rutas_resueltas).items() if cantidad > 1]
    if repetidas:
        raise ValueError(f"Se encontraron rutas repetidas entre splits: {repetidas[:5]}")

    return {
        "desarrollo_x": desarrollo_x,
        "desarrollo_y": desarrollo_y,
        "test_x": test_x,
        "test_y": test_y,
        "conteos": {
            "train": dict(Counter(CLASES[int(i)] for i in train_y)),
            "val": dict(Counter(CLASES[int(i)] for i in val_y)),
            "desarrollo_train_mas_val": dict(
                Counter(CLASES[int(i)] for i in desarrollo_y)
            ),
            "test_reservado": dict(Counter(CLASES[int(i)] for i in test_y)),
        },
    }


# ============================================================
# BLOQUE 5: Carga, preprocesamiento y aumento de imagenes
# ============================================================


def decodificar_imagen(ruta: tf.Tensor, etiqueta: tf.Tensor):
    """Convierte una ruta en una imagen RGB de 224x224 pixeles."""

    contenido = tf.io.read_file(ruta)
    imagen = tf.io.decode_image(contenido, channels=3, expand_animations=False)
    imagen.set_shape([None, None, 3])
    imagen = tf.image.resize(imagen, (224, 224), antialias=True)
    return tf.cast(imagen, tf.float32), etiqueta


def crear_aumento(seed: int) -> keras.Sequential:
    """Define transformaciones realistas aplicadas solo al entrenamiento."""

    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(0.05, fill_mode="reflect", seed=seed + 1),
            layers.RandomZoom(0.15, fill_mode="reflect", seed=seed + 2),
            layers.RandomTranslation(0.08, 0.08, fill_mode="reflect", seed=seed + 3),
            layers.RandomContrast(0.15, seed=seed + 4),
        ],
        name="aumento_datos",
    )


def crear_dataset_tf(
    archivos: list[Path],
    etiquetas: np.ndarray,
    batch_size: int,
    entrenamiento: bool,
    seed: int,
):
    """Construye un pipeline tf.data sin mezclar validacion ni test."""

    rutas = np.asarray([str(ruta) for ruta in archivos])
    dataset = tf.data.Dataset.from_tensor_slices((rutas, etiquetas))
    if entrenamiento:
        dataset = dataset.shuffle(len(rutas), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(decodificar_imagen, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)

    if entrenamiento:
        aumento = crear_aumento(seed)

        def preparar_train(imagenes, y):
            imagenes = aumento(imagenes, training=True)
            return preprocess_input(imagenes), y

        dataset = dataset.map(preparar_train, num_parallel_calls=tf.data.AUTOTUNE)
    else:

        def preparar_evaluacion(imagenes, y):
            return preprocess_input(imagenes), y

        dataset = dataset.map(preparar_evaluacion, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.prefetch(tf.data.AUTOTUNE)


# ============================================================
# BLOQUE 6: Construccion de MobileNetV2 y su cabezal
# ============================================================


def crear_modelo() -> tuple[keras.Model, keras.Model]:
    """Crea un modelo nuevo desde ImageNet; se llama una vez por cada fold."""

    base = MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
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
    modelo = keras.Model(entradas, salidas, name="motor_turismo_mobilenetv2_kfold")
    return modelo, base


def compilar(modelo: keras.Model, learning_rate: float) -> None:
    """Configura Adam, entropia cruzada y accuracy para una fase."""

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def descongelar(base: keras.Model, ultimas_capas: int) -> None:
    """Habilita las ultimas capas excepto BatchNormalization."""

    base.trainable = True
    limite = max(0, len(base.layers) - ultimas_capas)
    for indice, capa in enumerate(base.layers):
        capa.trainable = indice >= limite and not isinstance(capa, layers.BatchNormalization)


# ============================================================
# BLOQUE 7: Callbacks y control del entrenamiento
# ============================================================


def callbacks_fold(carpeta: Path, fase: str, paciencia: int):
    """Guarda el mejor modelo y detiene cada fase cuando deja de mejorar."""

    return [
        keras.callbacks.ModelCheckpoint(
            carpeta / f"mejor_{fase}.keras",
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
        keras.callbacks.CSVLogger(carpeta / f"historial_{fase}.csv"),
        keras.callbacks.TerminateOnNaN(),
    ]


def pesos_de_clase(etiquetas: np.ndarray) -> dict[int, float]:
    """Compensa el numero diferente de ejemplos de las clases del fold."""

    pesos = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(CLASES)), y=etiquetas
    )
    return {indice: float(peso) for indice, peso in enumerate(pesos)}


def mejor_epoca(historial: keras.callbacks.History) -> int:
    """Devuelve la epoca con mayor accuracy de validacion, empezando en uno."""

    valores = historial.history.get("val_accuracy", [])
    return int(np.argmax(valores) + 1) if valores else 1


# ============================================================
# BLOQUE 8: Calculo de accuracy, precision, recall y F1-score
# ============================================================


def calcular_metricas(
    verdaderas: np.ndarray, probabilidades: np.ndarray
) -> tuple[dict, dict, np.ndarray]:
    """Calcula las metricas globales y por clase a partir de probabilidades."""

    probabilidades = np.asarray(probabilidades, dtype=np.float64)
    probabilidades /= probabilidades.sum(axis=1, keepdims=True)
    predichas = probabilidades.argmax(axis=1)
    one_hot = np.eye(len(CLASES), dtype=np.float64)[verdaderas]
    reporte = classification_report(
        verdaderas,
        predichas,
        labels=np.arange(len(CLASES)),
        target_names=CLASES,
        output_dict=True,
        zero_division=0,
    )
    matriz = confusion_matrix(
        verdaderas, predichas, labels=np.arange(len(CLASES))
    )
    metricas = {
        "cantidad_imagenes": int(len(verdaderas)),
        "accuracy": float(accuracy_score(verdaderas, predichas)),
        "balanced_accuracy": float(balanced_accuracy_score(verdaderas, predichas)),
        "precision_macro": float(
            precision_score(verdaderas, predichas, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(verdaderas, predichas, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(verdaderas, predichas, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(verdaderas, predichas, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(verdaderas, predichas, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(verdaderas, predichas, average="weighted", zero_division=0)
        ),
        "f1_micro": float(f1_score(verdaderas, predichas, average="micro")),
        "top2_accuracy": float(
            top_k_accuracy_score(
                verdaderas,
                probabilidades,
                k=2,
                labels=np.arange(len(CLASES)),
            )
        ),
        "log_loss": float(
            log_loss(verdaderas, probabilidades, labels=np.arange(len(CLASES)))
        ),
        "roc_auc_ovr_macro": float(
            roc_auc_score(one_hot, probabilidades, average="macro", multi_class="ovr")
        ),
        "roc_auc_ovr_weighted": float(
            roc_auc_score(
                one_hot, probabilidades, average="weighted", multi_class="ovr"
            )
        ),
        "matthews_corrcoef": float(matthews_corrcoef(verdaderas, predichas)),
        "cohen_kappa": float(cohen_kappa_score(verdaderas, predichas)),
        "confianza_media": float(probabilidades.max(axis=1).mean()),
    }
    return metricas, reporte, matriz


# ============================================================
# BLOQUE 9: Guardado reproducible de tablas y predicciones
# ============================================================


def guardar_json(ruta: Path, contenido) -> None:
    ruta.write_text(
        json.dumps(contenido, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def guardar_csv(ruta: Path, filas: list[dict], campos: list[str] | None = None) -> None:
    if not filas:
        return
    campos = campos or list(filas[0].keys())
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)


def guardar_resultados_evaluacion(
    carpeta: Path,
    archivos: list[Path],
    verdaderas: np.ndarray,
    probabilidades: np.ndarray,
    metricas: dict,
    reporte: dict,
    matriz: np.ndarray,
) -> None:
    """Guarda metricas, tabla por clase, matriz y prediccion de cada imagen."""

    carpeta.mkdir(parents=True, exist_ok=True)
    guardar_json(carpeta / "metricas.json", metricas)
    guardar_csv(
        carpeta / "metricas_por_clase.csv",
        [{"clase": clase, **reporte[clase]} for clase in CLASES],
    )
    guardar_csv(
        carpeta / "matriz_confusion.csv",
        [
            {
                "clase_real": CLASES[i],
                **{CLASES[j]: int(matriz[i, j]) for j in range(len(CLASES))},
            }
            for i in range(len(CLASES))
        ],
    )
    predichas = probabilidades.argmax(axis=1)
    filas = []
    for ruta, real, predicha, probs in zip(
        archivos, verdaderas, predichas, probabilidades
    ):
        fila = {
            "archivo": str(ruta),
            "clase_real": CLASES[int(real)],
            "clase_predicha": CLASES[int(predicha)],
            "correcta": bool(real == predicha),
            "confianza": float(probs[int(predicha)]),
        }
        fila.update({f"prob_{clase}": float(probs[i]) for i, clase in enumerate(CLASES)})
        filas.append(fila)
    guardar_csv(carpeta / "predicciones.csv", filas)


# ============================================================
# BLOQUE 10: Curva ROC One-vs-Rest para las ocho clases
# ============================================================


def generar_curva_roc(
    verdaderas: np.ndarray,
    probabilidades: np.ndarray,
    carpeta: Path,
    titulo: str,
) -> dict:
    """Genera una linea ROC y un AUC para cada clase, mas macro y micro."""

    one_hot = np.eye(len(CLASES), dtype=np.float64)[verdaderas]
    curvas: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    auc_clases: dict[str, float] = {}
    filas_csv: list[dict] = []

    for indice, clase in enumerate(CLASES):
        fpr, tpr, umbrales = roc_curve(one_hot[:, indice], probabilidades[:, indice])
        curvas[clase] = (fpr, tpr, umbrales)
        auc_clases[clase] = float(auc(fpr, tpr))
        for x, y, umbral in zip(fpr, tpr, umbrales):
            filas_csv.append(
                {
                    "curva": clase,
                    "uno_menos_especificidad_fpr": float(x),
                    "sensibilidad_tpr": float(y),
                    "umbral": float(umbral),
                }
            )

    fpr_micro, tpr_micro, umbrales_micro = roc_curve(
        one_hot.ravel(), probabilidades.ravel()
    )
    auc_micro = float(auc(fpr_micro, tpr_micro))
    for x, y, umbral in zip(fpr_micro, tpr_micro, umbrales_micro):
        filas_csv.append(
            {
                "curva": "promedio_micro",
                "uno_menos_especificidad_fpr": float(x),
                "sensibilidad_tpr": float(y),
                "umbral": float(umbral),
            }
        )

    todos_fpr = np.unique(np.concatenate([curvas[c][0] for c in CLASES]))
    tpr_macro = np.zeros_like(todos_fpr)
    for clase in CLASES:
        tpr_macro += np.interp(todos_fpr, curvas[clase][0], curvas[clase][1])
    tpr_macro /= len(CLASES)
    auc_macro_curva = float(auc(todos_fpr, tpr_macro))
    for x, y in zip(todos_fpr, tpr_macro):
        filas_csv.append(
            {
                "curva": "promedio_macro",
                "uno_menos_especificidad_fpr": float(x),
                "sensibilidad_tpr": float(y),
                "umbral": "",
            }
        )

    plt.figure(figsize=(13, 10))
    for indice, clase in enumerate(CLASES):
        fpr, tpr, _ = curvas[clase]
        plt.plot(
            fpr,
            tpr,
            color=COLORES_ROC[indice],
            linewidth=2.2,
            label=f"{NOMBRES_GRAFICO[indice]}: AUC {auc_clases[clase]:.3f}",
        )
    plt.plot(
        fpr_micro,
        tpr_micro,
        color="#111111",
        linestyle=":",
        linewidth=2.5,
        label=f"Promedio micro: AUC {auc_micro:.3f}",
    )
    plt.plot(
        todos_fpr,
        tpr_macro,
        color="#6a3d9a",
        linestyle="--",
        linewidth=3,
        label=f"Promedio macro: AUC {auc_macro_curva:.3f}",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        color="#5c6b73",
        linestyle="--",
        linewidth=1.7,
        label="Referencia aleatoria: AUC 0.500",
    )
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xlabel("1 - Especificidad (tasa de falsos positivos)", fontsize=12)
    plt.ylabel("Sensibilidad (tasa de verdaderos positivos)", fontsize=12)
    plt.title(titulo, fontsize=15, fontweight="bold")
    plt.grid(alpha=0.22)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=True,
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0.12, 1, 1))
    plt.savefig(carpeta / "curva_roc_multiclase.png", dpi=220, bbox_inches="tight")
    plt.close()

    guardar_csv(carpeta / "puntos_curva_roc.csv", filas_csv)
    resumen = {
        "auc_por_clase": auc_clases,
        "auc_micro": auc_micro,
        "auc_macro_curva": auc_macro_curva,
    }
    guardar_json(carpeta / "auc_roc.json", resumen)
    return resumen


# ============================================================
# BLOQUE 11: Graficos de matriz de confusion y F1 por clase
# ============================================================


def graficar_matriz(matriz: np.ndarray, carpeta: Path, titulo: str) -> None:
    normalizada = matriz / np.maximum(matriz.sum(axis=1, keepdims=True), 1)
    figura, eje = plt.subplots(figsize=(10, 8))
    imagen = eje.imshow(normalizada, cmap="Blues", vmin=0, vmax=1)
    figura.colorbar(imagen, ax=eje, label="Proporcion por clase real")
    eje.set_xticks(np.arange(len(CLASES)), labels=NOMBRES_GRAFICO, rotation=45, ha="right")
    eje.set_yticks(np.arange(len(CLASES)), labels=NOMBRES_GRAFICO)
    eje.set_xlabel("Clase predicha")
    eje.set_ylabel("Clase real")
    eje.set_title(titulo, fontweight="bold")
    for i in range(len(CLASES)):
        for j in range(len(CLASES)):
            color = "white" if normalizada[i, j] > 0.55 else "black"
            eje.text(
                j,
                i,
                f"{matriz[i, j]}\n{normalizada[i, j]:.1%}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
    figura.tight_layout()
    figura.savefig(carpeta / "matriz_confusion.png", dpi=220, bbox_inches="tight")
    plt.close(figura)


def graficar_f1(reporte: dict, carpeta: Path, titulo: str) -> None:
    valores = [float(reporte[clase]["f1-score"]) for clase in CLASES]
    figura, eje = plt.subplots(figsize=(10, 6))
    barras = eje.barh(NOMBRES_GRAFICO, valores, color="#2f7ebc")
    eje.set_xlim(0, 1)
    eje.set_xlabel("F1-score")
    eje.set_title(titulo, fontweight="bold")
    eje.grid(axis="x", alpha=0.22)
    eje.invert_yaxis()
    for barra, valor in zip(barras, valores):
        eje.text(min(valor + 0.012, 0.96), barra.get_y() + barra.get_height() / 2, f"{valor:.3f}", va="center")
    figura.tight_layout()
    figura.savefig(carpeta / "f1_por_clase.png", dpi=220, bbox_inches="tight")
    plt.close(figura)


# ============================================================
# BLOQUE 12: Entrenamiento de un fold completo
# ============================================================


def entrenar_fold(
    k: int,
    numero_fold: int,
    indices_train: np.ndarray,
    indices_val: np.ndarray,
    archivos: list[Path],
    etiquetas: np.ndarray,
    carpeta_fold: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    """Ejecuta cabeza + fine-tuning y evalua el fold que no vio el modelo."""

    carpeta_fold.mkdir(parents=True, exist_ok=True)
    archivo_estado = carpeta_fold / "resultado_fold.json"
    archivo_npz = carpeta_fold / "predicciones_validacion.npz"
    modelo_fold = carpeta_fold / f"modelo_k{k}_fold{numero_fold}.keras"

    if args.reanudar and archivo_estado.is_file() and archivo_npz.is_file() and modelo_fold.is_file():
        guardado = np.load(archivo_npz)
        if np.array_equal(guardado["indices"], indices_val):
            print(f"K={k} fold={numero_fold}: ya estaba completo; se reutiliza.")
            return guardado["probabilidades"], json.loads(
                archivo_estado.read_text(encoding="utf-8")
            )

    keras.backend.clear_session()
    semilla_fold = args.seed + k * 100 + numero_fold
    configurar_entorno(semilla_fold)

    train_x = [archivos[int(i)] for i in indices_train]
    train_y = etiquetas[indices_train]
    val_x = [archivos[int(i)] for i in indices_val]
    val_y = etiquetas[indices_val]
    train_ds = crear_dataset_tf(
        train_x, train_y, args.batch_size, True, semilla_fold
    )
    val_ds = crear_dataset_tf(val_x, val_y, args.batch_size, False, semilla_fold)
    class_weight = pesos_de_clase(train_y)
    modelo, base = crear_modelo()

    print(f"\nK={k} FOLD={numero_fold}: FASE 1 - cabeza clasificadora")
    compilar(modelo, args.lr_cabeza)
    historial_cabeza = modelo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epocas_cabeza,
        class_weight=class_weight,
        callbacks=callbacks_fold(carpeta_fold, "cabeza", args.paciencia),
        verbose=2,
    )

    print(f"\nK={k} FOLD={numero_fold}: FASE 2 - fine-tuning")
    descongelar(base, args.capas_descongeladas)
    compilar(modelo, args.lr_finetuning)
    historial_finetuning = modelo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epocas_finetuning,
        class_weight=class_weight,
        callbacks=callbacks_fold(carpeta_fold, "finetuning", args.paciencia),
        verbose=2,
    )

    mejor_finetuning = carpeta_fold / "mejor_finetuning.keras"
    modelo = keras.models.load_model(mejor_finetuning)
    probabilidades = np.asarray(modelo.predict(val_ds, verbose=0), dtype=np.float64)
    metricas, reporte, matriz = calcular_metricas(val_y, probabilidades)
    metricas.update(
        {
            "k": k,
            "fold": numero_fold,
            "cantidad_train": int(len(indices_train)),
            "cantidad_validacion": int(len(indices_val)),
            "mejor_epoca_cabeza": mejor_epoca(historial_cabeza),
            "epocas_ejecutadas_cabeza": len(historial_cabeza.history["loss"]),
            "mejor_epoca_finetuning": mejor_epoca(historial_finetuning),
            "epocas_ejecutadas_finetuning": len(historial_finetuning.history["loss"]),
            "pesos_clase": class_weight,
        }
    )

    guardar_resultados_evaluacion(
        carpeta_fold, val_x, val_y, probabilidades, metricas, reporte, matriz
    )
    graficar_matriz(
        matriz, carpeta_fold, f"Matriz de confusion - K={k}, fold {numero_fold}"
    )
    graficar_f1(reporte, carpeta_fold, f"F1 por clase - K={k}, fold {numero_fold}")
    generar_curva_roc(
        val_y,
        probabilidades,
        carpeta_fold,
        f"Curvas ROC One-vs-Rest - K={k}, fold {numero_fold}",
    )
    np.savez_compressed(
        archivo_npz,
        indices=indices_val,
        etiquetas=val_y,
        probabilidades=probabilidades,
    )
    shutil.copy2(mejor_finetuning, modelo_fold)
    guardar_json(archivo_estado, metricas)
    print(
        f"K={k} fold={numero_fold}: accuracy={metricas['accuracy']:.4f}, "
        f"F1 macro={metricas['f1_macro']:.4f}, "
        f"AUC macro={metricas['roc_auc_ovr_macro']:.4f}"
    )

    del modelo, base, train_ds, val_ds
    keras.backend.clear_session()
    gc.collect()
    return probabilidades, metricas


# ============================================================
# BLOQUE 13: Validacion cruzada Stratified K-Fold
# ============================================================


def ejecutar_kfold(
    k: int,
    archivos: list[Path],
    etiquetas: np.ndarray,
    salida: Path,
    args: argparse.Namespace,
) -> dict:
    """Entrena K modelos y construye predicciones out-of-fold completas."""

    carpeta_k = salida / f"k{k}"
    carpeta_k.mkdir(parents=True, exist_ok=True)
    separador = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.seed)
    oof = np.full((len(etiquetas), len(CLASES)), np.nan, dtype=np.float64)
    resultados_folds: list[dict] = []

    for numero_fold, (indices_train, indices_val) in enumerate(
        separador.split(np.zeros(len(etiquetas)), etiquetas), start=1
    ):
        print("\n" + "=" * 72)
        print(
            f"STRATIFIED K-FOLD: K={k}, FOLD={numero_fold}/{k}, "
            f"train={len(indices_train)}, val={len(indices_val)}"
        )
        print("=" * 72)
        probabilidades, metricas_fold = entrenar_fold(
            k,
            numero_fold,
            indices_train,
            indices_val,
            archivos,
            etiquetas,
            carpeta_k / f"fold_{numero_fold}",
            args,
        )
        oof[indices_val] = probabilidades
        resultados_folds.append(metricas_fold)

    if np.isnan(oof).any():
        raise RuntimeError(f"Quedaron predicciones OOF vacias para K={k}")

    metricas_oof, reporte_oof, matriz_oof = calcular_metricas(etiquetas, oof)
    carpeta_oof = carpeta_k / "resultado_oof"
    guardar_resultados_evaluacion(
        carpeta_oof, archivos, etiquetas, oof, metricas_oof, reporte_oof, matriz_oof
    )
    roc_oof = generar_curva_roc(
        etiquetas,
        oof,
        carpeta_oof,
        f"Curvas ROC multiclase con predicciones OOF - K={k}",
    )
    graficar_matriz(
        matriz_oof, carpeta_oof, f"Matriz de confusion acumulada OOF - K={k}"
    )
    graficar_f1(reporte_oof, carpeta_oof, f"F1 por clase OOF - K={k}")

    nombres_metricas = [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "roc_auc_ovr_macro",
        "top2_accuracy",
        "log_loss",
        "matthews_corrcoef",
        "cohen_kappa",
    ]
    resumen_cv = {}
    for nombre in nombres_metricas:
        valores = np.asarray([fila[nombre] for fila in resultados_folds], dtype=float)
        resumen_cv[nombre] = {
            "media": float(valores.mean()),
            "desviacion_estandar": float(valores.std(ddof=1)) if len(valores) > 1 else 0.0,
            "minimo": float(valores.min()),
            "maximo": float(valores.max()),
        }

    guardar_csv(carpeta_k / "metricas_por_fold.csv", resultados_folds)
    resumen = {
        "k": k,
        "cantidad_desarrollo": int(len(etiquetas)),
        "metricas_cv_media_desviacion": resumen_cv,
        "metricas_oof": metricas_oof,
        "roc_oof": roc_oof,
        "folds": resultados_folds,
    }
    guardar_json(carpeta_k / "resumen_kfold.json", resumen)
    np.savez_compressed(
        carpeta_oof / "predicciones_oof.npz",
        etiquetas=etiquetas,
        probabilidades=oof,
    )
    print(
        f"\nRESUMEN K={k}: accuracy={resumen_cv['accuracy']['media']:.4f} "
        f"± {resumen_cv['accuracy']['desviacion_estandar']:.4f}; "
        f"F1 macro={resumen_cv['f1_macro']['media']:.4f} "
        f"± {resumen_cv['f1_macro']['desviacion_estandar']:.4f}"
    )
    return resumen


# ============================================================
# BLOQUE 14: Comparacion entre los valores de K
# ============================================================


def comparar_k(resultados: list[dict], salida: Path) -> int:
    """Crea una tabla K=3 vs K=5 y selecciona K por F1 macro medio."""

    filas = []
    for resultado in resultados:
        resumen = resultado["metricas_cv_media_desviacion"]
        filas.append(
            {
                "k": resultado["k"],
                "accuracy_media": resumen["accuracy"]["media"],
                "accuracy_desviacion": resumen["accuracy"]["desviacion_estandar"],
                "precision_macro_media": resumen["precision_macro"]["media"],
                "recall_macro_media": resumen["recall_macro"]["media"],
                "f1_macro_media": resumen["f1_macro"]["media"],
                "f1_macro_desviacion": resumen["f1_macro"]["desviacion_estandar"],
                "roc_auc_macro_media": resumen["roc_auc_ovr_macro"]["media"],
                "roc_auc_macro_desviacion": resumen["roc_auc_ovr_macro"]["desviacion_estandar"],
            }
        )
    guardar_csv(salida / "comparacion_k3_k5.csv", filas)
    ganador = max(
        resultados,
        key=lambda r: (
            r["metricas_cv_media_desviacion"]["f1_macro"]["media"],
            r["metricas_cv_media_desviacion"]["accuracy"]["media"],
        ),
    )
    k_ganador = int(ganador["k"])
    guardar_json(
        salida / "seleccion_k.json",
        {
            "criterio_principal": "mayor F1 macro promedio",
            "desempate": "mayor accuracy promedio",
            "k_seleccionado": k_ganador,
            "comparacion": filas,
        },
    )
    return k_ganador


# ============================================================
# BLOQUE 15: Entrenamiento final con todas las imagenes de desarrollo
# ============================================================


def entrenar_modelo_final(
    archivos: list[Path],
    etiquetas: np.ndarray,
    resultado_k: dict,
    salida: Path,
    args: argparse.Namespace,
) -> tuple[Path, dict]:
    """Entrena una vez con train+val usando epocas elegidas por K-Fold."""

    carpeta = salida / "modelo_final"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_modelo = carpeta / "modelo_mobilenetv2_kfold_v2_final.keras"
    ruta_meta = carpeta / "metadatos_modelo_final.json"
    if args.reanudar and ruta_modelo.is_file() and ruta_meta.is_file():
        print("El modelo final ya existe; se reutiliza.")
        return ruta_modelo, json.loads(ruta_meta.read_text(encoding="utf-8"))

    epocas_cabeza = max(
        1,
        int(round(np.median([f["mejor_epoca_cabeza"] for f in resultado_k["folds"]]))),
    )
    epocas_ft = max(
        1,
        int(round(np.median([f["mejor_epoca_finetuning"] for f in resultado_k["folds"]]))),
    )
    print(
        f"\nMODELO FINAL: {epocas_cabeza} epocas de cabeza y "
        f"{epocas_ft} epocas de fine-tuning elegidas por K-Fold."
    )
    keras.backend.clear_session()
    configurar_entorno(args.seed + 9999)
    train_ds = crear_dataset_tf(
        archivos, etiquetas, args.batch_size, True, args.seed + 9999
    )
    modelo, base = crear_modelo()
    class_weight = pesos_de_clase(etiquetas)
    callbacks_final = lambda fase: [
        keras.callbacks.CSVLogger(carpeta / f"historial_{fase}.csv"),
        keras.callbacks.TerminateOnNaN(),
    ]

    compilar(modelo, args.lr_cabeza)
    modelo.fit(
        train_ds,
        epochs=epocas_cabeza,
        class_weight=class_weight,
        callbacks=callbacks_final("cabeza"),
        verbose=2,
    )
    descongelar(base, args.capas_descongeladas)
    compilar(modelo, args.lr_finetuning)
    modelo.fit(
        train_ds,
        epochs=epocas_ft,
        class_weight=class_weight,
        callbacks=callbacks_final("finetuning"),
        verbose=2,
    )
    modelo.save(ruta_modelo)
    metadatos = {
        "k_seleccionado": resultado_k["k"],
        "criterio_epocas": "mediana de la mejor epoca de los folds",
        "epocas_cabeza": epocas_cabeza,
        "epocas_finetuning": epocas_ft,
        "cantidad_desarrollo": int(len(etiquetas)),
        "pesos_clase": class_weight,
    }
    guardar_json(ruta_meta, metadatos)
    return ruta_modelo, metadatos


# ============================================================
# BLOQUE 16: Evaluacion unica del modelo final con el test reservado
# ============================================================


def evaluar_test_final(
    ruta_modelo: Path,
    archivos: list[Path],
    etiquetas: np.ndarray,
    salida: Path,
    args: argparse.Namespace,
) -> dict:
    """Usa el test solamente al final, despues de seleccionar K y las epocas."""

    carpeta = salida / "evaluacion_test_interno"
    carpeta.mkdir(parents=True, exist_ok=True)
    modelo = keras.models.load_model(ruta_modelo)
    test_ds = crear_dataset_tf(
        archivos, etiquetas, args.batch_size, False, args.seed
    )
    probabilidades = np.asarray(modelo.predict(test_ds, verbose=0), dtype=np.float64)
    metricas, reporte, matriz = calcular_metricas(etiquetas, probabilidades)
    guardar_resultados_evaluacion(
        carpeta, archivos, etiquetas, probabilidades, metricas, reporte, matriz
    )
    roc = generar_curva_roc(
        etiquetas,
        probabilidades,
        carpeta,
        "Curvas ROC multiclase - test interno reservado",
    )
    graficar_matriz(matriz, carpeta, "Matriz de confusion - test interno reservado")
    graficar_f1(reporte, carpeta, "F1 por clase - test interno reservado")
    resultado = {**metricas, "roc": roc}
    guardar_json(carpeta / "resumen_test_interno.json", resultado)
    print(
        f"TEST FINAL: accuracy={metricas['accuracy']:.4f}, "
        f"F1 macro={metricas['f1_macro']:.4f}, "
        f"AUC macro={metricas['roc_auc_ovr_macro']:.4f}"
    )
    return resultado


# ============================================================
# BLOQUE 17: Flujo principal y guardado del resumen completo
# ============================================================


def main() -> None:
    args = argumentos()
    args.dataset = args.dataset.resolve()
    args.salida = args.salida.resolve()
    if any(k < 2 for k in args.k):
        raise ValueError("Cada valor de K debe ser al menos 2.")
    if len(set(args.k)) != len(args.k):
        raise ValueError("Los valores de K no deben repetirse.")
    if args.epocas_cabeza < 1 or args.epocas_finetuning < 1:
        raise ValueError("Las epocas deben ser mayores que cero.")
    if not 1 <= args.capas_descongeladas <= 154:
        raise ValueError("--capas-descongeladas debe estar entre 1 y 154.")

    dispositivos_gpu = configurar_entorno(args.seed)
    inventario = preparar_inventario(args.dataset)
    desarrollo_x = inventario["desarrollo_x"]
    desarrollo_y = inventario["desarrollo_y"]
    test_x = inventario["test_x"]
    test_y = inventario["test_y"]

    print("Clases:", CLASES)
    print("Conteos:", json.dumps(inventario["conteos"], ensure_ascii=False, indent=2))
    print("GPU detectada:", dispositivos_gpu or "ninguna; se usara CPU")
    print("Valores de K:", args.k)
    print("SMOTE: no se aplica")
    if args.solo_verificar:
        print("Verificacion completada. No se entreno ningun modelo.")
        return

    args.salida.mkdir(parents=True, exist_ok=True)
    guardar_json(args.salida / "inventario_dataset.json", inventario["conteos"])
    guardar_json(args.salida / "clases.json", CLASES)

    resultados_k = [
        ejecutar_kfold(k, desarrollo_x, desarrollo_y, args.salida, args)
        for k in args.k
    ]
    k_seleccionado = comparar_k(resultados_k, args.salida)
    resultado_seleccionado = next(r for r in resultados_k if r["k"] == k_seleccionado)

    resumen_final = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "version": "MobileNetV2 K-Fold v2",
        "dataset": str(args.dataset),
        "salida": str(args.salida),
        "clases": CLASES,
        "conteos": inventario["conteos"],
        "k_evaluados": args.k,
        "k_seleccionado": k_seleccionado,
        "smote_aplicado": False,
        "test_usado_durante_kfold": False,
        "tensorflow": tf.__version__,
        "gpu_tensorflow": dispositivos_gpu,
        "resultados_kfold": resultados_k,
    }

    if not args.sin_modelo_final:
        ruta_modelo, metadatos_modelo = entrenar_modelo_final(
            desarrollo_x,
            desarrollo_y,
            resultado_seleccionado,
            args.salida,
            args,
        )
        resultado_test = evaluar_test_final(
            ruta_modelo, test_x, test_y, args.salida, args
        )
        resumen_final["modelo_final"] = str(ruta_modelo)
        resumen_final["metadatos_modelo_final"] = metadatos_modelo
        resumen_final["resultado_test_interno"] = resultado_test

    guardar_json(args.salida / "resumen_completo_v2.json", resumen_final)
    print("\nENTRENAMIENTO VERSION 2 TERMINADO")
    print("K seleccionado:", k_seleccionado)
    print("Resultados:", args.salida)


if __name__ == "__main__":
    main()
