"""Evaluacion final reproducible del modelo de turismo de 8 clases.

No modifica ni reentrena el modelo. Evalua una sola vez el test interno y el
test externo y genera metricas, predicciones y graficos PNG.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (
    accuracy_score,
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
    top_k_accuracy_score,
)
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
    parser = argparse.ArgumentParser(description="Evalua el modelo final sin reentrenarlo.")
    parser.add_argument(
        "--modelo",
        type=Path,
        default=Path(
            r"D:\proyecto_turismo_ia\entrenamiento_mobilenetv2_v2"
            r"\modelo_mobilenetv2_limpio_final.keras"
        ),
    )
    parser.add_argument(
        "--test-interno",
        type=Path,
        default=Path(r"D:\proyecto_turismo_ia\dataset_limpio_v2\test"),
    )
    parser.add_argument(
        "--test-externo",
        type=Path,
        default=Path(r"D:\proyecto_turismo_ia\test_externo_places365\final"),
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=Path(
            r"D:\proyecto_turismo_ia\entrenamiento_mobilenetv2_v2"
            r"\evaluacion_final"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def archivos_y_etiquetas(carpeta: Path) -> tuple[list[Path], np.ndarray]:
    archivos: list[Path] = []
    etiquetas: list[int] = []
    encontradas = sorted(p.name for p in carpeta.iterdir() if p.is_dir())
    if encontradas != sorted(CLASES):
        raise ValueError(
            f"Clases invalidas en {carpeta}. Esperadas={sorted(CLASES)}, "
            f"encontradas={encontradas}"
        )
    for indice, clase in enumerate(CLASES):
        for archivo in sorted((carpeta / clase).rglob("*")):
            if archivo.is_file() and archivo.suffix.lower() in EXTENSIONES:
                archivos.append(archivo)
                etiquetas.append(indice)
    if not archivos:
        raise ValueError(f"No se encontraron imagenes en {carpeta}")
    return archivos, np.asarray(etiquetas, dtype=np.int64)


def cargar_imagen(ruta: tf.Tensor) -> tf.Tensor:
    contenido = tf.io.read_file(ruta)
    imagen = tf.io.decode_image(contenido, channels=3, expand_animations=False)
    imagen.set_shape([None, None, 3])
    imagen = tf.image.resize(imagen, (224, 224), antialias=True)
    return preprocess_input(tf.cast(imagen, tf.float32))


def predecir(
    modelo: keras.Model, archivos: list[Path], batch_size: int
) -> np.ndarray:
    rutas = [str(ruta) for ruta in archivos]
    dataset = tf.data.Dataset.from_tensor_slices(rutas)
    dataset = dataset.map(cargar_imagen, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return np.asarray(modelo.predict(dataset, verbose=1), dtype=np.float64)


def expected_calibration_error(
    verdaderas: np.ndarray, predichas: np.ndarray, probabilidades: np.ndarray, bins: int = 10
) -> tuple[float, list[dict[str, float | int]]]:
    confianza = probabilidades.max(axis=1)
    aciertos = (verdaderas == predichas).astype(float)
    limites = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    tabla = []
    for i in range(bins):
        izquierda, derecha = limites[i], limites[i + 1]
        mascara = (confianza > izquierda) & (confianza <= derecha)
        if i == 0:
            mascara = (confianza >= izquierda) & (confianza <= derecha)
        cantidad = int(mascara.sum())
        if cantidad:
            conf_media = float(confianza[mascara].mean())
            acc_bin = float(aciertos[mascara].mean())
            ece += cantidad / len(verdaderas) * abs(acc_bin - conf_media)
        else:
            conf_media = 0.0
            acc_bin = 0.0
        tabla.append(
            {
                "limite_inferior": float(izquierda),
                "limite_superior": float(derecha),
                "cantidad": cantidad,
                "confianza_media": conf_media,
                "exactitud": acc_bin,
            }
        )
    return float(ece), tabla


def calcular_metricas(
    verdaderas: np.ndarray, probabilidades: np.ndarray
) -> tuple[dict, dict, np.ndarray, list[dict]]:
    # Evita diferencias minimas de punto flotante al calcular log-loss.
    probabilidades = probabilidades / probabilidades.sum(axis=1, keepdims=True)
    predichas = probabilidades.argmax(axis=1)
    confianza = probabilidades.max(axis=1)
    aciertos = predichas == verdaderas
    matriz = confusion_matrix(verdaderas, predichas, labels=np.arange(len(CLASES)))
    reporte = classification_report(
        verdaderas,
        predichas,
        labels=np.arange(len(CLASES)),
        target_names=CLASES,
        output_dict=True,
        zero_division=0,
    )
    ece, calibracion = expected_calibration_error(
        verdaderas, predichas, probabilidades
    )
    one_hot = np.eye(len(CLASES), dtype=np.float64)[verdaderas]
    umbrales = {}
    for umbral in (0.50, 0.60, 0.70, 0.80, 0.90):
        mascara = confianza >= umbral
        umbrales[f"{umbral:.2f}"] = {
            "cobertura": float(mascara.mean()),
            "cantidad": int(mascara.sum()),
            "accuracy_si_se_acepta": (
                float(aciertos[mascara].mean()) if mascara.any() else None
            ),
        }
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
        "brier_multiclase": float(np.mean(np.sum((probabilidades - one_hot) ** 2, axis=1))),
        "roc_auc_ovr_macro": float(
            roc_auc_score(one_hot, probabilidades, average="macro", multi_class="ovr")
        ),
        "matthews_corrcoef": float(matthews_corrcoef(verdaderas, predichas)),
        "cohen_kappa": float(cohen_kappa_score(verdaderas, predichas)),
        "confianza_media": float(confianza.mean()),
        "confianza_media_aciertos": float(confianza[aciertos].mean()) if aciertos.any() else None,
        "confianza_media_errores": float(confianza[~aciertos].mean()) if (~aciertos).any() else None,
        "ece_10_bins": ece,
        "rendimiento_por_umbral_confianza": umbrales,
    }
    return metricas, reporte, matriz, calibracion


def guardar_csv(ruta: Path, filas: list[dict], campos: list[str] | None = None) -> None:
    if not filas:
        return
    campos = campos or list(filas[0].keys())
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)


def color_calor(valor: float) -> tuple[int, int, int]:
    valor = max(0.0, min(1.0, valor))
    return (int(245 - 190 * valor), int(248 - 130 * valor), int(255 - 35 * valor))


def grafico_matriz(matriz: np.ndarray, ruta: Path, titulo: str) -> None:
    fuente = ImageFont.load_default()
    celda, margen_x, margen_y = 92, 155, 85
    ancho = margen_x + celda * len(CLASES) + 35
    alto = margen_y + celda * len(CLASES) + 85
    imagen = Image.new("RGB", (ancho, alto), "white")
    draw = ImageDraw.Draw(imagen)
    draw.text((20, 18), titulo, fill="black", font=fuente)
    draw.text((margen_x + 210, 42), "Clase predicha", fill="black", font=fuente)
    draw.text((20, margen_y - 28), "Clase real", fill="black", font=fuente)
    normalizada = matriz / np.maximum(matriz.sum(axis=1, keepdims=True), 1)
    for i, clase in enumerate(CLASES):
        draw.text((20, margen_y + i * celda + celda // 2), clase, fill="black", font=fuente)
        draw.text((margen_x + i * celda + 8, margen_y - 18), clase, fill="black", font=fuente)
        for j in range(len(CLASES)):
            x0, y0 = margen_x + j * celda, margen_y + i * celda
            x1, y1 = x0 + celda, y0 + celda
            draw.rectangle((x0, y0, x1, y1), fill=color_calor(float(normalizada[i, j])), outline="white")
            texto = f"{matriz[i, j]}\n{normalizada[i, j] * 100:.1f}%"
            draw.multiline_text((x0 + 26, y0 + 31), texto, fill="black", font=fuente, align="center")
    imagen.save(ruta)


def grafico_f1(reporte: dict, ruta: Path, titulo: str) -> None:
    fuente = ImageFont.load_default()
    ancho, alto = 920, 560
    imagen = Image.new("RGB", (ancho, alto), "white")
    draw = ImageDraw.Draw(imagen)
    draw.text((25, 18), titulo, fill="black", font=fuente)
    origen_x, max_ancho = 150, 700
    for i, clase in enumerate(CLASES):
        y = 65 + i * 58
        valor = float(reporte[clase]["f1-score"])
        draw.text((25, y + 11), clase, fill="black", font=fuente)
        draw.rectangle((origen_x, y, origen_x + max_ancho, y + 34), fill=(235, 238, 242))
        draw.rectangle((origen_x, y, origen_x + int(max_ancho * valor), y + 34), fill=(48, 130, 210))
        draw.text((origen_x + int(max_ancho * valor) + 8, y + 11), f"{valor:.3f}", fill="black", font=fuente)
    draw.text((origen_x, 535), "0.0", fill="black", font=fuente)
    draw.text((origen_x + max_ancho - 18, 535), "1.0", fill="black", font=fuente)
    imagen.save(ruta)


def grafico_calibracion(tabla: list[dict], ruta: Path, titulo: str) -> None:
    fuente = ImageFont.load_default()
    ancho, alto = 680, 620
    imagen = Image.new("RGB", (ancho, alto), "white")
    draw = ImageDraw.Draw(imagen)
    draw.text((25, 18), titulo, fill="black", font=fuente)
    x0, y0, lado = 75, 555, 500
    draw.line((x0, y0, x0 + lado, y0 - lado), fill=(150, 150, 150), width=2)
    draw.line((x0, y0, x0 + lado, y0), fill="black", width=2)
    draw.line((x0, y0, x0, y0 - lado), fill="black", width=2)
    puntos = [
        (
            x0 + int(float(f["confianza_media"]) * lado),
            y0 - int(float(f["exactitud"]) * lado),
        )
        for f in tabla
        if int(f["cantidad"]) > 0
    ]
    for x, y in puntos:
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(220, 70, 60))
    if len(puntos) > 1:
        draw.line(puntos, fill=(220, 70, 60), width=3)
    for i in range(6):
        v = i / 5
        draw.text((x0 + int(v * lado) - 8, y0 + 12), f"{v:.1f}", fill="black", font=fuente)
        draw.text((35, y0 - int(v * lado) - 5), f"{v:.1f}", fill="black", font=fuente)
    draw.text((270, 590), "Confianza", fill="black", font=fuente)
    draw.text((8, 42), "Exactitud", fill="black", font=fuente)
    imagen.save(ruta)


def evaluar_conjunto(
    nombre: str,
    carpeta: Path,
    modelo: keras.Model,
    salida: Path,
    batch_size: int,
) -> dict:
    print(f"\nEvaluando {nombre}: {carpeta}")
    archivos, verdaderas = archivos_y_etiquetas(carpeta)
    probabilidades = predecir(modelo, archivos, batch_size)
    if probabilidades.shape != (len(archivos), len(CLASES)):
        raise ValueError(f"Salida inesperada del modelo: {probabilidades.shape}")
    predichas = probabilidades.argmax(axis=1)
    metricas, reporte, matriz, calibracion = calcular_metricas(verdaderas, probabilidades)

    carpeta_salida = salida / nombre
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    (carpeta_salida / "metricas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    filas_reporte = []
    for clase in CLASES:
        fila = {"clase": clase, **reporte[clase]}
        filas_reporte.append(fila)
    guardar_csv(carpeta_salida / "metricas_por_clase.csv", filas_reporte)
    guardar_csv(carpeta_salida / "calibracion.csv", calibracion)
    guardar_csv(
        carpeta_salida / "matriz_confusion.csv",
        [
            {"clase_real": CLASES[i], **{CLASES[j]: int(matriz[i, j]) for j in range(len(CLASES))}}
            for i in range(len(CLASES))
        ],
    )
    filas_predicciones = []
    for ruta, real, predicha, probs in zip(archivos, verdaderas, predichas, probabilidades):
        orden = np.argsort(probs)[::-1]
        filas_predicciones.append(
            {
                "archivo": str(ruta),
                "clase_real": CLASES[int(real)],
                "clase_predicha": CLASES[int(predicha)],
                "correcta": bool(real == predicha),
                "confianza": float(probs[predicha]),
                "segunda_clase": CLASES[int(orden[1])],
                "segunda_probabilidad": float(probs[orden[1]]),
            }
        )
    guardar_csv(carpeta_salida / "predicciones.csv", filas_predicciones)
    grafico_matriz(matriz, carpeta_salida / "matriz_confusion.png", f"Matriz de confusion - {nombre}")
    grafico_f1(reporte, carpeta_salida / "f1_por_clase.png", f"F1 por clase - {nombre}")
    grafico_calibracion(calibracion, carpeta_salida / "calibracion.png", f"Calibracion - {nombre}")
    print(
        f"{nombre}: accuracy={metricas['accuracy']:.4f}, "
        f"F1 macro={metricas['f1_macro']:.4f}, ECE={metricas['ece_10_bins']:.4f}"
    )
    return {"ruta": str(carpeta), **metricas, "por_clase": reporte}


def main() -> None:
    args = argumentos()
    if not args.modelo.is_file():
        raise FileNotFoundError(f"No existe el modelo: {args.modelo}")
    for carpeta in (args.test_interno, args.test_externo):
        if not carpeta.is_dir():
            raise FileNotFoundError(f"No existe el test: {carpeta}")
    args.salida.mkdir(parents=True, exist_ok=True)
    modelo = keras.models.load_model(args.modelo)
    if modelo.output_shape[-1] != len(CLASES):
        raise ValueError(
            f"El modelo tiene {modelo.output_shape[-1]} salidas; se esperaban {len(CLASES)}"
        )
    resumen = {
        "modelo": str(args.modelo),
        "clases": CLASES,
        "nota": "Los tests no se usaron para entrenar ni ajustar hiperparametros.",
        "conjuntos": {},
    }
    resumen["conjuntos"]["test_interno"] = evaluar_conjunto(
        "test_interno", args.test_interno, modelo, args.salida, args.batch_size
    )
    resumen["conjuntos"]["test_externo_places365"] = evaluar_conjunto(
        "test_externo_places365", args.test_externo, modelo, args.salida, args.batch_size
    )
    (args.salida / "resumen_completo.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nResultados guardados en: {args.salida}")


if __name__ == "__main__":
    main()
