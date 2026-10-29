"""Motor multimodal para validar atractivos turísticos.

Combina el modelo CNN de ocho clases, similitud por embeddings, Gemma mediante
Ollama y validación geográfica/altimétrica contra la base GIS.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import threading
import unicodedata
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import ollama
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from geopy.distance import geodesic
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model

from gemma_semantica import (
    CLASES_GEMMA,
    TIPOS_CONTENIDO,
    VISIBILIDADES,
    ajustar_gemma_con_evidencia_visual,
    decidir_estado_semantico,
    normalizar_respuesta_gemma,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("TURISMO_MODEL_PATH", BASE_DIR / "models" / "modelo_mobilenetv2_limpio_final.keras"))
EMBEDDINGS_PATH = Path(os.getenv("TURISMO_EMBEDDINGS_PATH", BASE_DIR / "fewshot_db" / "embeddings_v2.pkl"))
EXCEL_PATH = Path(os.getenv("TURISMO_GIS_PATH", BASE_DIR / "lugares_turisticos_app_final_v3.xlsx"))
TEMP_DIR = BASE_DIR / "temp_uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

CLASES = ["bosque", "cascada", "desierto", "lago", "montana", "nevado", "playa", "volcan"]
IMG_SIZE = (224, 224)
TOP_K = 5
VECINOS_VOTO = 15
MAX_ALTITUDE_DIFF = 300.0
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma4:e2b-it-qat")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
GEMMA_ENABLED = os.getenv("GEMMA_ENABLED", "true").lower() in {"1", "true", "yes", "si"}
GEMMA_NUM_GPU = int(os.getenv("GEMMA_NUM_GPU", "0"))
GEMMA_NUM_CTX = int(os.getenv("GEMMA_NUM_CTX", "2048"))
GEMMA_NUM_BATCH = int(os.getenv("GEMMA_NUM_BATCH", "64"))
GEMMA_TIMEOUT_SECONDS = float(os.getenv("GEMMA_TIMEOUT_SECONDS", "180"))
MOTOR_VERSION = "2.3.0"

CLASE_A_SUBTIPOS = {
    "bosque": ["bosques", "zonas paisajisticas", "zona paisajistica"],
    "cascada": ["cascadas", "caidas de agua", "caida de agua"],
    "desierto": ["desiertos", "planicies"],
    "lago": ["lagos", "lagunas"],
    "montana": ["cerros", "montanas", "cordilleras"],
    "nevado": ["nevados"],
    "playa": ["playas", "costas"],
    "volcan": ["volcanes"],
}

GEMMA_SCHEMA = {
    "type": "object",
    "properties": {
        "clase": {"type": "string", "enum": list(CLASES_GEMMA)},
        "tipo_contenido": {"type": "string", "enum": list(TIPOS_CONTENIDO)},
        "atractivo_visible": {"type": "boolean"},
        "visibilidad": {"type": "string", "enum": list(VISIBILIDADES)},
        "personas_como_contexto": {"type": "boolean"},
        "descripcion_breve": {"type": "string", "minLength": 1},
    },
    "required": [
        "clase",
        "tipo_contenido",
        "atractivo_visible",
        "visibilidad",
        "personas_como_contexto",
        "descripcion_breve",
    ],
    "additionalProperties": False,
}

app = FastAPI(
    title="Motor IA de Turismo",
    version=MOTOR_VERSION,
    description="CNN + embeddings + Gemma/Ollama + GIS",
)


def normalizar_texto(valor: Any) -> str:
    texto = str(valor).strip().lower()
    return "".join(
        caracter
        for caracter in unicodedata.normalize("NFD", texto)
        if unicodedata.category(caracter) != "Mn"
    )


def limpiar_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(clave): limpiar_json(valor) for clave, valor in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [limpiar_json(valor) for valor in obj]
    if isinstance(obj, np.generic):
        return limpiar_json(obj.item())
    if isinstance(obj, float):
        return None if np.isnan(obj) or np.isinf(obj) else obj
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def cargar_recursos():
    for ruta in (MODEL_PATH, EMBEDDINGS_PATH, EXCEL_PATH):
        if not ruta.is_file():
            raise FileNotFoundError(f"Falta un recurso obligatorio: {ruta}")

    print(f"Cargando modelo de 8 clases: {MODEL_PATH}")
    modelo = tf.keras.models.load_model(MODEL_PATH, compile=False)
    if int(modelo.output_shape[-1]) != len(CLASES):
        raise ValueError(f"El modelo tiene {modelo.output_shape[-1]} salidas; se esperaban {len(CLASES)}")
    extractor = Model(
        inputs=modelo.inputs,
        outputs=modelo.get_layer("embedding_dense").output,
        name="extractor_embeddings",
    )

    print(f"Cargando embeddings: {EMBEDDINGS_PATH}")
    with EMBEDDINGS_PATH.open("rb") as archivo:
        paquete = pickle.load(archivo)
    if not isinstance(paquete, dict) or paquete.get("version") != 2:
        raise ValueError("La base de embeddings no corresponde al formato v2")
    if paquete.get("clases") != CLASES:
        raise ValueError("El orden de clases de los embeddings no coincide con la API")
    if paquete.get("modelo_sha256") != sha256(MODEL_PATH):
        raise ValueError("Los embeddings fueron generados con otro modelo")

    matriz = np.asarray(paquete["embeddings"], dtype=np.float32)
    labels = np.asarray(paquete["labels"], dtype=str)
    files = list(paquete["files"])
    paths = list(paquete["paths"])
    if matriz.ndim != 2 or matriz.shape[1] != int(paquete["dimension"]):
        raise ValueError(f"Matriz de embeddings inválida: {matriz.shape}")
    if not (len(matriz) == len(labels) == len(files) == len(paths)):
        raise ValueError("Los metadatos de embeddings tienen longitudes incompatibles")

    print(f"Cargando GIS: {EXCEL_PATH}")
    gis = pd.read_excel(EXCEL_PATH)
    gis.columns = [str(columna).strip() for columna in gis.columns]
    requeridas = {"Nombre", "SubTipo", "Latitud", "Longitud", "Altitud"}
    faltantes = requeridas - set(gis.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas GIS: {sorted(faltantes)}")
    return modelo, extractor, paquete, matriz, labels, files, paths, gis


(
    model,
    embedding_model,
    embeddings_db,
    embeddings_matrix,
    embedding_labels,
    embedding_files,
    embedding_paths,
    gis_df,
) = cargar_recursos()

MODEL_LOCK = threading.Lock()
ollama_client = ollama.Client(host=OLLAMA_HOST, timeout=GEMMA_TIMEOUT_SECONDS)
print(f"Motor {MOTOR_VERSION} listo. Clases: {CLASES}")


def cargar_imagen(file_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(file_bytes)) as original:
            imagen = original.convert("RGB").resize(IMG_SIZE, Image.Resampling.LANCZOS)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail="El archivo no es una imagen válida") from error
    matriz = np.asarray(imagen, dtype=np.float32)
    return preprocess_input(matriz)[None, ...]


def guardar_imagen_temporal(file_bytes: bytes, filename: str | None) -> Path:
    extension = Path(filename or "imagen.jpg").suffix.lower()
    if extension not in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}:
        extension = ".jpg"
    ruta = TEMP_DIR / f"{uuid.uuid4().hex}{extension}"
    ruta.write_bytes(file_bytes)
    return ruta


def analizar_imagen(img_array: np.ndarray) -> dict[str, Any]:
    with MODEL_LOCK:
        probabilidades = np.asarray(model.predict(img_array, verbose=0)[0], dtype=float)
        vector = np.asarray(embedding_model.predict(img_array, verbose=0)[0], dtype=np.float32)

    indice = int(np.argmax(probabilidades))
    clase_cnn = CLASES[indice]
    confianza = float(probabilidades[indice])
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    similitudes = embeddings_matrix @ vector

    cantidad_voto = min(VECINOS_VOTO, len(similitudes))
    indices_voto = np.argpartition(similitudes, -cantidad_voto)[-cantidad_voto:]
    indices_voto = indices_voto[np.argsort(similitudes[indices_voto])[::-1]]
    votos: dict[str, float] = defaultdict(float)
    maximos: dict[str, float] = defaultdict(lambda: -1.0)
    for idx in indices_voto:
        clase = str(embedding_labels[idx])
        similitud = float(similitudes[idx])
        votos[clase] += max(similitud, 0.0)
        maximos[clase] = max(maximos[clase], similitud)
    clase_embeddings = max(votos, key=votos.get)
    similitud_clase = maximos[clase_embeddings]

    cantidad_top = min(TOP_K, len(similitudes))
    indices_top = np.argpartition(similitudes, -cantidad_top)[-cantidad_top:]
    indices_top = indices_top[np.argsort(similitudes[indices_top])[::-1]]
    top = [
        {
            "clase": str(embedding_labels[idx]),
            "archivo": embedding_files[idx],
            "ruta_dataset": embedding_paths[idx],
            "similitud": float(similitudes[idx]),
        }
        for idx in indices_top
    ]
    return {
        "clase_cnn": clase_cnn,
        "confianza_cnn": confianza,
        "clase_embeddings": clase_embeddings,
        "similitud_embeddings": float(similitud_clase),
        "top_similares": top,
        "votos_embeddings": {clase: float(puntaje) for clase, puntaje in sorted(votos.items())},
    }


def extraer_contenido_ollama(respuesta: Any) -> str:
    mensaje = getattr(respuesta, "message", None)
    if mensaje is not None:
        contenido = getattr(mensaje, "content", None)
        if contenido is not None:
            return str(contenido)
    return str(respuesta["message"]["content"])


def validar_con_gemma(image_path: Path) -> dict[str, Any]:
    if not GEMMA_ENABLED:
        return {
            "disponible": False,
            "validacion": "DUDOSO",
            "clase": "dudoso",
            "tipo_contenido": None,
            "atractivo_visible": None,
            "visibilidad": "nula",
            "personas_como_contexto": False,
            "descripcion_breve": "Gemma está deshabilitado por configuración.",
            "descripcion": "Gemma está deshabilitado por configuración.",
            "respuesta_cruda": "",
        }

    prompt = f"""
Analiza la imagen completa para determinar si es una fotografía directa de un
atractivo turístico natural. No reconozcas identidades, rostros ni personas
concretas. Las personas solo pueden describirse como contexto o como oclusión.

Primero determina tipo_contenido:
- captura_directa: fotografía tomada directamente de la escena real.
- pantalla: se ve un monitor, laptop, celular, televisor o proyección mostrando
  el paisaje.
- documento: el paisaje aparece dentro de una página o documento.
- contenido_impreso: fotografía, folleto, cartel, revista o imagen impresa.
- otro_indirecto: cualquier otra representación del atractivo que no sea una
  captura directa de la escena real.

Evalúa visibilidad de forma conservadora:
- alta: el atractivo está nítido, ocupa una parte importante de la imagen y no
  presenta obstáculos relevantes.
- media: el atractivo se distingue claramente, con desenfoque u obstáculos
  leves que no impiden evaluarlo.
- baja: existe desenfoque de movimiento evidente, oscuridad importante, o las
  personas y objetos ocupan la mayor parte de la imagen y el atractivo solo se
  observa parcialmente o al fondo.
- nula: el atractivo no aparece o no puede verificarse visualmente.

personas_como_contexto=true solamente cuando hay personas visibles y estas no
impiden evaluar el atractivo. Si no aparece ninguna persona, usa false. Si una
multitud o personas en primer plano ocultan casi todo el atractivo, no son solo
contexto: usa personas_como_contexto=false, visibilidad=baja y clase=dudoso.

Reglas obligatorias:
1. Si el contenido principal es pantalla, documento, contenido_impreso u
   otro_indirecto, asigna clase=no_turistico aunque dentro aparezca un paisaje.
2. En una captura_directa, clasifica normalmente solo si el atractivo real es
   visible con visibilidad alta o media.
3. La presencia de una familia o personas como contexto no invalida la foto si
   el atractivo sigue claramente visible. En ese caso usa
   personas_como_contexto=true.
4. Si personas u objetos ocultan casi todo el atractivo, o la imagen está muy
   borrosa y no permite verificarlo, asigna clase=dudoso y visibilidad baja o
   nula.
5. Si no aparece ningún atractivo turístico real, asigna clase=no_turistico y
   atractivo_visible=false.
6. No identifiques ni describas rostros, edades, géneros o identidades.
7. Una cascada o montaña todavía reconocible no implica visibilidad media o
   alta. Si toda la escena muestra movimiento fuerte o falta de nitidez, usa
   visibilidad=baja y clase=dudoso.

Clases turísticas permitidas: {', '.join(CLASES)}.
Usa bosque cuando predominen árboles o vegetación forestal aunque exista un
arroyo. Usa lago solamente para una masa amplia de agua interior. Usa cascada
cuando exista una caída visible de agua. Usa dudoso cuando la visibilidad u
oclusión no permita decidir con seguridad.

Devuelve exclusivamente el objeto JSON del esquema. descripcion_breve debe ser
una descripción general de la escena, sin identificar personas.
""".strip()
    try:
        respuesta = ollama_client.chat(
            model=GEMMA_MODEL,
            messages=[{"role": "user", "content": prompt, "images": [str(image_path)]}],
            format=GEMMA_SCHEMA,
            options={
                "temperature": 0,
                "num_gpu": GEMMA_NUM_GPU,
                "num_ctx": GEMMA_NUM_CTX,
                "num_batch": GEMMA_NUM_BATCH,
            },
            think=False,
        )
        texto = extraer_contenido_ollama(respuesta).strip()
        datos = json.loads(texto)
        resultado = normalizar_respuesta_gemma(datos)
        return {
            "disponible": True,
            **resultado,
            "respuesta_cruda": texto,
        }
    except Exception as error:
        return {
            "disponible": False,
            "validacion": "DUDOSO",
            "clase": "dudoso",
            "tipo_contenido": None,
            "atractivo_visible": None,
            "visibilidad": "nula",
            "personas_como_contexto": False,
            "descripcion_breve": "No fue posible consultar o validar la respuesta de Gemma.",
            "descripcion": "No fue posible consultar o validar la respuesta de Gemma.",
            "respuesta_cruda": f"{type(error).__name__}: {error}",
        }


def fusionar_clases(visual: dict[str, Any], gemma: dict[str, Any]) -> dict[str, Any]:
    senales = [visual["clase_cnn"], visual["clase_embeddings"]]
    if gemma["disponible"] and gemma["validacion"] == "TURISTICO":
        senales.append(gemma["clase"])
    conteo = Counter(senales)
    maximo = max(conteo.values())
    empatadas = [clase for clase, cantidad in conteo.items() if cantidad == maximo]
    clase_final = visual["clase_cnn"] if visual["clase_cnn"] in empatadas else empatadas[0]
    conflicto_gemma = bool(
        gemma["disponible"]
        and gemma["validacion"] == "TURISTICO"
        and gemma["clase"] != clase_final
    )
    return {
        "clase_final": clase_final,
        "senales": senales,
        "acuerdo_visual": int(conteo[clase_final]),
        "conflicto_gemma": conflicto_gemma,
    }


def parse_radio(value: Any) -> float:
    try:
        if pd.isna(value):
            return 3.0
        return float(str(value).lower().replace("km", "").strip())
    except (TypeError, ValueError):
        return 3.0


def es_subtipo_compatible(clase: str, subtipo: Any) -> bool:
    texto = normalizar_texto(subtipo)
    return any(valor in texto for valor in CLASE_A_SUBTIPOS.get(clase, []))


def buscar_gis(clase: str, lat: float, lon: float, alt: float) -> list[dict[str, Any]]:
    candidatos: list[dict[str, Any]] = []
    for _, row in gis_df.iterrows():
        try:
            if not es_subtipo_compatible(clase, row.get("SubTipo")):
                continue
            if pd.isna(row.get("Latitud")) or pd.isna(row.get("Longitud")):
                continue
            lugar_lat = float(row["Latitud"])
            lugar_lon = float(row["Longitud"])
            distancia = float(geodesic((lat, lon), (lugar_lat, lugar_lon)).km)
            radio = parse_radio(row.get("Radio"))
            if pd.isna(row.get("Altitud")):
                lugar_alt = None
                diferencia_altitud = None
                altitud_valida = False
            else:
                lugar_alt = float(row["Altitud"])
                diferencia_altitud = abs(float(alt) - lugar_alt)
                altitud_valida = diferencia_altitud <= MAX_ALTITUDE_DIFF
            candidatos.append(
                {
                    "nombre": row.get("Nombre"),
                    "subtipo": row.get("SubTipo"),
                    "departamento": row.get("Departamento"),
                    "provincia": row.get("Provincia"),
                    "distrito": row.get("Distrito"),
                    "latitud": lugar_lat,
                    "longitud": lugar_lon,
                    "altitud": lugar_alt,
                    "radio_km": radio,
                    "distancia_km": round(distancia, 3),
                    "diferencia_altitud_m": round(diferencia_altitud, 2) if diferencia_altitud is not None else None,
                    "dentro_radio": distancia <= radio,
                    "altitud_valida": altitud_valida,
                    "subtipo_compatible": True,
                }
            )
        except (TypeError, ValueError):
            continue
    return sorted(candidatos, key=lambda candidato: candidato["distancia_km"])


def elegir_lugar(candidatos: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (
            candidato
            for candidato in candidatos
            if candidato["dentro_radio"] and candidato["altitud_valida"]
        ),
        None,
    )


def calcular_score(
    visual: dict[str, Any],
    fusion: dict[str, Any],
    gemma: dict[str, Any],
    lugar: dict[str, Any] | None,
) -> tuple[int, dict[str, int]]:
    confianza = visual["confianza_cnn"]
    score_cnn = 20 if confianza >= 0.95 else 16 if confianza >= 0.85 else 12 if confianza >= 0.70 else 6 if confianza >= 0.50 else 0
    similitud = visual["similitud_embeddings"]
    score_similitud = 20 if similitud >= 0.95 else 16 if similitud >= 0.90 else 12 if similitud >= 0.85 else 6 if similitud >= 0.75 else 0
    acuerdo = fusion["acuerdo_visual"]
    score_acuerdo = 15 if acuerdo >= 3 else 10 if acuerdo == 2 else 0
    score_distancia = 0
    score_altitud = 0
    score_gis = 0
    if lugar:
        distancia = lugar["distancia_km"]
        score_distancia = 20 if distancia <= 1 else 15 if distancia <= 3 else 10
        diferencia = lugar.get("diferencia_altitud_m")
        if diferencia is not None:
            score_altitud = 10 if diferencia <= 50 else 7 if diferencia <= 150 else 5
        score_gis = 10
    score_gemma = 0
    if gemma["disponible"] and gemma["validacion"] == "TURISTICO":
        score_gemma = 5 if not fusion["conflicto_gemma"] else 2
    detalle = {
        "cnn_confidence": score_cnn,
        "visual_similarity": score_similitud,
        "visual_agreement": score_acuerdo,
        "gps_distance": score_distancia,
        "altitude": score_altitud,
        "gis_validation": score_gis,
        "gemma_validation": score_gemma,
    }
    return int(sum(detalle.values())), detalle


def decidir_estado(
    visual: dict[str, Any],
    fusion: dict[str, Any],
    gemma: dict[str, Any],
    lugar: dict[str, Any] | None,
    score: int,
) -> tuple[str, str]:
    estado_semantico = decidir_estado_semantico(gemma)
    if estado_semantico is not None:
        return estado_semantico
    if fusion["conflicto_gemma"]:
        return "CASO_DUDOSO", "Gemma y los clasificadores visuales no coinciden en la clase dominante."
    if lugar and fusion["acuerdo_visual"] >= 2 and score >= 70:
        return (
            "LUGAR_REGISTRADO",
            "Existe acuerdo visual y coincidencia geográfica y altimétrica con un lugar registrado.",
        )
    if not lugar and fusion["acuerdo_visual"] >= 2 and score >= 45:
        return (
            "POSIBLE_NUEVO_ATRACTIVO",
            "La imagen es turísticamente coherente, pero no coincide con un lugar GIS cercano.",
        )
    return (
        "CASO_DUDOSO",
        "La evidencia visual, semántica o geográfica no es suficiente para decidir automáticamente.",
    )


def procesar_imagen(file_bytes: bytes, filename: str, lat: float, lon: float, alt: float) -> dict[str, Any]:
    img_array = cargar_imagen(file_bytes)
    temporal = guardar_imagen_temporal(file_bytes, filename)
    try:
        visual = analizar_imagen(img_array)
        gemma = validar_con_gemma(temporal)
        gemma = ajustar_gemma_con_evidencia_visual(gemma, visual["confianza_cnn"])
        fusion = fusionar_clases(visual, gemma)
        candidatos = buscar_gis(fusion["clase_final"], lat, lon, alt)
        lugar = elegir_lugar(candidatos)
        score, detalle = calcular_score(visual, fusion, gemma, lugar)
        estado, motivo = decidir_estado(visual, fusion, gemma, lugar, score)
        return limpiar_json(
            {
                "version_motor": MOTOR_VERSION,
                "archivo": filename,
                "estado": estado,
                "motivo_decision": motivo,
                "motor_completo": bool(gemma["disponible"]),
                "score_total": score,
                "detalle_score": detalle,
                "clasificacion_modelo": visual["clase_cnn"],
                "confianza_modelo": round(visual["confianza_cnn"] * 100, 2),
                "clase_embeddings": visual["clase_embeddings"],
                "similitud_visual": round(visual["similitud_embeddings"] * 100, 2),
                "gemma": gemma,
                "fusion": fusion,
                "clase_final_gis": fusion["clase_final"],
                "gps_usuario": {"latitud": lat, "longitud": lon, "altitud": alt},
                "lugar_registrado": lugar,
                "top_similares": [
                    {**resultado, "similitud": round(resultado["similitud"] * 100, 2)}
                    for resultado in visual["top_similares"]
                ],
                "candidatos_gis": candidatos[:5],
            }
        )
    finally:
        temporal.unlink(missing_ok=True)


def estado_ollama() -> dict[str, Any]:
    if not GEMMA_ENABLED:
        return {
            "servicio_disponible": False,
            "modelo_configurado": GEMMA_MODEL,
            "modelo_instalado": False,
            "detalle": "Gemma deshabilitado mediante GEMMA_ENABLED.",
        }
    try:
        respuesta = ollama_client.list()
        modelos = getattr(respuesta, "models", None)
        if modelos is None and isinstance(respuesta, dict):
            modelos = respuesta.get("models", [])
        nombres = []
        for item in modelos or []:
            nombre = getattr(item, "model", None)
            if nombre is None and isinstance(item, dict):
                nombre = item.get("model") or item.get("name")
            if nombre:
                nombres.append(str(nombre))
        return {
            "servicio_disponible": True,
            "modelo_configurado": GEMMA_MODEL,
            "modelo_instalado": GEMMA_MODEL in nombres,
            "modelos_disponibles": nombres,
        }
    except Exception as error:
        return {
            "servicio_disponible": False,
            "modelo_configurado": GEMMA_MODEL,
            "modelo_instalado": False,
            "detalle": f"{type(error).__name__}: {error}",
        }


@app.get("/")
def inicio():
    return {
        "mensaje": "Motor IA de Turismo funcionando",
        "version": MOTOR_VERSION,
        "clases": CLASES,
        "documentacion": "/docs",
        "salud": "/health",
    }


@app.get("/health")
def health():
    gemma = estado_ollama()
    return {
        "estado": "OK" if gemma["servicio_disponible"] and gemma["modelo_instalado"] else "DEGRADADO",
        "modelo_cnn": MODEL_PATH.name,
        "salidas_modelo": int(model.output_shape[-1]),
        "embeddings": int(len(embeddings_matrix)),
        "dimension_embeddings": int(embeddings_matrix.shape[1]),
        "lugares_gis": int(len(gis_df)),
        "gemma": gemma,
    }


@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    lat: float = Form(...),
    lon: float = Form(...),
    alt: float = Form(...),
):
    if not -90 <= lat <= 90:
        raise HTTPException(status_code=422, detail="La latitud debe estar entre -90 y 90")
    if not -180 <= lon <= 180:
        raise HTTPException(status_code=422, detail="La longitud debe estar entre -180 y 180")
    if not -500 <= alt <= 9000:
        raise HTTPException(status_code=422, detail="La altitud debe estar entre -500 y 9000 m")
    contenido = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contenido:
        raise HTTPException(status_code=400, detail="El archivo está vacío")
    if len(contenido) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="La imagen supera el máximo de 15 MB")
    return await run_in_threadpool(
        procesar_imagen,
        contenido,
        file.filename or "imagen",
        lat,
        lon,
        alt,
    )


@app.get("/lugares")
def listar_lugares():
    columnas = ["Código", "Nombre", "Departamento", "Provincia", "Distrito", "Latitud", "Longitud"]
    existentes = [columna for columna in columnas if columna in gis_df.columns]
    return limpiar_json({"total": int(len(gis_df)), "lugares": gis_df[existentes].to_dict(orient="records")})


@app.get("/lugares/buscar/{texto}")
def buscar_lugares(texto: str):
    consulta = texto.strip()
    if not consulta:
        raise HTTPException(status_code=422, detail="La búsqueda no puede estar vacía")
    mascara = pd.Series(False, index=gis_df.index)
    for columna in ("Nombre", "Departamento", "Provincia", "Distrito", "SubTipo"):
        if columna in gis_df.columns:
            mascara |= gis_df[columna].astype(str).str.contains(consulta, case=False, na=False, regex=False)
    columnas = ["Código", "Nombre", "Departamento", "Provincia", "Distrito", "SubTipo", "Latitud", "Longitud"]
    existentes = [columna for columna in columnas if columna in gis_df.columns]
    data = gis_df.loc[mascara, existentes].to_dict(orient="records")
    return limpiar_json({"busqueda": consulta, "total": len(data), "resultados": data})


@app.get("/lugares/{codigo}")
def obtener_lugar(codigo: int):
    if "Código" not in gis_df.columns:
        raise HTTPException(status_code=500, detail="No existe la columna Código")
    codigos = pd.to_numeric(gis_df["Código"], errors="coerce")
    lugar = gis_df[codigos == codigo]
    if lugar.empty:
        raise HTTPException(status_code=404, detail="Lugar no encontrado")
    return limpiar_json(lugar.iloc[0].to_dict())
