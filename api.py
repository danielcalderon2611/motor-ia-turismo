import io
import os
import uuid
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import ollama

from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic

# =========================================
# FASTAPI
# =========================================

app = FastAPI(title="API Turismo IA")

# ==========================
# BASE DE LUGARES TURÍSTICOS
# ==========================

EXCEL_LUGARES = "lugares_turisticos_app_final_v3.xlsx"

lugares_df = pd.read_excel(EXCEL_LUGARES)

lugares_df.columns = [
    str(col).strip()
    for col in lugares_df.columns
]

lugares_df = lugares_df.fillna("No disponible")

# =========================================
# CONFIGURACIÓN
# =========================================

MODEL_PATH = "models/modelo_mobilenetv2_final.keras"
EMBEDDINGS_PATH = "fewshot_db/embeddings.pkl"
EXCEL_PATH = EXCEL_LUGARES

IMG_SIZE = (224, 224)
TOP_K = 5
MAX_ALTITUDE_DIFF = 300

GEMMA_MODEL = "gemma4:e2b"
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# =========================================
# MAPEO CLASE IA / GEMMA → SUBTIPO GIS
# =========================================

CLASE_A_SUBTIPOS = {
    "lago": ["lagos", "lagunas"],
    "rio": ["ríos", "rios", "corrientes de agua"],
    "playa": ["playas", "costas"],
    "bosque": ["bosques", "zonas paisajísticas", "zona paisajística"],
    "sendero": ["bosques", "zonas paisajísticas", "zona paisajística"],
    "mirador": ["miradores", "zonas paisajísticas", "zona paisajística"],
    "montana": ["cerros", "montañas", "cordilleras"],
    "montaña": ["cerros", "montañas", "cordilleras"],
    "nevado": ["nevados"],
    "volcan": ["volcanes"],
    "volcán": ["volcanes"],
    "cascada": ["cascadas", "caídas de agua", "caida de agua"],
    "desierto": ["desiertos", "planicies"],
    "ruinas": ["sitios arqueológicos", "restos arqueológicos"],
    "iglesia": ["iglesias", "templos"],
}

CLASES_GEMMA_VALIDAS = [
    "bosque",
    "rio",
    "lago",
    "playa",
    "montana",
    "montaña",
    "nevado",
    "volcan",
    "volcán",
    "cascada",
    "desierto",
    "ruinas",
    "sendero",
    "mirador",
]

# =========================================
# LIMPIEZA JSON
# =========================================

def limpiar_json(obj):
    if isinstance(obj, dict):
        return {k: limpiar_json(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [limpiar_json(v) for v in obj]

    if isinstance(obj, (np.float32, np.float64)):
        value = float(obj)
        return None if np.isnan(value) or np.isinf(value) else value

    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)

    if isinstance(obj, float):
        return None if np.isnan(obj) or np.isinf(obj) else obj

    return obj

# =========================================
# CARGA GLOBAL
# =========================================

print("Cargando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)

dummy_input = np.zeros((1, 224, 224, 3))
model.predict(dummy_input, verbose=0)

embedding_model = Model(
    inputs=model.inputs,
    outputs=model.layers[-2].output
)

print("Cargando embeddings...")
with open(EMBEDDINGS_PATH, "rb") as f:
    embeddings_db = pickle.load(f)

class_names = sorted(list(set(item["class"] for item in embeddings_db)))
print("Clases detectadas:", class_names)

print("Cargando GIS...")
gis_df = pd.read_excel(EXCEL_PATH)
gis_df.columns = [str(col).strip() for col in gis_df.columns]

print("API lista.")

# =========================================
# IMAGEN
# =========================================

def guardar_imagen_temporal(file_bytes, filename):
    ext = os.path.splitext(filename)[1]

    if ext == "":
        ext = ".jpg"

    temp_name = f"{uuid.uuid4()}{ext}"
    temp_path = os.path.join(TEMP_DIR, temp_name)

    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    return temp_path


def cargar_imagen(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array

# =========================================
# GEMMA 4 - VALIDADOR OFICIAL TURÍSTICO
# =========================================

def validar_con_gemma(image_path, predicted_class, confidence, best_class, similarity):
    prompt = f"""
Analiza la imagen como validador turístico.

Contexto del sistema:
- Clase MobileNetV2: {predicted_class}
- Confianza MobileNetV2: {confidence * 100:.2f}%
- Clase por embeddings: {best_class}
- Similitud visual: {similarity * 100:.2f}%

Debes responder exactamente con este formato:

VALIDACION: TURISTICO / NO_TURISTICO / DUDOSO
CLASE_GEMMA: bosque / rio / lago / playa / montana / nevado / volcan / cascada / desierto / ruinas / sendero / mirador / ciudad / no_turistico / dudoso

Reglas:
- TURISTICO: paisaje natural, bosque, río, lago, montaña, playa, sendero, mirador, catarata, nevado, volcán, ruinas o atractivo turístico.
- TURISTICO también aplica si hay personas, siempre que el paisaje sea el elemento principal.
- NO_TURISTICO: selfie cerrada, comida, carro, mascota, meme, captura de pantalla, documento, interior de casa, publicidad u objeto común.
- DUDOSO: imagen borrosa, ambigua o difícil de interpretar.
"""

    try:
        response = ollama.chat(
            model=GEMMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_path],
                }
            ],
        )

        text = response["message"]["content"].strip().lower()

        validacion = "DUDOSO"
        clase_gemma = "dudoso"

        if "validacion:" in text:
            linea_validacion = [
                line for line in text.splitlines()
                if "validacion:" in line
            ][0]

            valor = linea_validacion.split(":", 1)[1].strip().upper()

            if "NO_TURISTICO" in valor:
                validacion = "NO_TURISTICO"
            elif "TURISTICO" in valor:
                validacion = "TURISTICO"
            elif "DUDOSO" in valor:
                validacion = "DUDOSO"

        if "clase_gemma:" in text:
            linea_clase = [
                line for line in text.splitlines()
                if "clase_gemma:" in line
            ][0]

            clase_gemma = linea_clase.split(":", 1)[1].strip().lower()

        return {
            "validacion": validacion,
            "clase_gemma": clase_gemma,
            "respuesta_cruda": text,
        }

    except Exception as e:
        print("Error Gemma:", e)

        return {
            "validacion": "DUDOSO",
            "clase_gemma": "dudoso",
            "respuesta_cruda": str(e),
        }

# =========================================
# ANALIZAR IMAGEN
# =========================================

def analizar_imagen(img_array):
    predictions = model.predict(img_array, verbose=0)[0]

    confidence = float(np.max(predictions))
    predicted_index = int(np.argmax(predictions))

    if predicted_index < len(class_names):
        predicted_class = class_names[predicted_index]
    else:
        predicted_class = "desconocido"

    query_embedding = embedding_model.predict(img_array, verbose=0)[0]

    results = []

    for item in embeddings_db:
        similarity = cosine_similarity(
            [query_embedding],
            [item["embedding"]]
        )[0][0]

        results.append({
            "class": item["class"],
            "file": item["file"],
            "path": item["path"],
            "similarity": float(similarity),
        })

    results = sorted(
        results,
        key=lambda x: x["similarity"],
        reverse=True
    )

    return predicted_class, confidence, results[:TOP_K]

# =========================================
# GIS
# =========================================

def parse_radio(value):
    try:
        if pd.isna(value):
            return 3.0

        return float(
            str(value)
            .replace("km", "")
            .replace("KM", "")
            .strip()
        )
    except:
        return 3.0


def es_subtipo_compatible(clase, subtipo):
    clase = str(clase).lower().strip()
    subtipo = str(subtipo).lower().strip()

    subtipos_validos = CLASE_A_SUBTIPOS.get(clase, [])

    for valido in subtipos_validos:
        if valido in subtipo:
            return True

    return False


def elegir_clase_final_gis(gemma_validacion, clase_gemma, best_class):
    clase_gemma = str(clase_gemma).lower().strip()
    best_class = str(best_class).lower().strip()

    if (
        gemma_validacion == "TURISTICO"
        and clase_gemma in CLASES_GEMMA_VALIDAS
    ):
        return clase_gemma

    return best_class


def buscar_gis(clase, lat, lon, alt):
    candidatos = []

    for _, row in gis_df.iterrows():
        try:
            subtipo = row.get("SubTipo")

            if not es_subtipo_compatible(clase, subtipo):
                continue

            if pd.isna(row.get("Latitud")) or pd.isna(row.get("Longitud")):
                continue

            lugar_lat = float(row["Latitud"])
            lugar_lon = float(row["Longitud"])

            if pd.isna(row.get("Altitud")):
                lugar_alt = None
                diferencia_altitud = None
                altitud_valida = False
            else:
                lugar_alt = float(row["Altitud"])
                diferencia_altitud = abs(alt - lugar_alt)
                altitud_valida = diferencia_altitud <= MAX_ALTITUDE_DIFF

            radio_km = parse_radio(row.get("Radio"))

            distancia = geodesic(
                (lat, lon),
                (lugar_lat, lugar_lon)
            ).km

            candidatos.append({
                "nombre": row.get("Nombre"),
                "subtipo": row.get("SubTipo"),
                "departamento": row.get("Departamento"),
                "provincia": row.get("Provincia"),
                "distrito": row.get("Distrito"),
                "latitud": lugar_lat,
                "longitud": lugar_lon,
                "altitud": lugar_alt,
                "radio_km": radio_km,
                "distancia_km": round(distancia, 3),
                "diferencia_altitud_m": (
                    round(diferencia_altitud, 2)
                    if diferencia_altitud is not None
                    else None
                ),
                "dentro_radio": distancia <= radio_km,
                "altitud_valida": altitud_valida,
                "subtipo_compatible": True,
            })

        except Exception:
            continue

    candidatos = sorted(
        candidatos,
        key=lambda x: x["distancia_km"]
    )

    return candidatos

# =========================================
# SCORE MULTIMODAL
# =========================================

def calcular_score_multimodal(confidence, best_similarity, lugar_registrado, gemma_validacion):
    score_total = 0
    detalle_score = {}

    confidence_percent = confidence * 100
    similarity_percent = best_similarity * 100

    # CNN: 25 puntos
    if confidence_percent >= 90:
        score_cnn = 25
    elif confidence_percent >= 80:
        score_cnn = 20
    elif confidence_percent >= 70:
        score_cnn = 15
    elif confidence_percent >= 60:
        score_cnn = 10
    else:
        score_cnn = 0

    score_total += score_cnn
    detalle_score["cnn_confidence"] = score_cnn

    # Similaridad visual: 25 puntos
    if similarity_percent >= 95:
        score_similarity = 25
    elif similarity_percent >= 90:
        score_similarity = 20
    elif similarity_percent >= 85:
        score_similarity = 15
    elif similarity_percent >= 75:
        score_similarity = 10
    else:
        score_similarity = 0

    score_total += score_similarity
    detalle_score["visual_similarity"] = score_similarity

    # Distancia GPS: 20 puntos
    score_distance = 0

    if lugar_registrado:
        distancia = lugar_registrado.get("distancia_km")

        if distancia is not None:
            if distancia <= 1:
                score_distance = 20
            elif distancia <= 3:
                score_distance = 15
            elif distancia <= 10:
                score_distance = 10
            elif distancia <= 30:
                score_distance = 5

    score_total += score_distance
    detalle_score["gps_distance"] = score_distance

    # Altitud: 10 puntos
    score_altitude = 0

    if lugar_registrado:
        diff_alt = lugar_registrado.get("diferencia_altitud_m")

        if diff_alt is not None:
            if diff_alt <= 50:
                score_altitude = 10
            elif diff_alt <= 150:
                score_altitude = 7
            elif diff_alt <= 300:
                score_altitude = 5

    score_total += score_altitude
    detalle_score["altitude"] = score_altitude

    # GIS: 10 puntos
    score_gis = 10 if lugar_registrado else 0
    score_total += score_gis
    detalle_score["gis_validation"] = score_gis

    # Gemma: 10 puntos
    if gemma_validacion == "TURISTICO":
        score_gemma = 10
    elif gemma_validacion == "DUDOSO":
        score_gemma = 5
    else:
        score_gemma = 0

    score_total += score_gemma
    detalle_score["gemma_validation"] = score_gemma

    return score_total, detalle_score

# =========================================
# RUTAS
# =========================================

@app.get("/")
def inicio():
    return {
        "mensaje": "API Turismo IA funcionando"
    }


@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    lat: float = Form(...),
    lon: float = Form(...),
    alt: float = Form(...),
):
    file_bytes = await file.read()
    temp_path = guardar_imagen_temporal(file_bytes, file.filename)

    try:
        img_array = cargar_imagen(file_bytes)

        predicted_class, confidence, top_results = analizar_imagen(img_array)

        best_result = top_results[0]
        best_class = best_result["class"]
        best_similarity = best_result["similarity"]

        gemma_resultado = validar_con_gemma(
            temp_path,
            predicted_class,
            confidence,
            best_class,
            best_similarity,
        )

        gemma_validacion = gemma_resultado["validacion"]
        clase_gemma = gemma_resultado["clase_gemma"]

        clase_final_gis = elegir_clase_final_gis(
            gemma_validacion,
            clase_gemma,
            best_class,
        )

        candidatos_gis = buscar_gis(
            clase_final_gis,
            lat,
            lon,
            alt,
        )

        lugar_registrado = None

        for candidato in candidatos_gis:
            if candidato["dentro_radio"] and candidato["altitud_valida"]:
                lugar_registrado = candidato
                break

        score_total, detalle_score = calcular_score_multimodal(
            confidence,
            best_similarity,
            lugar_registrado,
            gemma_validacion,
        )
        if gemma_validacion == "NO_TURISTICO":
            estado = "IMAGEN_NO_VALIDA"
            motivo = "Gemma 4 validó que la imagen no corresponde a un entorno turístico o paisajístico."

        elif gemma_validacion == "DUDOSO" and score_total < 70:
            estado = "CASO_DUDOSO"
            motivo = "Gemma 4 consideró la imagen ambigua y el score no fue suficiente para aceptarla automáticamente."

        elif score_total >= 85:
            estado = "LUGAR_REGISTRADO"
            motivo = "La imagen coincide visual, geográfica y topográficamente con un atractivo turístico registrado."

        elif lugar_registrado and gemma_validacion == "TURISTICO" and score_total >= 70:
             estado = "LUGAR_REGISTRADO"
             motivo = "Gemma 4 validó la imagen como turística y existe coincidencia geoespacial con un lugar registrado."

        # NUEVA REGLA
        elif gemma_validacion == "TURISTICO" and lugar_registrado is None:
            estado = "POSIBLE_NUEVO_ATRACTIVO"
            motivo = "Gemma 4 validó la imagen como turística, pero no existe coincidencia geográfica cercana en la base registrada."

        else:
            estado = "IMAGEN_NO_VALIDA"
            motivo = "La imagen no presenta suficiente coherencia visual, geográfica o semántica."

        respuesta = {
            "archivo": file.filename,
            "estado": estado,
            "motivo_decision": motivo,
            "score_total": score_total,
            "detalle_score": detalle_score,
            "gemma_validacion": gemma_validacion,
            "clase_visual_gemma": clase_gemma,
            "clase_final_gis": clase_final_gis,
            "gemma_respuesta_cruda": gemma_resultado["respuesta_cruda"],
            "clasificacion_modelo": predicted_class,
            "confianza_modelo": round(confidence * 100, 2),
            "mejor_clase_visual": best_class,
            "similitud_visual": round(best_similarity * 100, 2),
            "gps_usuario": {
                "latitud": lat,
                "longitud": lon,
                "altitud": alt,
            },
            "lugar_registrado": lugar_registrado,
            "top_similares": [
                {
                    "clase": r["class"],
                    "archivo": r["file"],
                    "similitud": round(r["similarity"] * 100, 2),
                }
                for r in top_results
            ],
            "candidatos_gis": candidatos_gis[:5],
        }

        return limpiar_json(respuesta)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
@app.get("/lugares")
def listar_lugares():

    columnas = [
        "Código",
        "Nombre",
        "Departamento",
        "Provincia",
        "Distrito",
        "Latitud",
        "Longitud",
    ]

    columnas_existentes = [
        c for c in columnas
        if c in lugares_df.columns
    ]

    data = lugares_df[
        columnas_existentes
    ].to_dict(
        orient="records"
    )

    return {
        "total": len(data),
        "lugares": data
    }            
@app.get("/lugares/{codigo}")
def obtener_lugar(codigo: int):

    if "Código" not in lugares_df.columns:
        raise HTTPException(
            status_code=500,
            detail="No existe columna Código"
        )

    lugar = lugares_df[
        lugares_df["Código"] == codigo
    ]

    if lugar.empty:
        raise HTTPException(
            status_code=404,
            detail="Lugar no encontrado"
        )

    return lugar.iloc[0].to_dict()
@app.get("/lugares/buscar/{texto}")
def buscar_lugares(texto: str):
    texto = texto.lower().strip()

    resultados = lugares_df[
        lugares_df["Nombre"].astype(str).str.lower().str.contains(texto, na=False)
        |
        lugares_df["Departamento"].astype(str).str.lower().str.contains(texto, na=False)
        |
        lugares_df["Provincia"].astype(str).str.lower().str.contains(texto, na=False)
        |
        lugares_df["Distrito"].astype(str).str.lower().str.contains(texto, na=False)
        |
        lugares_df["SubTipo"].astype(str).str.lower().str.contains(texto, na=False)
    ]

    columnas = [
        "Código",
        "Nombre",
        "Departamento",
        "Provincia",
        "Distrito",
        "SubTipo",
        "Latitud",
        "Longitud",
    ]

    columnas_existentes = [
        c for c in columnas
        if c in lugares_df.columns
    ]

    data = resultados[columnas_existentes].to_dict(orient="records")

    return {
        "busqueda": texto,
        "total": len(data),
        "resultados": data
    }
