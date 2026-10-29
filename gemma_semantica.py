"""Reglas deterministas para interpretar la salida semántica de Gemma."""

from __future__ import annotations

import unicodedata
from typing import Any


CLASES_TURISTICAS = (
    "bosque",
    "cascada",
    "desierto",
    "lago",
    "montana",
    "nevado",
    "playa",
    "volcan",
)
CLASES_GEMMA = CLASES_TURISTICAS + ("no_turistico", "dudoso")
TIPOS_CONTENIDO = (
    "captura_directa",
    "pantalla",
    "documento",
    "contenido_impreso",
    "otro_indirecto",
)
TIPOS_CONTENIDO_INDIRECTO = frozenset(TIPOS_CONTENIDO) - {"captura_directa"}
VISIBILIDADES = ("alta", "media", "baja", "nula")
UMBRAL_CONFIANZA_VISUAL_DUDOSA = 0.70


def _normalizar_texto(valor: Any) -> str:
    texto = str(valor).strip().lower()
    return "".join(
        caracter
        for caracter in unicodedata.normalize("NFD", texto)
        if unicodedata.category(caracter) != "Mn"
    )


def normalizar_respuesta_gemma(datos: dict[str, Any]) -> dict[str, Any]:
    """Valida y normaliza el JSON de Gemma aplicando reglas conservadoras."""

    if not isinstance(datos, dict):
        raise ValueError("La respuesta de Gemma debe ser un objeto JSON")

    clase = _normalizar_texto(datos.get("clase", ""))
    tipo_contenido = _normalizar_texto(datos.get("tipo_contenido", ""))
    visibilidad = _normalizar_texto(datos.get("visibilidad", ""))
    atractivo_visible = datos.get("atractivo_visible")
    personas_como_contexto = datos.get("personas_como_contexto")
    descripcion = str(datos.get("descripcion_breve", "")).strip()

    if clase not in CLASES_GEMMA:
        raise ValueError(f"Clase Gemma desconocida: {clase}")
    if tipo_contenido not in TIPOS_CONTENIDO:
        raise ValueError(f"Tipo de contenido desconocido: {tipo_contenido}")
    if visibilidad not in VISIBILIDADES:
        raise ValueError(f"Visibilidad desconocida: {visibilidad}")
    if not isinstance(atractivo_visible, bool):
        raise ValueError("atractivo_visible debe ser booleano")
    if not isinstance(personas_como_contexto, bool):
        raise ValueError("personas_como_contexto debe ser booleano")
    if not descripcion:
        raise ValueError("descripcion_breve no puede estar vacía")

    # Una representación indirecta nunca se acepta como fotografía del atractivo,
    # aunque la escena mostrada dentro de ella sea turística.
    if tipo_contenido in TIPOS_CONTENIDO_INDIRECTO:
        clase = "no_turistico"
    elif not atractivo_visible:
        clase = "no_turistico"
        visibilidad = "nula"
    elif clase in CLASES_TURISTICAS:
        # La baja visibilidad incluye desenfoque u oclusión severa. Estos casos se
        # reservan para revisión humana, incluso si Gemma propone una clase.
        if visibilidad in {"baja", "nula"}:
            clase = "dudoso"

    if clase in CLASES_TURISTICAS:
        validacion = "TURISTICO"
    elif clase == "no_turistico":
        validacion = "NO_TURISTICO"
    else:
        validacion = "DUDOSO"

    return {
        "validacion": validacion,
        "clase": clase,
        "tipo_contenido": tipo_contenido,
        "atractivo_visible": atractivo_visible,
        "visibilidad": visibilidad,
        "personas_como_contexto": personas_como_contexto,
        "descripcion_breve": descripcion,
        "descripcion": descripcion,
    }


def decidir_estado_semantico(gemma: dict[str, Any]) -> tuple[str, str] | None:
    """Devuelve un estado terminal de Gemma o None para seguir con CNN/GIS."""

    if not gemma.get("disponible", False):
        return (
            "CASO_DUDOSO",
            "CNN, embeddings y GIS fueron evaluados, pero Gemma no está disponible o su respuesta no es válida; se requiere revisión humana.",
        )

    tipo_contenido = gemma.get("tipo_contenido")
    if tipo_contenido in TIPOS_CONTENIDO_INDIRECTO:
        return (
            "IMAGEN_NO_VALIDA",
            "La imagen corresponde a una pantalla, documento o contenido indirecto, no a una captura directa del atractivo.",
        )

    if gemma.get("clase") == "no_turistico" or gemma.get("validacion") == "NO_TURISTICO":
        return "IMAGEN_NO_VALIDA", "Gemma no detectó un atractivo turístico real y visible."

    if gemma.get("clase") == "dudoso" or gemma.get("validacion") == "DUDOSO":
        return (
            "CASO_DUDOSO",
            "Gemma consideró ambigua la imagen por baja visibilidad, desenfoque u oclusión del atractivo.",
        )

    return None


def ajustar_gemma_con_evidencia_visual(
    gemma: dict[str, Any],
    confianza_cnn: float,
) -> dict[str, Any]:
    """Vuelve dudosa una aceptación semántica débil respaldada por un CNN débil."""

    ajustada = dict(gemma)
    if not ajustada.get("disponible") or ajustada.get("validacion") != "TURISTICO":
        return ajustada
    if float(confianza_cnn) >= UMBRAL_CONFIANZA_VISUAL_DUDOSA:
        return ajustada

    visibilidad = ajustada.get("visibilidad")
    personas = ajustada.get("personas_como_contexto") is True
    if visibilidad != "media" and not personas:
        return ajustada

    clase_original = ajustada.get("clase")
    ajustada.update(
        {
            "validacion": "DUDOSO",
            "clase": "dudoso",
            "visibilidad": "baja",
            "ajuste_semantico": (
                "La aceptación de Gemma se degradó a dudosa porque la escena tiene "
                "visibilidad limitada o personas en primer plano y la confianza del "
                f"clasificador visual fue {float(confianza_cnn) * 100:.2f}%."
            ),
            "clase_original_gemma": clase_original,
        }
    )
    if personas:
        ajustada["personas_como_contexto"] = False
    return ajustada
