"""Ejecuta las pruebas visuales del contrato semántico de Gemma."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import api


ESTADOS_ESPERADOS = {
    "01_": {"IMAGEN_NO_VALIDA"},
    "02_": {"IMAGEN_NO_VALIDA"},
    "03_": {"LUGAR_REGISTRADO", "POSIBLE_NUEVO_ATRACTIVO"},
    "04_": {"CASO_DUDOSO"},
    "05_": {"LUGAR_REGISTRADO", "POSIBLE_NUEVO_ATRACTIVO"},
    "06_": {"CASO_DUDOSO"},
    "07_": {"IMAGEN_NO_VALIDA"},
}


def coordenadas_cascada() -> tuple[float, float, float]:
    for _, fila in api.gis_df.iterrows():
        if api.es_subtipo_compatible("cascada", fila.get("SubTipo")):
            return float(fila["Latitud"]), float(fila["Longitud"]), float(fila["Altitud"])
    raise RuntimeError("No se encontró una cascada con coordenadas en la base GIS")


def esperado_para(nombre: str) -> set[str]:
    for prefijo, estados in ESTADOS_ESPERADOS.items():
        if nombre.startswith(prefijo):
            return estados
    raise ValueError(f"Nombre de prueba sin escenario reconocido: {nombre}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("carpeta", type=Path)
    parser.add_argument(
        "--solo",
        nargs="*",
        default=[],
        help="Prefijos de escenarios que deben ejecutarse, por ejemplo 04_ 06_",
    )
    args = parser.parse_args()

    imagenes = sorted(
        ruta
        for ruta in args.carpeta.rglob("*")
        if ruta.is_file() and ruta.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if args.solo:
        imagenes = [ruta for ruta in imagenes if any(ruta.name.startswith(prefijo) for prefijo in args.solo)]
    if not imagenes:
        raise SystemExit("No se encontraron imágenes")

    lat, lon, alt = coordenadas_cascada()
    aprobadas = 0
    print(f"Pruebas: {len(imagenes)} | Coordenadas GIS: {lat}, {lon}, {alt}", flush=True)

    for indice, ruta in enumerate(imagenes, start=1):
        print(f"\n[{indice}/{len(imagenes)}] {ruta.name}", flush=True)
        resultado = api.procesar_imagen(ruta.read_bytes(), ruta.name, lat, lon, alt)
        esperados = esperado_para(ruta.name)
        aprobado = resultado["estado"] in esperados
        aprobadas += int(aprobado)
        resumen = {
            "archivo": ruta.name,
            "estado": resultado["estado"],
            "esperados": sorted(esperados),
            "prueba_aprobada": aprobado,
            "motivo_decision": resultado["motivo_decision"],
            "clasificacion_modelo": resultado["clasificacion_modelo"],
            "confianza_modelo": resultado["confianza_modelo"],
            "clase_final_gis": resultado["clase_final_gis"],
            "score_total": resultado["score_total"],
            "gemma": resultado["gemma"],
        }
        print(json.dumps(resumen, ensure_ascii=False, indent=2), flush=True)

    print(f"\nRESULTADO: {aprobadas}/{len(imagenes)} pruebas aprobadas", flush=True)
    return 0 if aprobadas == len(imagenes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
