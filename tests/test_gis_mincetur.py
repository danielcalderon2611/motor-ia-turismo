from pathlib import Path

import pandas as pd


GIS_PATH = Path(__file__).resolve().parents[1] / "lugares_turisticos_app_final_v4.xlsx"


def test_inventario_gis_auditado_tiene_estructura_valida():
    assert GIS_PATH.is_file()

    datos = pd.read_excel(GIS_PATH)
    requeridas = {
        "Código",
        "Nombre",
        "Departamento",
        "Provincia",
        "Distrito",
        "SubTipo",
        "Latitud",
        "Longitud",
        "Altitud",
    }

    assert requeridas.issubset(datos.columns)
    assert len(datos) == 530
    assert datos["Código"].is_unique
    assert datos[["Latitud", "Longitud", "Altitud"]].notna().all().all()
    assert datos["Latitud"].between(-19.0, -0.4).all()
    assert datos["Longitud"].between(-82.0, -68.0).all()


def test_cascada_timon_usa_coordenadas_oficiales():
    datos = pd.read_excel(GIS_PATH)
    cascada = datos.loc[datos["Código"] == 1754].iloc[0]

    assert cascada["Nombre"] == "Cascada Timón"
    assert abs(float(cascada["Latitud"]) - (-7.449139960880176)) < 1e-10
    assert abs(float(cascada["Longitud"]) - (-76.90192676225385)) < 1e-10
    assert float(cascada["Altitud"]) == 320.0
