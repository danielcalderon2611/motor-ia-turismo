import unittest

from gemma_semantica import (
    ajustar_gemma_con_evidencia_visual,
    decidir_estado_semantico,
    normalizar_respuesta_gemma,
)


def respuesta(
    *,
    clase="montana",
    tipo_contenido="captura_directa",
    atractivo_visible=True,
    visibilidad="alta",
    personas_como_contexto=False,
    descripcion_breve="Escena analizada",
):
    return {
        "clase": clase,
        "tipo_contenido": tipo_contenido,
        "atractivo_visible": atractivo_visible,
        "visibilidad": visibilidad,
        "personas_como_contexto": personas_como_contexto,
        "descripcion_breve": descripcion_breve,
    }


def evaluar(datos):
    normalizada = normalizar_respuesta_gemma(datos)
    gemma = {"disponible": True, **normalizada}
    return normalizada, decidir_estado_semantico(gemma)


class PruebasSemanticaGemma(unittest.TestCase):
    def test_laptop_mostrando_paisaje_es_imagen_no_valida(self):
        normalizada, estado = evaluar(respuesta(tipo_contenido="pantalla"))
        self.assertEqual(normalizada["clase"], "no_turistico")
        self.assertEqual(estado[0], "IMAGEN_NO_VALIDA")

    def test_documento_o_fotografia_impresa_es_imagen_no_valida(self):
        for tipo in ("documento", "contenido_impreso", "otro_indirecto"):
            with self.subTest(tipo=tipo):
                normalizada, estado = evaluar(respuesta(tipo_contenido=tipo))
                self.assertEqual(normalizada["clase"], "no_turistico")
                self.assertEqual(estado[0], "IMAGEN_NO_VALIDA")

    def test_familia_frente_a_montana_visible_continua_flujo_turistico(self):
        normalizada, estado = evaluar(
            respuesta(personas_como_contexto=True, visibilidad="media")
        )
        self.assertEqual(normalizada["validacion"], "TURISTICO")
        self.assertIsNone(estado)

    def test_personas_ocultando_casi_todo_es_caso_dudoso(self):
        normalizada, estado = evaluar(
            respuesta(personas_como_contexto=True, visibilidad="baja")
        )
        self.assertEqual(normalizada["clase"], "dudoso")
        self.assertEqual(estado[0], "CASO_DUDOSO")

    def test_paisaje_directo_sin_personas_continua_clasificacion(self):
        normalizada, estado = evaluar(respuesta())
        self.assertEqual(normalizada["validacion"], "TURISTICO")
        self.assertIsNone(estado)

    def test_imagen_borrosa_es_caso_dudoso(self):
        _, estado = evaluar(respuesta(visibilidad="nula"))
        self.assertEqual(estado[0], "CASO_DUDOSO")

    def test_objeto_cotidiano_sin_atractivo_es_imagen_no_valida(self):
        normalizada, estado = evaluar(
            respuesta(clase="no_turistico", atractivo_visible=False, visibilidad="nula")
        )
        self.assertEqual(normalizada["clase"], "no_turistico")
        self.assertEqual(normalizada["visibilidad"], "nula")
        self.assertEqual(estado[0], "IMAGEN_NO_VALIDA")

    def test_gemma_no_disponible_es_caso_dudoso(self):
        estado = decidir_estado_semantico({"disponible": False})
        self.assertEqual(estado[0], "CASO_DUDOSO")

    def test_respuesta_incompleta_se_rechaza(self):
        with self.assertRaises(ValueError):
            normalizar_respuesta_gemma({"clase": "montana"})

    def test_visibilidad_media_y_baja_confianza_se_degrada_a_dudoso(self):
        normalizada = normalizar_respuesta_gemma(respuesta(visibilidad="media"))
        ajustada = ajustar_gemma_con_evidencia_visual(
            {"disponible": True, **normalizada},
            0.49,
        )
        self.assertEqual(ajustada["clase"], "dudoso")
        self.assertEqual(ajustada["validacion"], "DUDOSO")
        self.assertEqual(ajustada["visibilidad"], "baja")

    def test_oclusores_no_se_conservan_como_personas_de_contexto(self):
        normalizada = normalizar_respuesta_gemma(
            respuesta(personas_como_contexto=True, visibilidad="media")
        )
        ajustada = ajustar_gemma_con_evidencia_visual(
            {"disponible": True, **normalizada},
            0.52,
        )
        self.assertFalse(ajustada["personas_como_contexto"])

    def test_familia_visible_y_alta_confianza_no_se_degrada(self):
        normalizada = normalizar_respuesta_gemma(
            respuesta(personas_como_contexto=True, visibilidad="media")
        )
        ajustada = ajustar_gemma_con_evidencia_visual(
            {"disponible": True, **normalizada},
            0.99,
        )
        self.assertEqual(ajustada["clase"], "montana")
        self.assertEqual(ajustada["validacion"], "TURISTICO")


if __name__ == "__main__":
    unittest.main()
