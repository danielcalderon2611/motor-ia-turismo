# Evaluacion final del modelo de turismo

Modelo evaluado: `modelo_mobilenetv2_limpio_final.keras`
Clases: bosque, cascada, desierto, lago, montana, nevado, playa y volcan.

Los conjuntos de prueba no participaron en el entrenamiento ni en la seleccion
del mejor checkpoint.

## Resumen

| Metrica | Test interno (849) | Test externo Places365 (797) |
|---|---:|---:|
| Accuracy | 91.99% | 63.99% |
| Balanced accuracy | 91.64% | 64.09% |
| Precision macro | 92.10% | 67.13% |
| Recall macro | 91.64% | 64.09% |
| F1 macro | 91.68% | 61.11% |
| F1 ponderado | 91.99% | 61.06% |
| Top-2 accuracy | 98.00% | 79.55% |
| ROC AUC OVR macro | 99.65% | 93.18% |
| Matthews correlation | 90.88% | 59.87% |
| Cohen kappa | 90.83% | 58.85% |
| Log-loss (menor es mejor) | 0.223 | 1.816 |
| ECE, 10 bins (menor es mejor) | 2.42% | 23.79% |

## F1 por clase

| Clase | Test interno | Test externo |
|---|---:|---:|
| bosque | 99.22% | 42.19% |
| cascada | 96.40% | 87.96% |
| desierto | 91.53% | 73.37% |
| lago | 89.12% | 62.59% |
| montana | 90.38% | 30.07% |
| nevado | 85.87% | 45.86% |
| playa | 89.52% | 78.87% |
| volcan | 91.43% | 68.00% |

## Interpretacion

El modelo rinde muy bien con imagenes parecidas al dominio del dataset limpio,
pero no generaliza con la misma calidad a una fuente visual diferente. En el
test externo confunde especialmente bosque con lago, montana con volcan o
desierto, y nevado con montana o volcan.

La confianza softmax tampoco debe tratarse como una garantia. En el test
externo la confianza media fue 87.78%, aunque la accuracy real fue 63.99%, y el
ECE fue 23.79%. Por eso el sistema debe combinar CNN, GIS y Gemma, conservar
una salida de revision/dudoso y registrar predicciones para monitoreo.

Como politica conservadora previa a produccion, una prediccion con menos de 90%
puede enviarse a revision. Incluso sobre ese umbral, el test externo alcanzo
78.26% de accuracy y 66.37% de cobertura; el umbral reduce riesgo, pero no lo
elimina.

## Prueba local

```powershell
& '.venv/Scripts/python.exe' 'probar_modelo.py' 'C:\ruta\imagen.jpg'
```

El comando devuelve la clase principal, confianza, top 3 y una marca de
revision. Para cambiar el umbral:

```powershell
& '.venv/Scripts/python.exe' 'probar_modelo.py' 'C:\ruta\imagen.jpg' --umbral-revision 0.95
```

Los resultados completos y graficos estan en:

`D:\proyecto_turismo_ia\entrenamiento_mobilenetv2_v2\evaluacion_final`
