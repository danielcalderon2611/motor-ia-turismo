# Evaluación final de MobileNetV2 con validación cruzada

Modelo desplegable: `models/modelo_mobilenetv2_limpio_final.keras`

Clases: bosque, cascada, desierto, lago, montana, nevado, playa y volcan.

El modelo se desarrolló mediante validación cruzada estratificada. Se compararon
K=3 y K=5 usando F1 macro promedio como criterio principal y accuracy promedio
como desempate. El test interno reservado no participó en el entrenamiento, la
selección de K ni la elección de épocas.

## Selección de K

| K | Accuracy media | Desviación | F1 macro medio | Desviación | AUC macro medio |
|---:|---:|---:|---:|---:|---:|
| 3 | **93.98%** | **0.36 pp** | **93.74%** | **0.42 pp** | **99.76%** |
| 5 | 93.71% | 0.62 pp | 93.47% | 0.63 pp | 99.75% |

Se seleccionó K=3 por su mayor F1 macro y mayor estabilidad entre folds. Las
épocas del modelo final se eligieron mediante la mediana de las mejores épocas
de K=3: 12 épocas de cabeza y 10 épocas de fine-tuning.

## Test interno reservado

Cantidad de imágenes: 849.

| Métrica | Resultado |
|---|---:|
| Accuracy | **93.88%** |
| Balanced accuracy | **93.52%** |
| Precision macro | **94.00%** |
| Recall macro | **93.52%** |
| F1 macro | **93.60%** |
| F1 ponderado | **93.87%** |
| Top-2 accuracy | **98.23%** |
| ROC AUC OVR macro | **99.77%** |
| Matthews correlation | **0.9303** |
| Cohen kappa | **0.9299** |
| Log-loss (menor es mejor) | **0.1795** |

## F1 por clase

| Clase | F1 |
|---|---:|
| bosque | 100.00% |
| cascada | 97.76% |
| desierto | 93.85% |
| lago | 90.43% |
| montana | 93.02% |
| nevado | 89.89% |
| playa | 91.67% |
| volcan | 92.17% |

## Comparación con el modelo anterior

Sobre las mismas 849 imágenes, el modelo anterior obtuvo 91.99% de accuracy y
91.68% de F1 macro. El modelo nuevo obtuvo 93.88% y 93.60%, respectivamente.
Los errores bajaron de 68 a 52, una reducción relativa de 23.53%.

## Artefactos reproducibles

- Entrenamiento: `entrenar_modelo_v2_kfold.py`.
- Informe de K=3: `REPORTE_VALIDACION_CRUZADA_K3.md`.
- Hashes del modelo, embeddings y datos GIS: `SHA256SUMS.txt`.
- Curva ROC final: `resultados_kfold_v2/curva_roc_test_interno.png`.
- Curva ROC OOF de K=3: `resultados_kfold_v2/curva_roc_oof_k3.png`.
- Matriz de confusión: `resultados_kfold_v2/matriz_confusion_test_interno.png`.
- F1 por clase: `resultados_kfold_v2/f1_por_clase_test_interno.png`.

## Prueba local

```powershell
& '.venv/Scripts/python.exe' 'probar_modelo.py' 'C:\ruta\imagen.jpg'
```

El comando devuelve la clase principal, confianza, top 3 y una marca de
revisión. Para cambiar el umbral:

```powershell
& '.venv/Scripts/python.exe' 'probar_modelo.py' 'C:\ruta\imagen.jpg' --umbral-revision 0.95
```
