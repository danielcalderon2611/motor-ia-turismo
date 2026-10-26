# Informe de validación cruzada K=3 — MobileNetV2

Fecha de verificación: 25 de octubre de 2026.

## Integridad de la evaluación

- Método: Stratified K-Fold con K=3.
- Clases: bosque, cascada, desierto, lago, montana, nevado, playa y volcan.
- Imágenes evaluadas fuera de muestra (OOF): 7,664.
- Suma de los conjuntos de validación: 2,555 + 2,555 + 2,554 = 7,664.
- Predicciones OOF encontradas: 7,664.
- Los tres folds tienen modelo, checkpoint, predicciones y `resultado_fold.json`.

Cada imagen recibió una predicción cuando perteneció al fold de validación. Por
ello, las métricas OOF resumen el comportamiento del procedimiento de validación
cruzada completo y no las predicciones sobre el conjunto usado para entrenar cada
modelo.

## Resultado por fold

| Fold | Imágenes | Accuracy | Precision macro | Recall macro | F1 macro | AUC ROC macro |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2,555 | 94.25% | 94.18% | 94.12% | 94.07% | 99.77% |
| 2 | 2,555 | 94.13% | 94.08% | 93.92% | 93.89% | 99.80% |
| 3 | 2,554 | 93.58% | 93.42% | 93.25% | 93.27% | 99.71% |

## Media y estabilidad entre folds

| Métrica | Media | Desviación estándar | Mínimo | Máximo |
|---|---:|---:|---:|---:|
| Accuracy | 93.98% | 0.36 pp | 93.58% | 94.25% |
| Balanced accuracy | 93.76% | 0.46 pp | 93.25% | 94.12% |
| Precision macro | 93.89% | 0.41 pp | 93.42% | 94.18% |
| Recall macro | 93.76% | 0.46 pp | 93.25% | 94.12% |
| F1 macro | 93.74% | 0.42 pp | 93.27% | 94.07% |
| F1 ponderado | 93.97% | 0.36 pp | 93.56% | 94.23% |
| AUC ROC macro OvR | 99.76% | 0.05 pp | 99.71% | 99.80% |

La dispersión es baja: el accuracy solo varió 0.67 puntos porcentuales entre el
mejor y el peor fold, y el F1 macro varió 0.81 puntos. Esto indica que el resultado
no depende fuertemente de una única partición K=3.

## Métricas OOF agregadas

| Métrica | Resultado |
|---|---:|
| Accuracy | 93.98% |
| Balanced accuracy | 93.76% |
| Precision macro | 93.87% |
| Recall macro | 93.76% |
| F1 macro | 93.75% |
| Precision ponderada | 94.09% |
| F1 ponderado | 93.97% |
| Top-2 accuracy | 98.67% |
| AUC ROC macro OvR | 99.75% |
| MCC | 0.9313 |
| Cohen's kappa | 0.9311 |
| Log loss | 0.1718 |
| Confianza media | 94.67% |

Se obtuvieron 7,203 aciertos y 461 errores. La cercanía entre accuracy, balanced
accuracy y F1 macro sugiere que el rendimiento global no está ocultando una caída
extrema en alguna clase. MCC y kappa por encima de 0.93 también muestran una
concordancia alta más allá del azar.

## Rendimiento por clase sobre las predicciones OOF

| Clase | Precision | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| bosque | 99.48% | 98.53% | 99.00% | 1,155 |
| cascada | 96.44% | 98.39% | 97.41% | 992 |
| desierto | 94.87% | 88.08% | 91.35% | 839 |
| lago | 91.99% | 95.49% | 93.71% | 842 |
| montana | 96.32% | 87.68% | 91.80% | 1,015 |
| nevado | 91.76% | 91.00% | 91.38% | 856 |
| playa | 91.92% | 96.29% | 94.05% | 1,051 |
| volcan | 88.18% | 94.64% | 91.29% | 914 |

Las clases más sólidas son bosque y cascada. Los recalls más bajos aparecen en
montana (87.68%) y desierto (88.08%). La precisión más baja corresponde a volcan
(88.18%), señal de que imágenes de otras clases llegan a clasificarse como volcán.

## Confusiones principales

1. montana → volcan: 65 imágenes.
2. desierto → playa: 39 imágenes.
3. desierto → nevado: 28 imágenes.
4. desierto → volcan: 25 imágenes.
5. nevado → playa: 24 imágenes.
6. montana → nevado: 20 imágenes.
7. playa → lago: 19 imágenes.
8. volcan → montana: 18 imágenes.

Estas confusiones son visualmente plausibles por la coexistencia de roca, nieve,
arena, agua y relieve. Si posteriormente se mejora el dataset, conviene priorizar
ejemplos difíciles que separen montana/volcan/nevado y desierto/playa.

## Curvas ROC

El AUC por clase está entre 99.61% y 100.00% aproximadamente. Los valores altos
indican una buena capacidad de ordenamiento One-vs-Rest, pero no sustituyen las
métricas basadas en la clase elegida. Para la discusión del modelo deben mostrarse
juntos el AUC, F1 por clase y la matriz de confusión.

## Conclusión final

K=3 ofrece un resultado alto y estable: accuracy OOF de 93.98%, F1 macro de 93.75%
y AUC macro de 99.75%. La comparación posterior confirmó que K=3 superó a K=5
en F1 macro y accuracy promedio, además de presentar menor dispersión entre folds.
Por esos motivos se seleccionó K=3 para determinar las épocas del modelo final.
