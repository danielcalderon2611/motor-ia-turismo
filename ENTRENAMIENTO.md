# Entrenamiento del modelo turístico

El archivo `entrenar_modelo.py` entrena una MobileNetV2 nueva con las ocho
clases del dataset limpio: bosque, cascada, desierto, lago, montaña, nevado,
playa y volcán. Utiliza únicamente `train` y `val`; mantiene `test` y el test
externo Places365 sin consultar hasta la evaluación final.

## Diseño

- Entrada RGB de 224 × 224 píxeles y `mobilenet_v2.preprocess_input`, compatible
  con la API actual.
- Pesos iniciales de ImageNet.
- Aumento moderado: espejo horizontal, rotación, zoom, traslación y contraste.
- Pesos de clase para compensar las diferencias de cantidad entre categorías.
- Fase 1: MobileNetV2 congelada y entrenamiento del clasificador.
- Fase 2: fine-tuning de las últimas 40 capas, conservando BatchNorm congelado.
- Early stopping, reducción automática del learning rate y checkpoints.
- Historiales CSV siempre disponibles y registros para TensorBoard cuando ese
  paquete opcional se encuentre instalado.
- Selección automática entre el mejor modelo de la fase 1 y el de la fase 2.

## Verificar antes de entrenar

```powershell
& '.venv/Scripts/python.exe' 'entrenar_modelo.py' --solo-verificar
```

## Entrenar

```powershell
& '.venv/Scripts/python.exe' 'entrenar_modelo.py'
```

La salida predeterminada es:

```text
D:\proyecto_turismo_ia\entrenamiento_mobilenetv2_v2
```

Allí se generan el modelo final, los dos mejores checkpoints, historiales CSV,
logs de TensorBoard, el orden de clases y los metadatos reproducibles.

## GPU en este equipo

El equipo tiene una NVIDIA GeForce RTX 3050 de 6 GB, pero TensorFlow 2.21 no
usa CUDA en Windows nativo. Para entrenar con GPU se recomienda WSL2 con una
instalación Linux compatible de TensorFlow. El mismo script funciona en ambos
entornos y detecta automáticamente la GPU.

No se debe copiar el modelo nuevo a `models/` ni regenerar embeddings hasta
haber completado la evaluación de los puntos 5 y 6.
