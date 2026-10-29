# Motor IA de turismo 2.4

El motor combina cuatro fuentes de evidencia:

1. MobileNetV2 de ocho clases.
2. Similitud contra 6,812 embeddings de 256 dimensiones.
3. Gemma 4 mediante el servicio local de Ollama.
4. Coordenadas, altitud y 530 registros de la base GIS.

## Recursos activos

- Modelo: `models/modelo_mobilenetv2_limpio_final.keras`
- Embeddings: `fewshot_db/embeddings_v2.pkl`
- GIS: `lugares_turisticos_app_final_v4.xlsx` (coordenadas auditadas contra SIGMINCETUR)
- API: `api.py`

Las clases son: bosque, cascada, desierto, lago, montana, nevado, playa y
volcan. Iglesia y ruinas no participan en el motor.

## Prueba directa

Colocar una sola imagen en `imagenes_prueba` y ejecutar, por ejemplo para el
Lago Titicaca:

```powershell
& '.venv/Scripts/python.exe' 'probar_motor_completo.py' --lat -15.8434721 --lon -70.0274309 --alt 3867
```

Para comprobar únicamente CNN, embeddings y GIS:

```powershell
& '.venv/Scripts/python.exe' 'probar_motor_completo.py' --lat -15.8434721 --lon -70.0274309 --alt 3867 --sin-gemma
```

## API local

```powershell
& '.venv/Scripts/python.exe' -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Abrir `http://127.0.0.1:8000/docs` para probar `/predict` desde Swagger.
El endpoint `/health` informa si CNN, embeddings, GIS, Ollama y el modelo Gemma
están disponibles.

Variables opcionales:

- `GEMMA_MODEL`: etiqueta de Ollama; por defecto `gemma4:e2b-it-qat`.
- `GEMMA_NUM_GPU`: capas de Gemma enviadas a la GPU. Por defecto `0` para
  evitar falta de VRAM en la RTX 3050 de 6 GB. En una GPU con memoria suficiente
  se puede ajustar durante las pruebas de despliegue.
- `GEMMA_NUM_CTX`: ventana de contexto local; por defecto `2048` para ahorrar
  memoria (es suficiente para la clasificación de una imagen).
- `GEMMA_NUM_BATCH`: lote interno de Ollama; por defecto `64` para reducir el
  uso máximo de memoria durante la inferencia.
- `GEMMA_TIMEOUT_SECONDS`: tiempo máximo por consulta; por defecto `180`.
- `OLLAMA_HOST`: servidor Ollama; por defecto `http://127.0.0.1:11434`.
- `GEMMA_ENABLED=false`: desactiva temporalmente Gemma.
- `TURISMO_MODEL_PATH`, `TURISMO_EMBEDDINGS_PATH`, `TURISMO_GIS_PATH`: rutas
  personalizadas para despliegue.

## Estado conocido de Gemma 4

El 21 de octubre de 2026 se verificó que las etiquetas `gemma4:e2b` y
`gemma4:e4b` interpretan incorrectamente imágenes en esta instalación de
Ollama para Windows. La variante `gemma4:e4b-it-qat` sí interpretó la imagen
de prueba del Lago Titicaca y devolvió `TURISTICO` y `lago` con el esquema JSON
del motor.

La variante QAT no cabe completamente en la VRAM disponible de la RTX 3050 de
6 GB mientras Windows usa la pantalla. Ejecutada por sí sola con
`GEMMA_NUM_GPU=0`, funciona correctamente en CPU/RAM. Sin embargo, los 12 GB de
RAM del equipo no alcanzan para mantener simultáneamente TensorFlow, la CNN y
`gemma4:e4b-it-qat`: el proyector visual de Ollama falla al reservar memoria.
Tampoco resolvieron el límite la descarga parcial a GPU, la precarga inversa ni
la reducción de contexto y lote.

La variante `gemma4:e2b-it-qat` conserva entrada visual, ocupa 4.3 GB y sí pudo
ejecutarse simultáneamente con TensorFlow, la CNN y los embeddings. Gemma elige
la clase visual y el motor deriva la validación para impedir combinaciones
contradictorias como `NO_TURISTICO` junto con `lago`. El motor mantiene la regla
conservadora: cualquier desacuerdo o indisponibilidad de Gemma produce
`CASO_DUDOSO`.
