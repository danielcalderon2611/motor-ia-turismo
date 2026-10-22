# Motor IA de Turismo

API multimodal para validar paisajes y atractivos turisticos. Combina:

- MobileNetV2 entrenado para ocho clases.
- Busqueda por similitud contra 6,812 embeddings.
- Gemma 4 visual mediante Ollama.
- Validacion de coordenadas y altitud contra la base GIS.

Clases activas: `bosque`, `cascada`, `desierto`, `lago`, `montana`, `nevado`,
`playa` y `volcan`.

## Entrega recomendada: Docker

Requisitos del servidor:

- Docker Engine con Docker Compose v2.
- Arquitectura `x86_64`.
- 16 GB de RAM como minimo; 32 GB recomendados.
- Al menos 40 GB de disco libre antes de construir y descargar los modelos.
- Acceso a Internet durante el primer arranque.

Clonar incluyendo los artefactos de Git LFS:

```bash
git lfs install
git clone https://github.com/danielcalderon2611/motor-ia-turismo.git
cd motor-ia-turismo
git lfs pull
cp .env.example .env
docker compose up -d --build
```

El primer arranque descarga `gemma4:e2b-it-qat` (aproximadamente 4.3 GB) y lo
guarda en el volumen persistente `ollama_data`. Los siguientes arranques lo
reutilizan.

Comprobar el despliegue:

```bash
docker compose ps
docker compose logs -f ollama-init
docker compose logs -f api
curl http://127.0.0.1:8000/health
```

Documentacion interactiva: `http://IP_DEL_SERVIDOR:8000/docs`.

Para detener sin borrar el modelo de Ollama:

```bash
docker compose down
```

No usar `docker compose down -v` salvo que se quiera eliminar tambien el modelo
descargado de Gemma.

## Prueba local sin Docker

Con Ollama iniciado y `gemma4:e2b-it-qat` instalado:

```powershell
& '.venv/Scripts/python.exe' -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Tambien puede probarse una imagen colocada en `imagenes_prueba`:

```powershell
& '.venv/Scripts/python.exe' 'probar_motor_completo.py' --lat -15.8434721 --lon -70.0274309 --alt 3867
```

## Archivos necesarios en produccion

- `api.py`
- `models/modelo_mobilenetv2_limpio_final.keras`
- `fewshot_db/embeddings_v2.pkl`
- `lugares_turisticos_app_final_v3.xlsx`
- `requirements-api.txt`
- `Dockerfile`
- `docker-compose.yml`

Los hashes esperados estan en `SHA256SUMS.txt`.

## Seguridad

El puerto de Ollama no se publica fuera de la red interna de Docker. La API no
incluye autenticacion: antes de exponerla a Internet debe colocarse detras de
HTTPS y un mecanismo de autenticacion, por ejemplo un Application Load Balancer,
API Gateway o un proxy inverso.

Consulta `DESPLIEGUE_AWS.md` para el traspaso a EC2 y `MOTOR_COMPLETO.md` para
los detalles de la logica del motor.
