# Entrega para despliegue en AWS EC2

## Arquitectura incluida

`docker-compose.yml` levanta tres servicios:

1. `ollama`: servidor local de Gemma 4 con volumen persistente.
2. `ollama-init`: descarga o verifica `gemma4:e2b-it-qat` y termina.
3. `api`: carga CNN, embeddings y GIS, y atiende por el puerto 8000.

Ollama no publica su puerto 11434 al exterior. Solo la API puede exponerse.

## Instancia sugerida

Despliegue CPU inicial:

- Minimo funcional: `m7i.xlarge`, 4 vCPU y 16 GiB de RAM.
- Recomendado: `m7i.2xlarge`, 8 vCPU y 32 GiB de RAM.
- Disco EBS `gp3`: 50 GiB o mas.
- Sistema operativo: Ubuntu Server 24.04 LTS x86_64.

El modo incluido usa `GEMMA_NUM_GPU=0`. Una instancia GPU requiere NVIDIA
Container Toolkit y una configuracion Compose adicional; no es necesaria para
el primer despliegue funcional.

## Preparacion de la instancia

Instalar Git, Git LFS y Docker Engine con Compose v2 siguiendo la documentacion
oficial de la distribucion. Despues:

```bash
git lfs install
git clone https://github.com/danielcalderon2611/motor-ia-turismo.git
cd motor-ia-turismo
git lfs pull
cp .env.example .env
docker compose config
docker compose up -d --build
```

El primer arranque puede tardar varios minutos porque descarga la imagen de
Ollama, las dependencias de TensorFlow y el modelo Gemma de 4.3 GB.

## Verificacion

```bash
docker compose ps
docker compose logs --tail=100 ollama-init
docker compose logs --tail=100 api
curl http://127.0.0.1:8000/health
```

La respuesta de `/health` debe mostrar:

- `estado: OK`
- ocho salidas en `salidas_modelo`
- 6,812 embeddings
- Gemma disponible e instalado

## Reglas de red

- SSH/22: permitir solamente desde la IP administrativa.
- API/8000: durante pruebas, permitir solamente desde las IP autorizadas.
- No abrir 11434; es el puerto interno de Ollama.
- Para produccion, publicar por HTTPS y no exponer directamente Uvicorn.

## Persistencia y actualizaciones

El volumen `ollama_data` conserva Gemma entre reinicios. Los comandos habituales
son:

```bash
docker compose pull
docker compose up -d --build
docker compose logs -f api
docker compose down
```

No ejecutar `docker compose down -v` salvo que se quiera borrar el volumen y
forzar una descarga completa de Gemma.

## Entrega y control de integridad

Los binarios se versionan con Git LFS. Despues de clonar, ejecutar siempre
`git lfs pull`. Para validar los artefactos:

```bash
sha256sum -c SHA256SUMS.txt
```

No deben subirse al repositorio claves SSH, credenciales de AWS, archivos `.env`
reales ni tokens de acceso.
