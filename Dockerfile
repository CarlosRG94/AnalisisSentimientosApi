FROM python:3.10-slim

RUN apt-get update && apt-get install -y git git-lfs
RUN git lfs install

WORKDIR /app

# Clona tu repo con LFS habilitado
RUN git clone https://github.com/CarlosRG94/AnalisisSentimientosApi
# Instalar dependencias (sin caché para reducir tamaño)
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto que usa FastAPI
EXPOSE 8000

# Comando para ejecutar la API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
