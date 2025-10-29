# Imagen base ligera con Python
FROM python:3.10-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar git-lfs
RUN apt-get update && apt-get install -y git-lfs
RUN git lfs install

# Copiar archivos de requisitos
COPY requirements.txt .

# Copiar el resto del proyecto
COPY . .

# Descargar los archivos LFS (como tu modelo .h5)
RUN git lfs pull


# Instalar dependencias (sin caché para reducir tamaño)
RUN pip install --no-cache-dir -r requirements.txt



# Exponer el puerto que usa FastAPI
EXPOSE 8000

# Comando para ejecutar la API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
