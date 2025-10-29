FROM python:3.10-slim

# Instalar git y git-lfs
RUN apt-get update && apt-get install -y git git-lfs
RUN git lfs install

# Crear carpeta de trabajo
WORKDIR /app

# Clonar el repositorio con LFS habilitado
RUN git clone https://github.com/CarlosRG94/AnalisisSentimientosApi.git .

# Descargar los archivos LFS (por ejemplo, el modelo .h5)
RUN git lfs pull

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto (Railway usa variables de entorno para el puerto real)
EXPOSE 8000

# Comando para iniciar FastAPI/Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
