import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


# Inicialización
app = FastAPI(title="API Análisis de Sentimientos IMDb Español")

# Cargar modelo y utilidades
model = load_model("modelo_comentarios_imdb2.h5")

with open("tokenizer_imdb2.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder2.pkl", "rb") as f:
    encoder = pickle.load(f)

max_len = 200  # igual que en el entrenamiento


# Función de limpieza
def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"@\w+", "", texto)
    texto = re.sub(r"[^a-záéíóúñü\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

# Modelo de datos de entrada
class Comentario(BaseModel):
    texto: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes poner "*" o el dominio de tu blog
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Endpoint principal
@app.post("/predict")
def predecir_sentimiento(data: Comentario):
    texto = limpiar_texto(data.texto)
    secuencia = tokenizer.texts_to_sequences([texto])
    secuencia_padded = pad_sequences(secuencia, maxlen=max_len, padding="post")

    pred = model.predict(secuencia_padded, verbose=0)[0][0]
    sentimiento = "Positivo" if pred >= 0.5 else "Negativo"

    return {
        "texto": data.texto,
        "texto_limpio": texto,
        "probabilidad": float(pred),
        "sentimiento": sentimiento
    }

# Ejecutar API local
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)