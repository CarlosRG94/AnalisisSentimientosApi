import pandas as pd
import numpy as np
import re
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout, Attention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gensim.downloader as api


# Cargar dataset
df = pd.read_csv("imdb_es.csv")[["review_es", "sentimiento"]]
df.rename(columns={"review_es": "comentario"}, inplace=True)

def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"http\S+", "", texto)       # eliminar URLs
    texto = re.sub(r"@\w+", "", texto)          # eliminar menciones
    texto = re.sub(r"[^a-záéíóúñü\s]", "", texto)  # quitar símbolos
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

df["comentario"] = df["comentario"].apply(limpiar_texto)

print("Valores únicos en sentimiento:", df["sentimiento"].unique())
print(df["sentimiento"].value_counts())

# --- Tokenización ---
max_words = 50000  # tamaño del vocabulario
max_len = 200      # longitud máxima por comentario

tokenizer = Tokenizer(num_words=max_words, oov_token="<UNK>")
tokenizer.fit_on_texts(df["comentario"])

X = tokenizer.texts_to_sequences(df["comentario"])
X = pad_sequences(X, maxlen=max_len, padding='post')
print("Tamaño del vocabulario:", len(tokenizer.word_index))
print("Porcentaje de ceros en X:", np.mean(X == 0))
total_palabras = len(tokenizer.word_index)
cobertura = (max_words / total_palabras) * 100
print(f"Palabras únicas totales: {total_palabras}")
print(f"Cobertura aproximada: {cobertura:.2f}% de las palabras más frecuentes")

# --- Etiquetas ---
encoder = LabelEncoder()
y = encoder.fit_transform(df["sentimiento"]) 

# Guardar tokenizer y label encoder (opcional)
with open("tokenizer_imdb2.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder2.pkl", "wb") as f:
    pickle.dump(encoder, f)

# --- Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(X.shape, y.shape)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# --- Embeddings preentrenados ---
print("🔹 Cargando embeddings preentrenados FastText (es)…")
fasttext_model = api.load("fasttext-wiki-news-subwords-300")  # embeddings en español
embedding_dim = fasttext_model.vector_size

# Matriz de embeddings
word_index = tokenizer.word_index
num_words = min(max_words, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embedding_dim))

for word, i in word_index.items():
    if i >= max_words:
        continue
    if word in fasttext_model:
        embedding_matrix[i] = fasttext_model[word]

# --- Modelo con Self-Attention ---
input_layer = Input(shape=(max_len,))

embedding_layer = Embedding(
    input_dim=num_words,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    trainable=True,
    embeddings_regularizer=tf.keras.regularizers.l2(1e-5)
)(input_layer)

# Primera capa BiLSTM (devuelve secuencia completa para aplicar atención)
x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(embedding_layer)

# Atención entre secuencias (Self-Attention)
attn_out = Attention()([x, x])  # query = key = value = x

# Combinación de atención + segunda capa BiLSTM más ligera
x = Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3))(attn_out)

# Capas densas finales
x = Dense(64, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(1, activation='sigmoid')(x)

# Construir modelo
model = Model(inputs=input_layer, outputs=output)

# Compilación
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=2)
]

# --- Entrenamiento ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=64,
    verbose=1,
    callbacks=callbacks
)
import matplotlib.pyplot as plt
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.show()

# --- Evaluación ---
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Accuracy en test: {acc:.4f}")

# Guardar
model.save("modelo_comentarios_imdb2.h5")
print("✅ Modelo guardado en modelo_comentarios.h5")

# Cargar
from tensorflow.keras.models import load_model
modelo_cargado = load_model("modelo_comentarios_imdb2.h5")

# --- Bucle interactivo ---
print("\n💬 Modo de prueba interactiva del modelo")
print("Escribe un comentario y el modelo predecirá su puntuación (1 a 5).")
print("Escribe 'salir' para terminar.\n")

while True:
    texto = input("👉 Escribe un comentario: ").strip()
    if texto.lower() in ["salir", "exit", "quit"]:
        break

    texto_limpio = limpiar_texto(texto)
    secuencia = tokenizer.texts_to_sequences([texto_limpio])
    secuencia_padded = pad_sequences(secuencia, maxlen=max_len, padding='post')

    pred = modelo_cargado.predict(secuencia_padded, verbose=0)[0][0]
    print(f"Confianza: {pred:.3f}")
    print("Predicción:", "Positivo" if pred >= 0.5 else "Negativo")