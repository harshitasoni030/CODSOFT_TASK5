"""Train a character-level LSTM on the uploaded dataset.

Run from the project root:   python training/train_model.py
"""
import os
import pickle
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------- Settings ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "english.csv")
CORPUS_PATH = os.path.join(BASE_DIR, "data", "corpus.txt")  # optional, see README
MODEL_PATH = os.path.join(BASE_DIR, "model", "text_generator.keras")
VOCAB_PATH = os.path.join(BASE_DIR, "model", "vocabulary.pkl")

SEQ_LENGTH = 40      # how many previous characters the model sees
EMBED_DIM = 32
LSTM_UNITS = 128
EPOCHS = 20
BATCH_SIZE = 64

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ---------- 1-2. Load dataset and extract text ----------
if os.path.exists(CORPUS_PATH):
    print("Using plain-text corpus:", CORPUS_PATH)
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        text = f.read()
else:
    print("Using the 'label' column of:", CSV_PATH)
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    print("Columns:", list(df.columns), "| rows:", len(df))
    # Each label is one character (the letter/digit drawn in the image).
    # Labels are joined in file order, exactly as they appear in the CSV.
    text = "".join(df["label"].tolist())

# ---------- 3. Clean text ----------
text = text.replace("\r", "").strip()
print("Characters in training text:", len(text))

# ---------- 4. Vocabulary (index 0 is reserved for padding) ----------
chars = sorted(set(text))
char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
vocab_size = len(chars) + 1
print("Vocabulary size:", vocab_size - 1, "characters")

# ---------- 5-6. Sequences: previous characters -> next character ----------
encoded = [char_to_idx[c] for c in text]
inputs, targets = [], []
for i in range(len(encoded) - 1):
    inputs.append(encoded[max(0, i - SEQ_LENGTH + 1): i + 1])
    targets.append(encoded[i + 1])

X = pad_sequences(inputs, maxlen=SEQ_LENGTH, padding="pre")
y = np.array(targets)
print("Training samples:", X.shape[0], "| input shape:", X.shape)

# ---------- 7. Model ----------
model = models.Sequential([
    layers.Input(shape=(SEQ_LENGTH,)),
    layers.Embedding(vocab_size, EMBED_DIM, mask_zero=True),
    layers.LSTM(LSTM_UNITS),
    layers.Dense(vocab_size, activation="softmax"),
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ---------- 8. Train ----------
history = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)
print("Final training loss:", round(history.history["loss"][-1], 4))
print("Final training accuracy:", round(history.history["accuracy"][-1], 4))
print("Final validation accuracy:", round(history.history["val_accuracy"][-1], 4))

# ---------- 9-10. Save model and vocabulary ----------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
with open(VOCAB_PATH, "wb") as f:
    pickle.dump(
        {"char_to_idx": char_to_idx, "idx_to_char": idx_to_char, "seq_length": SEQ_LENGTH},
        f,
    )
print("Saved:", MODEL_PATH)
print("Saved:", VOCAB_PATH)
