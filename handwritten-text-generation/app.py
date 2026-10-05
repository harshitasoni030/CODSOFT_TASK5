"""Flask app that serves the trained character-level RNN."""
import os
import pickle

import numpy as np
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "text_generator.keras")
VOCAB_PATH = os.path.join(BASE_DIR, "model", "vocabulary.pkl")

app = Flask(__name__)

model = None
char_to_idx, idx_to_char, seq_length = {}, {}, 40

if os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH):
    model = load_model(MODEL_PATH)
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    char_to_idx = vocab["char_to_idx"]
    idx_to_char = vocab["idx_to_char"]
    seq_length = vocab["seq_length"]
else:
    print("Model not found. Run: python training/train_model.py")


def sample_next(probs, temperature):
    """Pick the next character index. Low temperature = safer, high = more random."""
    probs = np.asarray(probs, dtype="float64")
    probs = np.log(probs + 1e-9) / temperature
    probs = np.exp(probs - np.max(probs))
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def generate_text(seed_text, length, temperature):
    """Generate text one character at a time, feeding each new character back in."""
    # Keep only characters the model knows from training.
    seed = "".join(c for c in seed_text if c in char_to_idx)
    if not seed:
        seed = np.random.choice(list(char_to_idx.keys()))

    ids = [char_to_idx[c] for c in seed]
    generated = []

    for _ in range(length):
        window = ids[-seq_length:]
        window = [0] * (seq_length - len(window)) + window  # pad on the left
        probs = model.predict(np.array([window]), verbose=0)[0]
        next_id = sample_next(probs, temperature)
        if next_id == 0:  # padding index, never a real character
            next_id = int(np.argmax(probs[1:])) + 1
        ids.append(next_id)
        generated.append(idx_to_char[next_id])

    return seed + "".join(generated)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    if model is None:
        return jsonify({"error": "Model not trained yet. Run training/train_model.py first."}), 503

    data = request.get_json(silent=True) or {}
    seed_text = str(data.get("seed_text", ""))
    try:
        length = int(data.get("length", 100))
        temperature = float(data.get("temperature", 0.8))
    except (TypeError, ValueError):
        return jsonify({"error": "length must be an integer and temperature a number."}), 400

    if not 1 <= length <= 1000:
        return jsonify({"error": "length must be between 1 and 1000."}), 400
    if not 0.1 <= temperature <= 2.0:
        return jsonify({"error": "temperature must be between 0.1 and 2.0."}), 400

    text = generate_text(seed_text, length, temperature)
    return jsonify({"generated_text": text})


if __name__ == "__main__":
    app.run(debug=True)
