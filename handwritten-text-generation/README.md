# Handwritten Text Generation (Character-Level RNN)

A character-level LSTM trained on the uploaded handwritten-characters dataset, served through a Flask web app.

## Project overview
The model reads a sequence of characters and predicts the next one. Repeating that step generates new text. A Flask API and a small web page let you choose a seed, a length and a temperature.

## Dataset
`data/english.csv` has 3,410 rows and two columns:

| Column | Meaning |
|--------|---------|
| `image` | Path to a handwritten image (e.g. `Img/img001-001.png`) |
| `label` | The single character in that image (`0-9`, `A-Z`, `a-z`) |

There are 62 classes with 55 samples each, stored in class order (`0` x55, `1` x55, ... `z` x55). The image files were not uploaded, and the CSV contains no sentences or words.

**What this means for the project:** the only text available is the `label` column, so `train_model.py` joins the labels in file order and trains on that string. The model therefore learns the structure of the label file (long runs of the same character, and mostly alphabetical order between runs), not English. Despite the name, the project generates character sequences, not drawn handwriting images.

To train on real sentences, put any plain-text file at `data/corpus.txt`. The training script uses it automatically instead of the CSV.

## Character-level RNN explained
1. Every unique character gets an integer ID (the vocabulary).
2. The text is cut into input sequences of 40 characters, each paired with the character that follows it.
3. An embedding layer turns IDs into vectors, an LSTM reads the sequence, and a softmax layer outputs a probability for every possible next character.
4. At generation time, one character is sampled, appended to the input, and the process repeats.
5. **Temperature** controls sampling: low values pick likely characters, high values add randomness.

## Model architecture
```
Input (40 character IDs)
-> Embedding (vocab size, 32, mask_zero=True)
-> LSTM (128 units)
-> Dense (vocab size, softmax)
```
Loss: sparse categorical cross-entropy. Optimizer: Adam. Index 0 is reserved for padding.

## Training instructions
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python training/train_model.py
```
This saves `model/text_generator.keras` and `model/vocabulary.pkl` (character-to-index and index-to-character maps). Settings such as `EPOCHS` and `SEQ_LENGTH` are at the top of the script.

Note: the script holds out the last 10% of the data for validation. Because the CSV is in class order, that held-out part contains characters the model never saw, so validation accuracy is very low. Treat that number as a data-ordering effect, not a bug.

## How to run Flask
```bash
python app.py
```
Open http://127.0.0.1:5000

## How to generate text
**Web page:** enter seed text, length (1-1000) and temperature (0.1-2.0), then click *Generate text*.

**API:**
```bash
curl -X POST http://127.0.0.1:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"seed_text": "The", "length": 100, "temperature": 0.8}'
```
Response: `{"generated_text": "..."}`. Seed characters not in the vocabulary are ignored.

## Future improvements
- Train on a real text corpus (`data/corpus.txt`) for meaningful words and sentences.
- Shuffle the data and split train/validation by class so validation is meaningful.
- Stack more LSTM/GRU layers and train longer.
- Use the character images with an image-generation model to produce actual handwriting.
- Add top-k or top-p sampling.
