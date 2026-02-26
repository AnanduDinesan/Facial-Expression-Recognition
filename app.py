import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────
app.config['UPLOAD_FOLDER']       = 'static/uploads'
app.config['MAX_CONTENT_LENGTH']  = 5 * 1024 * 1024  # 5MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

IMG_SIZE = 48

# ⚠️ Must match your train_val_ds.class_names order exactly
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

EMOTION_EMOJI = {
    'angry':    '😠',
    'disgust':  '🤢',
    'fear':     '😨',
    'happy':    '😊',
    'neutral':  '😐',
    'sad':      '😢',
    'surprise': '😲'
}

EMOTION_COLOR = {
    'angry':    '#ef4444',
    'disgust':  '#84cc16',
    'fear':     '#8b5cf6',
    'happy':    '#f59e0b',
    'neutral':  '#94a3b8',
    'sad':      '#3b82f6',
    'surprise': '#ec4899'
}

# ── Load Model ────────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model('fer_efficientnet_v2_final.keras')
print("Model loaded successfully!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """
    Load image and prepare it for EfficientNetB0.
    - Convert to RGB (handles grayscale & RGBA inputs)
    - Resize to 48x48
    - Add batch dimension
    - Do NOT normalize — EfficientNet's preprocess_input inside model handles it
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 48, 48, 3)
    return img_array

def predict_emotion(image_path):
    """Run model inference and return predictions sorted by confidence"""
    img_array   = preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)[0]  # Shape: (7,)

    results = []
    for emotion, prob in zip(EMOTIONS, predictions):
        results.append({
            'emotion': emotion.capitalize(),
            'emoji':   EMOTION_EMOJI[emotion],
            'color':   EMOTION_COLOR[emotion],
            'prob':    float(prob),
            'pct':     round(float(prob) * 100, 1)
        })

    # Sort highest confidence first
    results.sort(key=lambda x: x['prob'], reverse=True)
    return results

# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, JPEG, GIF or WEBP'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        results = predict_emotion(filepath)
        return jsonify({
            'success':     True,
            'image_url':   f'/static/uploads/{filename}',
            'predictions': results,
            'top':         results[0]       # Highest confidence emotion
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Run ───────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000)