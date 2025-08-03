from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2, numpy as np, joblib, base64, io, os, logging
from PIL import Image

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Define path to model folder (../model relative to app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

try:
    model = joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    class_names = joblib.load(os.path.join(MODEL_DIR, "class_names.pkl"))
    logging.info("✅ Model, scaler, and class names loaded successfully.")
except FileNotFoundError as e:
    logging.error(f"❌ Model files not found: {e}")
    model = scaler = class_names = None

def extract_hog_features(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    hog = cv2.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9)
    return hog.compute(img).flatten()

def extract_color_histogram(img):
    hists = [cv2.calcHist([img],[ch],None,[32],[0,256]) for ch in range(3)]
    return np.concatenate(hists).flatten()

def extract_features(img):
    return np.concatenate([extract_hog_features(img), extract_color_histogram(img)])

def preprocess_image(b64):
    try:
        img_bytes = base64.b64decode(b64.split(",")[1])
        img_pil = Image.open(io.BytesIO(img_bytes))
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return cv2.resize(img_cv, (32, 32))
    except Exception as e:
        logging.error(f"❌ Image preprocessing failed: {e}")
        return None

@app.route('/')
def home():
    return "✅ Flask backend is running!"

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/classes')
def classes():
    if class_names is None:
        return jsonify({'error': 'Model not loaded'}), 500
    return jsonify({'classes': class_names})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json(silent=True)
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    image = preprocess_image(data['image'])
    if image is None:
        return jsonify({'error': 'Image processing failed'}), 400

    features = extract_features(image).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = int(model.predict(features_scaled)[0])
    probabilities = model.predict_proba(features_scaled)[0]
    predicted_class = class_names[prediction]

    top3 = sorted(
        ({class_names[i]: float(p) for i, p in enumerate(probabilities)}).items(),
        key=lambda x: x[1], reverse=True
    )[:3]

    return jsonify({
        'prediction': predicted_class,
        'confidence': float(probabilities[prediction]),
        'top_predictions': [{'class': cls, 'confidence': conf} for cls, conf in top3]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
