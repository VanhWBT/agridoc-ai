import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Giảm log rác từ TensorFlow

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# --- Khởi tạo Flask app ---
app = Flask(__name__)
CORS(app)

# --- Load model ---
MODEL_PATH = "saurieng_mobilenetv2_model.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ ERROR: Cannot load model: {e}")
    model = None

# Danh sách nhãn
CLASS_NAMES = ['ALLOCARIDARA_ATTACK', 'Leaf_Algal', 'Leaf_Blight', 'Leaf_Healthy', 'Leaf_Phomopsis']


# --- Route kiểm tra server ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🌿 AgriDoc AI backend is running! Use POST /predict to diagnose."})


# --- Route dự đoán ---
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({'error': 'Model failed to load on server!'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        file = request.files['file']
        img = Image.open(file.stream).convert("RGB").resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        predictions = model.predict(img_array)
        class_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        disease = CLASS_NAMES[class_idx]

        return jsonify({
            'predicted_disease': disease,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- Chạy server ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
