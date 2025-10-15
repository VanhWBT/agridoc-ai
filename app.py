import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

MODEL_PATH = "saurieng_mobilenetv2_model.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    model = None

CLASS_NAMES = ['ALLOCARIDARA_ATTACK', 'Leaf_Algal', 'Leaf_Blight', 'Leaf_Healthy', 'Leaf_Phomopsis']


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🌱 Phần mềm quản lý AI AgriDoc đang chạy!"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok" if model is not None else "model_missing"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        # Mở ảnh và đảm bảo đúng 3 kênh màu (RGB)
        img = Image.open(file.stream).convert("RGB").resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        disease = CLASS_NAMES[class_idx]

        return jsonify({
            'predicted_disease': disease,
            'confidence': confidence
        })

    except Exception as e:
        print(f">>> Exception in /predict: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
