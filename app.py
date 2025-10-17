import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"      # Ẩn log TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # Tắt GPU (Render không có GPU)

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np

# --- Cấu hình Flask ---
app = Flask(__name__)
CORS(app)

# --- Load model (chỉ load 1 lần khi khởi động) ---
MODEL_PATH = "saurieng_mobilenetv2_model.keras"
CLASS_NAMES = ['ALLOCARIDARA_ATTACK', 'Leaf_Algal', 'Leaf_Blight', 'Leaf_Healthy', 'Leaf_Phomopsis']

model = None
try:
    # Load model an toàn hơn
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ ERROR loading model: {e}")

# --- Kiểm tra server ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🌱 AgriDoc AI backend đang chạy! Dùng POST /predict để chẩn đoán."})

# --- Kiểm tra model có sẵn ---
@app.route("/health", methods=["GET"])
def health():
    status = "ok" if model else "model_missing"
    return jsonify({"status": status})

# --- API dự đoán ---
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        file = request.files["file"]
        img = Image.open(file.stream).convert("RGB").resize((224, 224))

        # Tiền xử lý ảnh
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

        # Dự đoán (gói trong tf.function để nhanh hơn)
        predictions = model.predict(img_array, verbose=0)
        class_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        disease = CLASS_NAMES[class_idx]

        return jsonify({
            "predicted_disease": disease,
            "confidence": round(confidence, 4)
        })
    except Exception as e:
        print(f"🔥 Error in predict: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
