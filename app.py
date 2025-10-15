from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app)  # Cho phép gọi từ Streamlit

# --- Load model ---
model = tf.keras.models.load_model("saurieng_mobilenetv2_model.keras")
class_names = ['ALLOCARIDARA_ATTACK', 'Leaf_Algal', 'Leaf_Blight', 'Leaf_Healthy', 'Leaf_Phomopsis']

# --- Route kiểm tra server ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🌱 AgriDoc AI backend is running!"})

# --- Route dự đoán ---
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = Image.open(file.stream).resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    disease = class_names[class_idx]

    return jsonify({
        'predicted_disease': disease,
        'confidence': confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
