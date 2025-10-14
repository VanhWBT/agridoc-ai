# Import "đồ nghề" cần thiết
from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# --- KHỞI TẠO "QUÁN ĂN" FLASK ---
app = Flask(__name__)

# --- THUÊ "ĐẦU BẾP" AI ---
# Load model AI xịn sò của mình
# Nhớ là file model phải nằm cùng cấp với file app.py này nhé
MODEL_PATH = 'saurieng_mobilenetv2_model.keras'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"--- Đã load 'đầu bếp' AI từ: {MODEL_PATH} ---")
except Exception as e:
    print(f"LỖI: Không thể load model. Chắc chắn rằng file '{MODEL_PATH}' tồn tại. Lỗi: {e}")
    model = None

# Lấy danh sách các "món ăn" (tên bệnh) mà "đầu bếp" có thể nấu
# Tạm thời mình hardcode ở đây, sau này có thể đọc tự động từ thư mục
CLASS_NAMES = ['ALLOCARIDARA_ATTACK', 'Leaf_Algal', 'Leaf_Blight', 'Leaf_Healthy', 'Leaf_Phomopsis']
print(f"--- 'Đầu bếp' có thể nấu các món: {CLASS_NAMES} ---")


# --- CÔNG THỨC SƠ CHẾ "NGUYÊN LIỆU" (Hàm Xử Lý Ảnh) ---
def prepare_image(image_bytes):
    # Đọc "nguyên liệu" ảnh từ dạng byte
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # Sơ chế về đúng kích thước 224x224 mà "đầu bếp" yêu cầu
    img = img.resize((224, 224))
    # Chuyển thành dạng "gia vị" số mà "đầu bếp" có thể nếm
    img_array = np.array(img)
    # Thêm một chiều để đúng khuôn dạng (batch size = 1)
    img_array = np.expand_dims(img_array, axis=0)
    # Chuẩn hóa "gia vị" (rescale về 0-1)
    img_array = img_array / 255.0
    return img_array

# --- MỞ "QUẦY ORDER" TẠI ĐỊA CHỈ /predict ---
@app.route('/predict', methods=['POST'])
def predict():
    # Kiểm tra xem "đầu bếp" có đang đi làm không
    if model is None:
        return jsonify({'error': 'Model is not loaded!'}), 500

    # Kiểm tra xem khách có gửi "nguyên liệu" (ảnh) không
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        # Đọc "nguyên liệu"
        img_bytes = file.read()
        # Đưa cho "phụ bếp" sơ chế
        prepared_image = prepare_image(img_bytes)
        
        # --- ĐƯA CHO "ĐẦU BẾP" TRỔ TÀI ---
        prediction = model.predict(prepared_image)
        
        # Lấy tên "món" mà "đầu bếp" tự tin nhất
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = np.max(prediction)
        
        # Trả "món ăn" cho khách
        return jsonify({
            'predicted_disease': predicted_class_name,
            'confidence': float(confidence)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- LỆNH "MỞ CỬA QUÁN" ---
if __name__ == '__main__':
    # Chạy quán ở địa chỉ IP của máy, cổng 5000
    # host='0.0.0.0' để các máy khác trong cùng mạng có thể truy cập
    app.run(host='0.0.0.0', port=5000, debug=True)