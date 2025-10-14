import requests
import os

# --- THÔNG TIN "ĐƠN HÀNG" ---

# Địa chỉ "quán ăn" của mình
API_URL = 'http://127.0.0.1:5000/predict'

# --- PHẦN NÀY BẠN CẦN THAY ĐỔI ---
# Đường dẫn đến "nguyên liệu" (ảnh) bạn muốn thử
# Ví dụ: 'dataset/Leaf_Blight/IMG_20211129_162330.jpg'
# Hãy chắc chắn rằng file ảnh này tồn tại nhé!
IMAGE_PATH = 'D:/NW/aa1.jpg'
# -----------------------------------


# --- TIẾN HÀNH "GỬI HÀNG" ---

try:
    # Mở file ảnh ở chế độ đọc byte (binary)
    with open(IMAGE_PATH, 'rb') as image_file:
        # Đóng gói "nguyên liệu" để gửi đi
        files = {'file': (os.path.basename(IMAGE_PATH), image_file, 'image/jpeg')}
        
        print(f">>> Đang gửi ảnh '{os.path.basename(IMAGE_PATH)}' đến 'quán ăn'...")
        
        # Gửi yêu cầu đến "quán" và chờ "món ăn" trả về
        response = requests.post(API_URL, files=files)
        
        # In kết quả
        print("\n--- HÓA ĐƠN TỪ 'QUÁN ĂN' ---")
        print(f"Status Code: {response.status_code}") # 200 là OK
        print("Nội dung món ăn:")
        print(response.json())
        print("--------------------------")

except FileNotFoundError:
    print(f"LỖI: Không tìm thấy 'nguyên liệu' tại đường dẫn: {'D:/NW/AgriDoc-AI/dataset/Leaf_Blight/Leaf_Blight_1.jpg'}")
    print("MẸO: Hãy kiểm tra lại tên thư mục và tên file ảnh bạn đã điền.")
except requests.exceptions.ConnectionError:
    print(f"LỖI: Không kết nối được tới 'quán ăn' tại {API_URL}")
    print("MẸO: 'Quán ăn' (file app.py) của bạn đã chạy chưa?")