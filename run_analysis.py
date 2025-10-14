# Import các "vũ khí" cần thiết
import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# --- PHẦN NÀY ĐÃ CHỈNH SỬA CHO BẠN ---
# Đường dẫn đến thư mục cha chứa tất cả các thư mục lá bệnh
data_dir = 'dataset'
# ------------------------------------

try:
    # Lấy danh sách các "binh đoàn" (tên các loại bệnh)
    classes = os.listdir(data_dir)
    # Lọc ra những file ẩn hệ thống nếu có (ví dụ .DS_Store trên macOS)
    classes = [cls for cls in classes if os.path.isdir(os.path.join(data_dir, cls))]
    print(f"Phát hiện {len(classes)} binh đoàn địch: {classes}\n")

    # === NHIỆM VỤ "ĐẾM QUÂN SỐ" ===
    print("--- Báo cáo Sĩ số các Binh đoàn ---")
    for cls in classes:
        # Đếm số lượng "lính" (ảnh) trong mỗi binh đoàn
        num_files = len(os.listdir(os.path.join(data_dir, cls)))
        print(f"-> Binh đoàn '{cls}': {num_files} lính")

    # === NHIỆM VỤ "XEM MẶT MŨI" ===
    print("\n--- Trinh sát Hình ảnh Địch ---")
    plt.figure(figsize=(15, 10))

    # Lấy 1 thằng lính ngẫu nhiên trong mỗi binh đoàn ra để "nhận diện"
    # Giả định có 6 lớp bệnh nên tạo layout 2 hàng 3 cột
    # Nếu bạn có số lượng lớp khác, có thể cần chỉnh lại subplot (ví dụ 3, 3 cho 9 lớp)
    num_classes = len(classes)
    for i, cls in enumerate(classes):
        # Tạo ô để vẽ, tối đa 6 ảnh
        if i < 6:
            plt.subplot(2, 3, i + 1)
            
            # Lấy danh sách tất cả các lính
            all_files = os.listdir(os.path.join(data_dir, cls))
            # Chọn ngẫu nhiên 1 thằng
            random_img_name = random.choice(all_files)
            random_img_path = os.path.join(data_dir, cls, random_img_name)
            
            # Đọc và hiển thị ảnh
            img = mpimg.imread(random_img_path)
            plt.imshow(img)
            plt.title(f"{cls}\n({img.shape[0]}x{img.shape[1]})") # Hiển thị tên và kích thước ảnh
            plt.axis('off')

    plt.tight_layout() # Tự động căn chỉnh cho đẹp
    plt.show() # Lệnh này sẽ mở một cửa sổ mới để hiển thị ảnh

except FileNotFoundError:
    print(f"LỖI: Không tìm thấy thư mục '{data_dir}'.")
    print("MẸO: Bạn đã tạo thư mục 'dataset' và bỏ các thư mục lá bệnh vào trong đó chưa?")

except Exception as e:
    print(f"Đã xảy ra một lỗi không xác định: {e}")