# Import các "vũ khí" cần thiết
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping 
import matplotlib.pyplot as plt
import os

# --- CẤU HÌNH "CHIẾN TRƯỜNG" ---
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50 # <<< Đặt số vòng học tối đa, "giám thị" sẽ tự dừng sớm
DATA_DIR = 'dataset'

# === BƯỚC 1: CHUẨN BỊ "LƯƠNG THẢO" (DATA AUGMENTATION) ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# === BƯỚC 2: TRIỆU HỒI "TƯỚNG QUÂN" (TRANSFER LEARNING) ===
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

for layer in base_model.layers:
    layer.trainable = False
    # ... code của base_model và vòng lặp for ở trên giữ nguyên

# ---> THÊM ĐOẠN CODE MỚI VÀO ĐÂY <---
# Mình sẽ "mở khóa" các lớp ở trên cùng để tinh chỉnh
# Bắt đầu mở từ lớp thứ 100 trở đi chẳng hạn
fine_tune_at = 100 
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True
# ------------------------------------

# ... code tạo model (x = base_model.output, ...) giữ nguyên

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
# Tùy chỉnh lại thuật toán Adam với tốc độ học nhỏ hơn
optimizer = Adam(learning_rate=0.000001) 
# Biên dịch lại model với thuật toán đã được tùy chỉnh
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print("\n--- CẤU TRÚC CỦA TÂN BINH AI ---")
model.summary()

# === TẠO "GIÁM THỊ" EARLY STOPPING ===
# Theo dõi 'val_loss', kiên nhẫn 5 vòng, và lấy lại bộ não tốt nhất.
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# === BƯỚC 3: BẮT ĐẦU "HUẤN LUYỆN" ===
print("\n--- BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN (CÓ GIÁM SÁT) ---")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping] # <<< CHO "GIÁM THỊ" VÀO LỚP
)

# === LƯU LẠI "TÂN BINH" SAU KHI TỐT NGHIỆP ===
model.save('saurieng_mobilenetv2_model.keras')
print("\n--- ĐÃ HUẤN LUYỆN XONG VÀ LƯU LẠI MÔ HÌNH VÀO FILE 'saurieng_mobilenetv2_model.keras' ---")

# === VẼ BIỂU ĐỒ KẾT QUẢ "HỌC TẬP" ===
acc = history.history['accuracy']
val_acc = history.history['val_accuracy'] # <<< ĐÃ SỬA LỖI TYPO
loss = history.history['loss']
val_loss = history.history['val_loss']     # <<< ĐÃ SỬA LỖI TYPO

# Lấy số epoch thực tế đã chạy trước khi dừng
actual_epochs = len(history.history['loss'])
epochs_range = range(actual_epochs)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title(f'Accuracy (Stopped at Epoch {actual_epochs})')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title(f'Loss (Stopped at Epoch {actual_epochs})')
plt.show()