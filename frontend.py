import streamlit as st
import requests
from PIL import Image
import io

# --- THIẾT KẾ "MẶT TIỀN" ---

# Đặt tên cho "quán ăn"
st.set_page_config(page_title="Bác Sĩ Cây Trồng", page_icon="🌿")
st.title(" 👨‍⚕️Mộc Sĩ Thông Thái")
st.write("""
Chào mừng đến với Bác sĩ Cây! Hãy tải lên một bức ảnh lá sầu riêng, 
người hiểu biết nhất về lĩnh vự này sẽ chẩn đoán bệnh giúp bạn.
""")

# Tạo một "bàn order" cho phép khách hàng upload ảnh
uploaded_file = st.file_uploader("Chọn một ảnh lá cây...", type=["jpg", "jpeg", "png"])

# Địa chỉ của "nhà bếp" API
API_URL = "https://agridoc-ai.onrender.com/predict"


# --- QUY TRÌNH "PHỤC VỤ" ---

if uploaded_file is not None:
    # Hiển thị "nguyên liệu" mà khách vừa đưa
    st.image(uploaded_file, caption="Ảnh bạn đã tải lên.", use_column_width=True)
    st.write("")
    st.write("Đang chẩn đoán...")

    # Gửi "nguyên liệu" đến "nhà bếp"
    try:
        # Đọc dữ liệu ảnh
        image_data = uploaded_file.getvalue()
        
        # Đóng gói để gửi đi
        files = {"file": (uploaded_file.name, image_data, uploaded_file.type)}
        
        # Gửi yêu cầu đến API
        response = requests.post(API_URL, files=files)
        
        if response.status_code == 200:
            # Nhận "món ăn" (kết quả) từ "nhà bếp"
            result = response.json()
            
            disease = result['predicted_disease']
            confidence = result['confidence']
            
            # Hiển thị kết quả cho khách
            st.success(f"**Chẩn đoán:** {disease}")
            st.info(f"**Độ tin cậy:** {confidence*100:.2f}%")
            
            if "Healthy" in disease:
                st.balloons()
                st.markdown(" Chúc mừng! Cây của bạn trông rất khỏe mạnh!")
            else:
                st.warning("Cây của bạn có dấu hiệu bị bệnh. Hãy xem xét các biện pháp chăm sóc phù hợp.")
        else:
            st.error(f"Lỗi từ 'nhà bếp': {response.json().get('error', 'Lỗi không xác định')}")

    except requests.exceptions.ConnectionError:
        st.error("Lỗi kết nối: Không thể kết nối đến 'nhà bếp' (backend API).")
        st.info("MẸO: Bạn đã chạy file `app.py` ở một terminal khác chưa?")
    except Exception as e:
        st.error(f"Đã có lỗi xảy ra: {e}")