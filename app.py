import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import joblib

# Load mô hình và scaler đã huấn luyện
model = joblib.load('heart_model.pkl')  # Thay bằng file thật
scaler = joblib.load('heart_scaler.pkl')

st.title("Dự đoán Bệnh Tim")
st.write("Nhập các chỉ số y tế bên dưới để dự đoán nguy cơ mắc bệnh tim.")

# Tạo form nhập liệu
age = st.number_input("Tuổi", min_value=1, value=50)
sex = st.selectbox("Giới tính", ["Nam", "Nữ"])
cp = st.selectbox("Loại đau ngực (cp)", [0,1,2,3])
trestbps = st.number_input("Huyết áp nghỉ (trestbps)", min_value=0, value=120)
chol = st.number_input("Cholesterol", min_value=0, value=200)
fbs = st.selectbox("Đường huyết > 120mg/dl (fbs)", [0,1])
restecg = st.selectbox("Điện tim đồ (restecg)", [0,1,2])
thalach = st.number_input("Nhịp tim tối đa (thalach)", min_value=0, value=150)
exang = st.selectbox("Đau ngực do gắng sức (exang)", [0,1])
oldpeak = st.number_input("ST chênh (oldpeak)", min_value=0.0, value=1.0)
slope = st.selectbox("Độ dốc ST (slope)", [0,1,2])
ca = st.selectbox("Số mạch vành (ca)", [0,1,2,3])
thal = st.selectbox("Thalassemia (thal)", [0,1,2,3])

# Mapping giới tính
sex_val = 1 if sex == "Nam" else 0

# Nút dự đoán
if st.button("Dự đoán"):
    input_data = np.array([[age, sex_val, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("Kết quả: Nguy cơ CAO mắc bệnh tim.")
    else:
        st.success("Kết quả: KHÔNG có dấu hiệu bệnh tim.")
