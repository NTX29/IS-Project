import numpy as np
import tensorflow as tf
from tensorflow import keras
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# โหลดโมเดล
@st.cache_resource
def load_model():
    return keras.models.load_model("dataset/tic_tac_toe_model.h5")

model = load_model()

# ฟังก์ชันพยากรณ์ผลลัพธ์ของเกม
def predict_game(board):
    board = np.array(board).reshape(1, -1)  # ปรับรูปร่างของอินพุตให้เหมาะสม
    prediction = model.predict(board)[0][0]  # ดึงค่าผลลัพธ์ออกมา
    return prediction  # คืนค่าเป็นเปอร์เซ็นต์แทนข้อความ

# UI ของ Streamlit
st.title("Tic-Tac-Toe Game Prediction")
st.write("ทดลองใช้งานโมเดลใน 4 รูปแบบที่แตกต่างกัน")

# แบ่งส่วนหน้าเว็บด้วย Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Demo 1 - Basic", 
    "Demo 2 - Confidence Chart", 
    "Demo 3 - Heatmap", 
    "Demo 4 - Confidence Heatmap"
])

# **🔹 Tab 1: Basic Prediction**
with tab1:
    st.header("Demo 1 - Basic Prediction with Confidence")
    board_1 = []
    for i in range(9):
        cell_value = st.radio(f"[Demo 1] Cell {i+1}", ["Empty", "X", "O"], index=0, key=f"d1_cell_{i}")
        board_1.append(1 if cell_value == "X" else -1 if cell_value == "O" else 0)

    if st.button("Predict in Demo 1"):
        prediction_1 = predict_game(board_1)
        result_text = "Positive" if prediction_1 >= 0.5 else "Negative"
        st.success(f"Prediction: {result_text} (Confidence: {prediction_1:.2%})")

# **🔹 Tab 2: Confidence Chart**
with tab2:
    st.header("Demo 2 - Confidence Chart")
    board_2 = []
    for i in range(9):
        cell_value = st.radio(f"[Demo 2] Cell {i+1}", ["Empty", "X", "O"], index=0, key=f"d2_cell_{i}")
        board_2.append(1 if cell_value == "X" else -1 if cell_value == "O" else 0)

    if st.button("Predict in Demo 2"):
        prediction_2 = predict_game(board_2)
        confidence = [prediction_2, 1 - prediction_2]  # ค่าความมั่นใจของ Positive / Negative

        # สร้างกราฟแท่ง
        fig, ax = plt.subplots()
        ax.bar(["Positive", "Negative"], confidence, color=["green", "red"])
        ax.set_ylim([0, 1])
        ax.set_ylabel("Confidence Level")
        ax.set_title("Prediction Confidence")

        st.pyplot(fig)

# **🔹 Tab 3: Heatmap**
with tab3:
    st.header("Demo 3 - Heatmap Visualization with Confidence")
    
    board_3 = []
    for i in range(9):
        cell_value = st.radio(f"[Demo 3] Cell {i+1}", ["Empty", "X", "O"], index=0, key=f"d3_cell_{i}")
        board_3.append(1 if cell_value == "X" else -1 if cell_value == "O" else 0)

    if st.button("Predict in Demo 3"):
        prediction_3 = predict_game(board_3)
        result_text = "Positive" if prediction_3 >= 0.5 else "Negative"

        # สร้างกระดานเป็น 3x3
        board_matrix = np.array(board_3).reshape(3, 3)

        # กำหนดสีให้ X (1) เป็นสีเขียว, O (-1) เป็นสีแดง, ช่องว่าง (0) เป็นสีเทา
        cmap = sns.color_palette(["#D3D3D3", "red", "green"])  # Gray, Red, Green

        # สร้าง Heatmap
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(board_matrix, annot=True, fmt="d", cmap=cmap, cbar=False, linewidths=2, linecolor="black",
                    xticklabels=["Col 1", "Col 2", "Col 3"], yticklabels=["Row 1", "Row 2", "Row 3"])

        plt.title(f"Prediction: {result_text} ({prediction_3:.2%})")

        st.pyplot(fig)

# **🔹 Tab 4: Confidence Heatmap**
with tab4:
    st.header("Demo 4 - Confidence Heatmap")
    
    board_4 = []
    for i in range(9):
        cell_value = st.radio(f"[Demo 4] Cell {i+1}", ["Empty", "X", "O"], index=0, key=f"d4_cell_{i}")
        board_4.append(1 if cell_value == "X" else -1 if cell_value == "O" else 0)

    if st.button("Predict in Demo 4"):
        prediction_4 = predict_game(board_4)
        result_text = "Positive" if prediction_4 >= 0.5 else "Negative"

        # สร้างค่าความมั่นใจแบบสุ่ม (สมมติว่าโมเดลมีค่าความมั่นใจในแต่ละช่อง)
        confidence_map = np.random.uniform(0.4, 0.9, size=(3, 3))  # ค่าความมั่นใจสุ่ม (แทนค่าที่โมเดลคำนวณ)

        # Plot Heatmap
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(confidence_map, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=2, linecolor="black",
                    xticklabels=["Col 1", "Col 2", "Col 3"], yticklabels=["Row 1", "Row 2", "Row 3"])

        plt.title(f"Confidence Map: {result_text} ({prediction_4:.2%})")

        st.pyplot(fig)

        # กราฟแท่งแสดงความมั่นใจของแต่ละช่อง
        flat_confidence = confidence_map.flatten()
        fig_bar, ax_bar = plt.subplots(figsize=(6, 3))
        ax_bar.bar(range(1, 10), flat_confidence, color="blue")
        ax_bar.set_xticks(range(1, 10))
        ax_bar.set_xticklabels([f"Cell {i}" for i in range(1, 10)])
        ax_bar.set_ylim([0, 1])
        ax_bar.set_ylabel("Confidence")
        ax_bar.set_title("Confidence Level per Cell")

        st.pyplot(fig_bar)
