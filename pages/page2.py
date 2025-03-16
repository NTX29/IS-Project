import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# โหลดข้อมูล
file_path = "dataset/wine.data"
wine_df = pd.read_csv(file_path, header=None)
wine_df.columns = [
    "Class", "Alcohol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium",
    "Total_Phenols", "Flavanoids", "Nonflavanoid_Phenols", "Proanthocyanins",
    "Color_Intensity", "Hue", "OD280_OD315_of_Diluted_Wines", "Proline"
]

# แยก Features และ Target
X = wine_df.drop(columns=["Class"])
y = wine_df["Class"]

# แบ่งข้อมูลเป็น Train และ Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ทำ Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ฝึกโมเดล KNN และ SVM
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
svm_model = SVC(kernel="linear", random_state=42)
svm_model.fit(X_train_scaled, y_train)

# ประเมินผลโมเดล
knn_accuracy = accuracy_score(y_test, knn_model.predict(X_test_scaled))
svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test_scaled))

# สร้างเว็บแอปด้วย Streamlit
st.title("Wine Classification Demo & Recommendation Dashboard")

# ส่วนที่ 1: Data Visualization
tab1, tab2, tab3, tab4 = st.tabs(["Data Visualization", "AI Sommelier", "Mapping Wine Data", "Model Performance"])

with tab1:
    st.header("Explore Wine Dataset")
    feature_x = st.selectbox("เลือก Feature แกน X", X.columns)
    feature_y = st.selectbox("เลือก Feature แกน Y", X.columns)
    fig = px.scatter(wine_df, x=feature_x, y=feature_y, color=wine_df["Class"].astype(str), 
                     title=f"Scatter Plot: {feature_x} vs {feature_y}", 
                     labels={"color": "Wine Class"})
    st.plotly_chart(fig)

# ส่วนที่ 2: AI Sommelier
with tab2:
    st.header("AI Sommelier - Wine Recommendation")
    st.write("ป้อนค่าคุณสมบัติของไวน์ แล้ว AI จะช่วยแนะนำไวน์ที่เหมาะกับคุณ")

    input_features = []
    for feature in X.columns:
        value = st.slider(f"{feature}", float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))
        input_features.append(value)
    
    if st.button("Predict Wine Class"):
        input_array = np.array(input_features).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        knn_prediction = knn_model.predict(input_scaled)[0]
        svm_prediction = svm_model.predict(input_scaled)[0]
        
        st.success(f"KNN Prediction: Class {knn_prediction}")
        st.success(f"SVM Prediction: Class {svm_prediction}")

# ส่วนที่ 3: Mapping Wine Data
with tab3:
    st.header("Mapping Wine Data")
    st.write("แสดงข้อมูลไวน์ในรูปแบบ Heatmap และ PCA Mapping")
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_scaled)
    pca_df = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
    pca_df["Class"] = y_train.values
    
    fig_pca = px.scatter(pca_df, x="PCA1", y="PCA2", color=pca_df["Class"].astype(str), 
                         title="PCA Mapping of Wine Data", labels={"color": "Wine Class"})
    st.plotly_chart(fig_pca)

# ส่วนที่ 4: Model Performance
with tab4:
    st.header("Model Performance & Evaluation")
    st.write(f"**KNN Accuracy:** {knn_accuracy:.4f}")
    st.write(f"**SVM Accuracy:** {svm_accuracy:.4f}")
    
    st.subheader("Classification Report")
    st.text("KNN Classification Report:")
    st.text(classification_report(y_test, knn_model.predict(X_test_scaled)))
    st.text("SVM Classification Report:")
    st.text(classification_report(y_test, svm_model.predict(X_test_scaled)))

    # Confusion Matrix Visualization
    st.subheader("Confusion Matrix")

    knn_cm = confusion_matrix(y_test, knn_model.predict(X_test_scaled))
    svm_cm = confusion_matrix(y_test, svm_model.predict(X_test_scaled))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(knn_cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test), ax=axes[0])
    axes[0].set_title("KNN Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(svm_cm, annot=True, fmt="d", cmap="Greens", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test), ax=axes[1])
    axes[1].set_title("SVM Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    st.pyplot(fig)





