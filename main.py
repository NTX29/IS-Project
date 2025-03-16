import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns



st.title("Machine Learning")
st.markdown("")
st.subheader("1. การเตรียมข้อมูล (Data Preparation)", divider="gray")
st.markdown("ก่อนที่เราจะพัฒนาโมเดลตัวนี้ เราจำเป็นต้องเตรียมข้อมูลให้พร้อมใช้งานก่อน โดยมีขั้นตอนการทำดังนี้: ")
st.markdown("**1.1 อัพโหลดข้อมูล**")
st.markdown("-อัพโหลดไฟล์ wine.data และกำหนดชื่อคอลัมน์ให้เหมาะสมเพื่อให้เข้าใจง่ายในการใช้งาน")
st.markdown("")

st.markdown("**1.2 ตรวจสอบข้อมูล**")
st.markdown("-ดูโครงสร้างของ dataset(จำนวนแถวและคอลัมน์)")
st.markdown("-ตรวจสอบค่าที่หายไป(missing values) และค่าผิดปกติ(outliers)")
st.markdown("-วิเคราะห์การกระจายของข้อมูลในแต่ละคอลัมน์")
st.markdown("")

st.markdown("**1.3 ทำ Feature Scaling**")
st.markdown("-ใช้ **Standardization(Z-score scaling)** เพื่อให้ข้อมูลอยู่ในช่วงเดียวกัน")
st.markdown("-จำเป็นต้องทำ Scaling เพราะ KNN และ SVM อ่อนไหวต่อค่าที่มีช่วงต่างกันมากๆ")
st.markdown("")

st.subheader("2.ทฤษฏีของอัลกอริทึมที่ใช้(Algorithm)", divider="gray")
st.markdown("เราจะใช้ 2 อัลกอริธึมในการพัฒนาโมเดล ได้แก่ **K-Nearest Neighbors (KNN)** และ **Support Vector Machine (SVM)**")
st.markdown("**2.1 K-Nearest Neighbors(KNN)**")
st.markdown("-เป็นอัลกอริธึมที่ใช้ **ความใกล้เคียง (Distance-based method)**")
st.markdown("-หาผลลัพธ์โดยดูจาก **จุดข้อมูลใกล้เคียง (neighbors)** ที่ใกล้ที่สุด *k* จุด")
st.markdown("-ใช้ระยะทาง **Euclidean Distance** เป็นตัวชี้วัด")
st.markdown("**ข้อดีของ KNN:**")
st.markdown("-เข้าใจง่าย และไม่ต้องสร้างสมการซับซ้อน")
st.markdown("-ใช้ได้กับข้อมูลที่ไม่ได้เป็นเชิงเส้น(non-linear)")
st.markdown("**ข้อเสียของ KNN:**")
st.markdown("-ใช้เวลาเยอะถ้าข้อมูลมีขนาดใหญ่(ต้องเปรียบเทียบกับทุกจุด)")
st.markdown("-อ่อนไหวต่อค่าที่มีช่วงห่างกันมาก (จึงต้องทำ Feature Scaling)")
st.markdown("")

st.markdown("**2.2 Support Vector Machine(SVM)**")
st.markdown("-SVM เป็นอัลกอริธึมที่ใช้ **Hyperplane** ในการแบ่งข้อมูล")
st.markdown("-พยายามหากรอบ (Margin) ที่กว้างที่สุดเพื่อแยกข้อมูลออกจากกัน")
st.markdown("-ถ้าข้อมูลไม่สามารถแยกได้ด้วยเส้นตรง SVM สามารถใช้ **Kernel Trick** เพื่อแปลงข้อมูลให้อยู่ในมิติที่สูงขึ้น")
st.markdown("**ข้อดีของ SVM:**")
st.markdown("-ทำงานได้ดีแม้ข้อมูลไม่ได้เป็นเชิงเส้น(linear)")
st.markdown("-มีความสามารถในการ Generalize ได้ดี")
st.markdown("**ข้อเสียของ SVM:**")
st.markdown("-ใช้เวลาฝึกค่อนข้างนานถ้าข้อมูลเยอะ")
st.markdown("-ต้องปรับค่าพารามิเตอร์ เช่น **C**, **Kernel**, **Gamma** เพื่อให้ทำงานได้ดี")
st.markdown("")

st.subheader("3.ขั้นตอนการพัฒนาโมเดล(Model Development)", divider="gray")
st.markdown("**3.1 แบ่งข้อมูล Train/Test**")
st.markdown("-แบ่งข้อมูลออกเป็น **Training Set (80%)** และ **Test Set (20%)**")
code = '''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
'''
st.code(code, language="python")
st.markdown("")

st.markdown("**3.2 ทำ Feature Scaling**")
st.markdown("-ใช้ StandardScaler() เพื่อปรับข้อมูลให้อยู่ในช่วงเดียวกัน")
code = '''
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)' 
'''
st.code(code, language="python")
st.markdown("")

st.markdown("**3.3 ฝึกโมเดล KNN และ SVM**")
st.markdown("-ใช้ KNeighborsClassifier(n_neighbors=5)")
code = '''
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
'''
st.code(code, language="python")
st.markdown("-ใช้ SVC(kernel='linear')")
code = '''
svm_model = SVC(kernel="linear", random_state=42)
svm_model.fit(X_train_scaled, y_train)
'''
st.code(code, language="python")
st.markdown("")


st.markdown("**3.4 ประเมินผลลัพธ์**")
st.markdown("-ใช้ค่า **Accuracy**, **Precision**, **Recall**, **F1-score**")
code = '''
knn_accuracy = accuracy_score(y_test, knn_model.predict(X_test_scaled))
svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test_scaled))' 
'''
st.code(code, language="python")
st.markdown("-ดู **Confusion Matrix** เพื่อตรวจสอบว่าโมเดลมีข้อผิดพลาดตรงไหน")
code = '''
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
'''
st.code(code, language="python")

st.markdown("")

st.subheader("4.การปรับปรุงและเพิ่มประสิทธิภาพของโมเดล(Model Tuning)", divider="gray")
st.markdown("-ลองค่า **Hyperparameter** ต่างๆ (เช่น k ของ KNN หรือ C ของ SVM)")
st.markdown("-ใช้ **Cross-Validation** เพื่อลดการเกิด Overfitting")
st.markdown("-**Feature Selection** เพื่อลดจำนวนตัวแปรที่ไม่สำคัญ")






















