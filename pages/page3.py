import streamlit as st
import pandas as pd

st.title("Neural Network")
st.markdown("")
st.subheader("1. การเตรียมข้อมูล (Data Preparation)", divider="gray")
st.markdown("ก่อนที่เราจะพัฒนาโมเดลตัวนี้ เราจำเป็นต้องเตรียมข้อมูลให้พร้อมใช้งานก่อน โดยมีขั้นตอนการทำดังนี้: ")
st.markdown("**1.1 อัพโหลดข้อมูล**")
st.markdown("เรามีข้อมูลอยู่ในไฟล์ tic-tac-toe.data ซึ่งเป็นชุดข้อมูลที่บันทึกสถานะของกระดาน Tic-Tac-Toe ในช่วงท้ายเกม พร้อมกับผลลัพธ์ (ชนะหรือแพ้):")
st.markdown("-มี 9 คอลัมน์ที่แสดงสถานะของแต่ละช่องบนกระดาน:")
st.markdown("-คอลัมน์ที่ 10 เป็นผลลัพธ์ (positive หรือ negative)")
st.markdown("")

st.markdown("**1.2 แปลงข้อมูลให้อยู่ในรูปแบบที่ใช้กับโมเดลได้**")
st.markdown("เนื่องจาก **Neural Network** ทำงานกับตัวเลข จึงต้องแปลงค่าหมวดหมู่ ('x', 'o', 'b', 'positive', 'negative') เป็นค่าตัวเลข:")
st.markdown("'x' → 1, 'o' → -1, 'b' → 0")
st.markdown("'positive' → 1 (ชนะ), 'negative's → 0 (แพ้)")
st.markdown("")

st.markdown("**1.3 แบ่งชุดข้อมูลออกเป็นชุด Train และ Test**")
st.markdown("เราแบ่งข้อมูลเป็น 2 ส่วน:")
st.markdown("-**Train Set (80%)**: ใช้สำหรับฝึกโมเดล")
st.markdown("-**Test Set (20%)**: ใช้สำหรับทดสอบโมเดล")
st.markdown("")

st.subheader("2.ทฤษฏีของอัลกอริทึมที่ใช้(Algorithm)", divider="gray")
st.markdown("โมเดลที่เราใช้คือ **Artificial Neural Network (ANN)** ซึ่งเป็นเครือข่ายนิวรอนเทียมที่เลียนแบบการทำงานของสมองมนุษย์")
st.markdown("**2.1 โครงสร้างของ Neural Network**")
st.markdown("โมเดลนี้เป็น **Feedforward Neural Network** ที่มีโครงสร้างดังนี้:")
st.markdown("-**Input Layer (9 นิวรอน)** รับค่าจากตำแหน่งบนกระดาน")
st.markdown("-**Hidden Layers (2 ชั้น)** ช่วยเรียนรู้รูปแบบที่ซับซ้อนซึ่ง **ชั้นที่ 1**: 32 นิวรอน, ใช้ ReLU Activation และ **ชั้นที่ 2**: 16 นิวรอน, ใช้ ReLU Activation")
st.markdown("-**Output Layer (1 นิวรอน)** ใช้ **Sigmoid Activation** เพื่อทำนาย 0 หรือ 1")
st.markdown("")

st.markdown("**2.2 Activation Functions**")
st.markdown("**ReLU (Rectified Linear Unit)**")
st.latex(r'''
f(x) = max(0, x)
''')
st.markdown("")
st.markdown("ใช้กับ **Hidden Layers** เพื่อทำให้โมเดลเรียนรู้ฟีเจอร์เชิงลึก")
st.markdown("**Sigmoid**")
st.latex(r'''
f(x) = \frac{1}{1 + e^{-x}}
''')
st.markdown("")
st.markdown("ใช้ใน Output Layer เพื่อทำให้ค่าที่ได้อยู่ระหว่าง 0 และ 1")
st.markdown("")

st.markdown("**2.3 Loss Function**")
st.markdown("เราต้องใช้ **Binary Crossentropy Loss** เนื่องจากเป็นปัญหาจำแนกประเภท (Classification) แบบสองคลาส (positive หรือ negative):")
st.latex(r'''
Loss = -\frac{1}{N} \sum_{i=1}^{N} y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)
''')
st.markdown("")
st.markdown("ซึ่งเหมาะกับโมเดลที่ใช้ **Sigmoid Activation** ใน Output Layer")
st.markdown("")

st.markdown("**2.4 Optimizer**")
st.markdown("เราใช้ **Adam Optimizer** ซึ่งเป็นการรวมกันของ **Momentum** และ **RMSprop** เพื่อให้โมเดลเรียนรู้เร็วขึ้นและลดการแกว่งของค่าถ่วงน้ำหนัก")
st.markdown("")

st.subheader("3.ขั้นตอนการพัฒนาโมเดล(Model Development)", divider="gray")
st.markdown("**3.1 โหลดและเตรียมข้อมูล**")
code = '''
import pandas as pd
from sklearn.model_selection import train_test_split

# โหลดข้อมูล
df = pd.read_csv("tic-tac-toe.data", header=None)

# ตั้งชื่อคอลัมน์
columns = [f"cell_{i}" for i in range(9)] + ["result"]
df.columns = columns

# แปลงค่าหมวดหมู่เป็นตัวเลข
mapping = {"x": 1, "o": -1, "b": 0, "positive": 1, "negative": 0}
df.replace(mapping, inplace=True)

# แยก Features และ Labels
X = df.drop(columns=["result"])
y = df["result"]

# แบ่งข้อมูล Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''
st.code(code, language="python")
st.markdown("")

st.markdown("**3.2 สร้างโมเดล Neural Network**")
code = '''
import tensorflow as tf
from tensorflow import keras

# สร้างโมเดล
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(9,)),  # Hidden Layer 1
    keras.layers.Dense(16, activation='relu'),                    # Hidden Layer 2
    keras.layers.Dense(1, activation='sigmoid')                   # Output Layer
])

# คอมไพล์โมเดล
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# แสดงโครงสร้างโมเดล
model.summary()
'''
st.code(code, language="python")
st.markdown("")

st.markdown("**3.3 ฝึกโมเดล**")
code = '''
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))
'''
st.code(code, language="python")
st.markdown("")

st.markdown("**3.4 ประเมินผลลัพธ์**")
code = '''
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
'''
st.code(code, language="python")
st.markdown("")

st.subheader("4.การปรับปรุงและเพิ่มประสิทธิภาพของโมเดล(Model Tuning)", divider="gray")

st.markdown("**4.1 ปรับโครงสร้างของโมเดล (Architecture Tuning)**")
st.markdown("การเลือกจำนวนชั้น (layers) และ**จำนวนนิวรอน (neurons)** ในแต่ละชั้นมีผลต่อประสิทธิภาพของโมเดล")
st.markdown("**วิธีปรับแต่ง:**")
st.markdown("-เพิ่ม **Hidden Layers** → เพื่อให้โมเดลเรียนรู้ Feature ที่ซับซ้อนขึ้น")
st.markdown("-ปรับจำนวน **Neurons** → เพื่อหาค่าที่เหมาะสม (มากไปอาจ Overfitting, น้อยไปอาจ Underfitting)")
st.markdown("-ใช้ **Batch Normalization** เพื่อช่วยให้การเรียนรู้มีความเสถียร")
st.markdown("")
st.markdown("**ตัวอย่างโครงสร้างที่ปรับปรุง**")
code = '''
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

# สร้างโมเดล
model = keras.Sequential([
    Dense(64, activation='relu', input_shape=(9,)),  # เพิ่ม Neurons
    BatchNormalization(),  # เพิ่ม Normalization เพื่อทำให้ค่าเสถียรขึ้น
    Dense(32, activation='relu'),
    Dropout(0.3),  # ลด Overfitting ด้วย Dropout
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # ใช้ sigmoid สำหรับ Binary Classification
])
'''
st.code(code, language="python")
st.markdown("")
st.markdown("**ผลลัพธ์ที่คาดหวัง**: โมเดลเรียนรู้ได้ดีขึ้นและลด Overfitting")
st.markdown("")

st.markdown("**4.2 ปรับค่าพารามิเตอร์ (Hyperparameter Tuning)**")
st.markdown("การเลือกค่าที่เหมาะสมสำหรับ **Learning Rate**, **Batch Size**, และ **Epochs** มีผลต่อความเร็วและประสิทธิภาพของโมเดล")


df = pd.DataFrame(
    {
        "name": ["Learning Rate", "Batch Size", "Epochs"],
        "importance": ["กำหนดความเร็วในการปรับค่าถ่วงน้ำหนัก", "ขนาดของชุดข้อมูลที่ใช้ในการเรียนรู้แต่ละรอบ", "จำนวนรอบที่โมเดลเรียนรู้ข้อมูล"],
        "value": ["0.01, 0.001, 0.0001", "8, 16, 32, 64", "50, 100, 200"],
    }
)
st.dataframe(
    df,
    column_config={
        "name": "Hyperparameter",
        "importance": "ความสำคัญ",
        "value": "ค่าที่ควรลอง",


    },
    hide_index=True,
)
st.markdown("")
st.markdown("**ผลลัพธ์ที่คาดหวัง**: หาค่าที่เหมาะสมที่สุดสำหรับการฝึกโมเดล") 
st.markdown("")

st.markdown("**4.3 ใช้เทคนิค Regularization ลด Overfitting**")
st.markdown("เมื่อโมเดลมีความซับซ้อนมากเกินไป อาจเกิด **Overfitting** ซึ่งทำให้โมเดลแม่นยำกับข้อมูลเทรนมากเกินไป แต่ทำงานกับข้อมูลใหม่ไม่ดี")
st.markdown("**วิธีป้องกัน Overfitting**")
st.markdown("-**Dropout Layers** → ปิดใช้งานบาง Neurons ในแต่ละรอบ")
st.markdown("-**L1 / L2 Regularization** → ควบคุมค่าน้ำหนักไม่ให้มีค่ามากเกินไป")
st.markdown("-**Early Stopping** → หยุดการฝึกเมื่อประสิทธิภาพเริ่มลดลง")
st.markdown("")

st.markdown("**ตัวอย่างการใช้ Regularization**")
code = '''
from keras.regularizers import l2

model = keras.Sequential([
    Dense(64, activation='relu', kernel_regularizer=l2(0.001), input_shape=(9,)),
    Dropout(0.3),  # ปิดใช้งานบาง Neurons
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(1, activation='sigmoid')
])
'''
st.code(code, language="python")
st.markdown("")
st.markdown("**ผลลัพธ์ที่คาดหวัง**: ลด Overfitting ทำให้โมเดล Generalize ได้ดีขึ้น")
st.markdown("")

st.markdown("**4.4 ใช้ Augmentation หรือ Data Balancing ถ้าจำเป็น**")
st.markdown("ถ้าข้อมูลมีความไม่สมดุล (เช่น positive มากกว่า negative มาก ๆ) เราสามารถใช้:")
st.markdown("-**Oversampling** → เพิ่มจำนวนข้อมูลที่มีน้อย")
st.markdown("-**Undersampling** → ลดจำนวนข้อมูลที่มีมาก")
st.markdown("")
st.markdown("**ใช้ SMOTE สำหรับ Oversampling**")
code = '''
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
'''
st.code(code, language="python")
st.markdown("")
st.markdown("**ผลลัพธ์ที่คาดหวัง**: ทำให้โมเดลไม่ Bias ไปที่ค่าที่มีมากกว่า")
st.markdown("")

st.markdown("**4.5 ใช้ Cross Validation เพื่อตรวจสอบความแม่นยำ**")
st.markdown("แทนที่จะใช้ Train/Test แค่ครั้งเดียว เราสามารถใช้ **K-Fold Cross Validation**")
code = '''
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=16, validation_data=(X_val_fold, y_val_fold))
'''
st.code(code, language="python")
st.markdown("")
st.markdown("**ผลลัพธ์ที่คาดหวัง**: ได้ค่าความแม่นยำที่เสถียรกว่า")




