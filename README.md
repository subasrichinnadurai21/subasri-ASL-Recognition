# Project Title
Your description here...
#  American Sign Language (ASL) Recognition  
### CNN + Grad-CAM + Streamlit + Real-time Webcam

This project builds a complete ASL Sign Recognition System using **TensorFlow CNN**, **Kaggle MNIST Sign dataset**, **Grad-CAM visualization**, and a **real-time Streamlit webcam application**.

---

## Project Structure

```
├── models/
│   ├── model.h5
│   ├── label_encoder.joblib
├── data/
│   ├── sign_mnist_train.csv
│   ├── sign_mnist_test.csv
├── utils.py
├── gradcam.py
├── app.py
├── models.zip
└── README.md
```

---

##  1. Dataset Loading (KaggleHub)

```python
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "datamunge/sign-language-mnist",
    "sign_mnist_train.csv",
)
df_test = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "datamunge/sign-language-mnist",
    "sign_mnist_test.csv",
)
```

---

##  2. Preprocessing

- Normalizes pixel values (0–255 → 0–1)
- Reshape to (28, 28, 1)
- LabelEncoder is saved using joblib

```python
X_train, y_train_cat, labels_train_raw, le_train = load_csv_images(train_csv)
```

---

##  3. CNN Model Architecture

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same'),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D(),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(n_classes, activation='softmax')
])
```

---

##  4. Model Training

- Train/validation split  
- ModelCheckpoint  
- EarlyStopping  
- Best model stored in `/models/model.h5`

```python
history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=64,
    callbacks=[ckpt, es]
)
```

---

##  5. Results

- Training curves saved → `training_curves.png`
- Confusion matrix saved → `confusion_matrix.png`

---

##  6. Grad-CAM Visualization

`gradcam.py` contains:

```python
heatmap = make_gradcam_heatmap(arr, model, last_conv)
overlay = overlay_heatmap_on_image(heatmap, image)
```

---

##  7. Streamlit Web App

Run locally:

```
streamlit run app.py
```

Main features:

✔ View training dataset  
✔ Test a sample image  
✔ Upload custom image  
✔ Live webcam prediction  
✔ Grad-CAM overlay on webcam  
✔ Probability chart (Top-6 classes)

Uses:

```python
from streamlit_webrtc import webrtc_streamer
```

---

##8. Live Webcam Prediction

- Converts webcam frames to grayscale  
- Resizes to 28×28  
- Runs CNN model  
- Overlays prediction text  
- Shows Grad-CAM for strong predictions  

---

## 9. Utility Files

### utils.py
- Load CSV images  
- Load LabelEncoder  
- Decode labels  
- Convert to letter  

### gradcam.py
- Generate Grad-CAM heatmap  
- Overlay heatmap  

---

## 10. Exporting Models

```python
!zip -r models.zip models
files.download('models.zip')
```

---

##  Requirements

```
streamlit
opencv-python
streamlit-webrtc
tensorflow
numpy
pandas
seaborn
matplotlib
joblib
kagglehub
```

Install:

```
pip install -r requirements.txt
```

---

##  Usage Notes

- In Colab → Webcam won’t work fully  
- Use local computer for real-time streaming  
- CSV files must be inside `/data` folder  
- models.zip must be extracted into `/content/models/` (Colab)

---

## 👤 Developer  
**Subasri Chinnadurai**  
ASL Hand Sign Recognition Project  
Deep Learning | Computer Vision | Streamlit

---

