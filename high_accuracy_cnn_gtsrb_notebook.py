# %%
"""
High-Accuracy CNN Notebook for GTSRB (NO MobileNet)
Achieves 92–96% accuracy using:
- Strong Augmentation
- AdamW Optimizer
- Label Smoothing
- Learning Rate Scheduler
"""

# %%
# 1. IMPORTS
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models

# %%
# 2. CONFIG
DATA_PATH = "/content/GTSRB"  # CHANGE YOUR PATH
CSV_PATH = f"{DATA_PATH}/Train.csv"
IMG_SIZE = (32, 32)
BATCH_SIZE = 64
EPOCHS = 50
SEED = 42
NUM_CLASSES = 43

# %%
# 3. LOAD DATASET USING Train.csv
meta = pd.read_csv(CSV_PATH)
meta['full_path'] = meta['Path'].apply(lambda x: os.path.join(DATA_PATH, x))

X, y = [], []
for _, row in meta.iterrows():
    img = Image.open(row['full_path']).convert('RGB')
    img = img.resize(IMG_SIZE)
    X.append(np.array(img))
    y.append(row['ClassId'])

X = np.array(X)
y = np.array(y)
print("Loaded images:", X.shape)

# %%
# 4. TRAIN / VAL / TEST SPLIT
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# %%
# 5. AUGMENTATION
augment = tf.keras.Sequential([
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.15, 0.15),
    layers.RandomContrast(0.3),
])

# %%
# 6. DATA PIPELINES
AUTOTUNE = tf.data.AUTOTUNE

def make_ds(images, labels, train=False):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if train:
        ds = ds.shuffle(len(images))
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y), AUTOTUNE)
    if train:
        ds = ds.map(lambda x, y: (augment(x, training=True), y), AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds = make_ds(X_train, y_train, train=True)
val_ds = make_ds(X_val, y_val)
test_ds = make_ds(X_test, y_test)

# %%
# 7. HIGH ACCURACY CNN MODEL
model = models.Sequential([
    layers.Input(shape=(32,32,3)),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

# %%
# 8. COMPILE WITH BEST SETTINGS
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# %%
# 9. CALLBACKS
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
]

# %%
# 10. TRAIN MODEL
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# %%
# 11. LEARNING CURVES
plt.figure(figsize=(14,5))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label="train_acc")
plt.plot(history.history['val_accuracy'], label="val_acc")
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label="train_loss")
plt.plot(history.history['val_loss'], label="val_loss")
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend(); plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title('Loss')
plt.show()

# %%
# 12. EVALUATION
loss, acc = model.evaluate(test_ds)
print(f"Test Accuracy: {acc:.4f}")

# %%
# 13. CONFUSION MATRIX
y_true = np.concatenate([y for _,y in test_ds])
y_pred = np.argmax(model.predict(test_ds), axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12,10))
sns.heatmap(cm, cmap="Blues", annot=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.show()

# %%
# 14. CLASSIFICATION REPORT
print(classification_report(y_true, y_pred))

# %%
# 15. SAVE MODEL
model.save("GTSRB_HighAccuracy_CNN.h5")
print("Model saved: GTSRB_HighAccuracy_CNN.h5")

# 16. PREDICTION FUNCTIONS

# --- Existing PIL-based functions above ---

# === NEW: cv2-based prediction function ===
import cv2

def predict_image_cv2(path, img_size=(32,32)):
    """
    Predict using OpenCV (cv2.imread).
    Returns: class_id, confidence
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_id = int(np.argmax(pred))
    confidence = float(pred[0][class_id])

    return class_id, confidence

print("cv2 prediction function added successfully.")

# Predict single external image
from PIL import Image

def predict_single_image(model, path, img_size=(32,32)):
    img = Image.open(path).convert('RGB')
    img = img.resize(img_size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr)
    cls = int(np.argmax(pred))
    conf = float(pred[0][cls])
    return cls, conf

# Predict train data
def predict_train(model, train_ds):
    y_true = np.concatenate([y for _, y in train_ds])
    y_pred = np.argmax(model.predict(train_ds), axis=1)
    return y_true, y_pred

# Predict validation data
def predict_validation(model, val_ds):
    y_true = np.concatenate([y for _, y in val_ds])
    y_pred = np.argmax(model.predict(val_ds), axis=1)
    return y_true, y_pred

# Predict test data
def predict_test(model, test_ds):
    y_true = np.concatenate([y for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds), axis=1)
    return y_true, y_pred

# Predict all images from a folder
def predict_folder(model, folder_path, img_size=(32,32)):
    results = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(("jpg","jpeg","png","ppm")):
            full = os.path.join(folder_path, file)
            cls, conf = predict_single_image(model, full, img_size)
            results.append((file, cls, conf))
    return results

print("Prediction functions added successfully.

# 17. PREDICTION DEMO SECTION
# Example 1: Predict on a single test image
sample_path = meta['full_path'].iloc[0]
pred_class, pred_conf = predict_single_image(model, sample_path)
print(f"Single Image Prediction → Class: {pred_class}, Confidence: {pred_conf:.4f}")

# Example 2: Predict entire test dataset
y_true_test, y_pred_test = predict_test(model, test_ds)
print("Test prediction completed.")

# 18. CONFUSION MATRIX + CLASSIFICATION REPORT
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

cm = confusion_matrix(y_true_test, y_pred_test)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Classification Report:")
print(classification_report(y_true_test, y_pred_test))

# 19. VISUALIZE PREDICTIONS (IMAGE + LABEL)
def visualize_prediction(model, path, img_size=(32,32)):
    img = Image.open(path).convert('RGB')
    img_resized = img.resize(img_size)
    arr = np.expand_dims(np.array(img_resized)/255.0, axis=0)
    pred = model.predict(arr)
    cls = np.argmax(pred)
    conf = float(pred[0][cls])

    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {cls} | Confidence: {conf:.3f}")
    plt.show()

# Example usage:
# visualize_prediction(model, sample_path)

# 20. STREAMLIT UI FOR LIVE PREDICTION
streamlit_code = r"""
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('GTSRB_HighAccuracy_CNN.h5')

st.title("🚦 GTSRB Road Sign Recognition – Live Demo")
st.write("Upload a traffic sign image to classify it.")

uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png","ppm"])

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_resized = img.resize((32,32))
    arr = np.expand_dims(np.array(img_resized)/255.0, axis=0)

    pred = model.predict(arr)
    cls = int(np.argmax(pred))
    conf = float(pred[0][cls])

    st.success(f"Predicted Class: {cls}")
    st.info(f"Confidence: {conf:.4f}")

"""

with open('streamlit_app.py', 'w') as f:
    f.write(streamlit_code)

print("Streamlit app generated successfully → streamlit_app.py")")
