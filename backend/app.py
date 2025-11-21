from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)

model = load_model("model/GTSRB_HighAccuracy_CNN.h5")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img_bytes = file.read()

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (32,32))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_id = int(np.argmax(pred))
    confidence = float(pred[0][class_id])

    return jsonify({"class": class_id, "confidence": confidence})

@app.route("/", methods=["GET"])
def home():
    return "GTSRB API is running!"