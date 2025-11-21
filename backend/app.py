# backend/app.py (TFLite-based)
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join("model", "model.tflite")

# Try to import tflite_runtime (small); fallback to tensorflow
try:
    from tflite_runtime.interpreter import Interpreter
    print("Using tflite_runtime")
except Exception:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        print("Using tensorflow lite interpreter")
    except Exception as e:
        raise RuntimeError("No tflite runtime or tensorflow available: " + str(e))

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route("/", methods=["GET"])
def home():
    return "GTSRB TFLite API is running!", 200

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_bytes = file.read()

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image data"}), 400

    img = cv2.resize(img, (32, 32))
    img = img.astype("float32") / 255.0
    inp = np.expand_dims(img, axis=0).astype(np.float32)

    # set input tensor
    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])

    class_id = int(np.argmax(output_data))
    confidence = float(np.max(output_data))

    return jsonify({"class": class_id, "confidence": confidence}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
