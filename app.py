from flask import Flask, render_template, request
import numpy as np
import os
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# =============================
# SAFE MODEL LOADING (CRASH PROOF)
# =============================
MODEL_PATH = "model/animal_model.keras"
model = None

try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", str(e))
    model = None


# =============================
# CLASS LABELS
# =============================
class_names = [
    "butterfly",
    "cat",
    "cow",
    "dog",
    "elephant",
    "hen",
    "horse",
    "sheep",
    "spider",
    "squirrel"
]


# =============================
# ANIMAL INFO
# =============================
animal_info = {
    "dog": "Dogs are loyal domestic animals.",
    "cat": "Cats are small carnivorous mammals.",
    "elephant": "Elephants are the largest land animals.",
    "cow": "Cows are domesticated farm animals.",
    "horse": "Horses are strong and fast animals.",
    "hen": "Hen is a domestic bird.",
    "butterfly": "Butterflies are colorful insects.",
    "sheep": "Sheep are wool-producing animals.",
    "spider": "Spiders are eight-legged arachnids.",
    "squirrel": "Squirrels are tree-dwelling rodents."
}


# =============================
# HOME ROUTE
# =============================
@app.route("/")
def home():
    return render_template("index.html")


# =============================
# PREDICT ROUTE
# =============================
@app.route("/predict", methods=["POST"])
def predict():

    # If model failed to load
    if model is None:
        return render_template(
            "index.html",
            animal="Model Error",
            confidence=0,
            info="Model failed to load. Check Render logs or fix model format.",
            image_path=None
        )

    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No image selected"

    os.makedirs("static/uploads", exist_ok=True)

    filepath = os.path.join("static/uploads", file.filename)
    file.save(filepath)

    # =============================
    # IMAGE PROCESSING
    # =============================
    img = Image.open(filepath).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # =============================
    # PREDICTION (SAFE)
    # =============================
    try:
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        animal = class_names[predicted_index]
        confidence = round(float(np.max(predictions)) * 100, 2)

        info = animal_info.get(animal, "Information unavailable.")

    except Exception as e:
        return render_template(
            "index.html",
            animal="Prediction Error",
            confidence=0,
            info=str(e),
            image_path=filepath
        )

    return render_template(
        "index.html",
        animal=animal,
        confidence=confidence,
        info=info,
        image_path=filepath
    )


# =============================
# RENDER PORT FIX
# =============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)