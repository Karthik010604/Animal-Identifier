from flask import Flask, render_template, request
import numpy as np
import os

from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Load model
model = load_model(
    "model/animal_model.keras",
    custom_objects={
        "preprocess_input": preprocess_input
    },
    compile=False
)

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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No image selected"

    os.makedirs("static/uploads", exist_ok=True)

    filepath = os.path.join(
        "static/uploads",
        file.filename
    )

    file.save(filepath)

    img = Image.open(filepath)
    img = img.convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Model already contains preprocess_input
    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions)

    animal = class_names[predicted_index]

    confidence = round(
        float(np.max(predictions)) * 100,
        2
    )

    info = animal_info.get(
        animal,
        "Information unavailable."
    )

    return render_template(
        "index.html",
        animal=animal,
        confidence=confidence,
        info=info,
        image_path=filepath
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)