from flask import Flask, render_template, request
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

app = Flask(__name__)

# Load TFLite model
interpreter = Interpreter(model_path="model/animal_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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

    # Open image directly from the upload stream (no disk write)
    img = Image.open(file.stream)
    img = img.convert("RGB")

    # Resize for the model
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference with TFLite
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]["index"])

    predicted_index = np.argmax(predictions)
    animal = class_names[predicted_index]
    confidence = round(float(np.max(predictions)) * 100, 2)
    info = animal_info.get(animal, "Information unavailable.")

    # Encode the image as base64 to display it without saving to disk
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_data_uri = f"data:image/jpeg;base64,{img_base64}"

    return render_template(
        "index.html",
        animal=animal,
        confidence=confidence,
        info=info,
        image_path=image_data_uri
    )


if __name__ == "__main__":
    app.run(debug=True)