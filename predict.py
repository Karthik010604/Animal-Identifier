import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load model
model = load_model("model/animal_model.keras")

class_names = [
    'butterfly',
    'cat',
    'cow',
    'dog',
    'elephant',
    'hen',
    'horse',
    'sheep',
    'spider',
    'squirrel'
]

img_path = input("Enter image path: ")

# Open image
img = Image.open(img_path)

# Fix mobile image rotation
img = ImageOps.exif_transpose(img)

# Convert to RGB
img = img.convert("RGB")

# Resize
img = img.resize((224, 224))

# Convert to array
img_array = np.array(img).astype("float32")

# SAME preprocessing as training
img_array = img_array / 255.0

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array, verbose=0)

predicted_index = np.argmax(predictions)
predicted_class = class_names[predicted_index]
confidence = float(np.max(predictions)) * 100

print("\nPrediction Results")
print("------------------")

for i, cls in enumerate(class_names):
    print(f"{cls}: {predictions[0][i] * 100:.2f}%")

print("\nAnimal:", predicted_class)
print(f"Confidence: {confidence:.2f}%")