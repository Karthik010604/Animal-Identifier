import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = tf.keras.models.load_model(
    "model/animal_model.keras",
    custom_objects={"preprocess_input": preprocess_input},
    compile=False
)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model/animal_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved model/animal_model.tflite")