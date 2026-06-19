from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

try:
    model = load_model(
        "model/animal_model.keras",
        custom_objects={
            "preprocess_input": preprocess_input
        },
        compile=False
    )

    print("MODEL LOADED SUCCESSFULLY")

except Exception as e:
    print("ERROR:")
    print(e)