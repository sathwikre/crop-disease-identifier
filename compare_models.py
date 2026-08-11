import tensorflow as tf
import numpy as np
from PIL import Image

H5_MODEL = "Team3model.h5"
TFLITE_MODEL = "Team3model_float16.tflite"

# Change this to the path of one actual leaf image
IMAGE_PATH = "test_image.jpg"

print("Loading H5 model...")
h5_model = tf.keras.models.load_model(H5_MODEL, compile=False)

print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load image exactly as your model expects
image = Image.open(IMAGE_PATH).convert("RGB")
image = image.resize((256, 256))

image_array = np.array(image, dtype=np.float32)

# IMPORTANT:
# Use the same preprocessing that your current app.py uses.
# If your app currently divides by 255, keep this:
image_array = image_array / 255.0

image_array = np.expand_dims(image_array, axis=0)

# -------------------------
# H5 prediction
# -------------------------
h5_prediction = h5_model.predict(image_array, verbose=0)
h5_class = np.argmax(h5_prediction[0])

# -------------------------
# TFLite prediction
# -------------------------
interpreter.set_tensor(
    input_details[0]["index"],
    image_array
)

interpreter.invoke()

tflite_prediction = interpreter.get_tensor(
    output_details[0]["index"]
)

tflite_class = np.argmax(tflite_prediction[0])

print("\n========== RESULT ==========")

print("H5 predicted class:     ", h5_class)
print("TFLite predicted class: ", tflite_class)

print(
    "H5 confidence:          ",
    float(np.max(h5_prediction[0]))
)

print(
    "TFLite confidence:      ",
    float(np.max(tflite_prediction[0]))
)

if h5_class == tflite_class:
    print("\n✅ SAME PREDICTION")
else:
    print("\n❌ DIFFERENT PREDICTION")