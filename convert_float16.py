import tensorflow as tf
import os

MODEL_PATH = "Team3model.h5"
OUTPUT_PATH = "Team3model_float16.tflite"

print("Loading original model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Converting to Float16 TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)

print("Conversion successful!")
print(f"Saved to: {OUTPUT_PATH}")
print(f"Model size: {size_mb:.2f} MB")