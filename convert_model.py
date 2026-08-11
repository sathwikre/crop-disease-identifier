import tensorflow as tf
import os

MODEL_PATH = "Team3model.h5"
OUTPUT_PATH = "Team3model.tflite"

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("Converting to TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Dynamic range quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

print("Conversion successful!")
print(f"Saved to: {OUTPUT_PATH}")
print(f"Size: {os.path.getsize(OUTPUT_PATH) / (1024 * 1024):.2f} MB")