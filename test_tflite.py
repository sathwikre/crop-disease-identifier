import tensorflow as tf
import numpy as np

MODEL_PATH = "Team3model.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input:")
print(input_details)

print("\nOutput:")
print(output_details)

input_shape = input_details[0]["shape"]

# Test with a random 256x256 RGB image
test_image = np.random.random(input_shape).astype(np.float32)

interpreter.set_tensor(
    input_details[0]["index"],
    test_image
)

interpreter.invoke()

prediction = interpreter.get_tensor(
    output_details[0]["index"]
)

print("\nPrediction shape:", prediction.shape)
print("Prediction successful!")
print("Predicted class:", np.argmax(prediction))