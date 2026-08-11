"""Download the trained model during the Render build."""

import shutil

from huggingface_hub import hf_hub_download

# Hugging Face repository that stores the TFLite model used by the Render build.
MODEL_REPO = "sath123-reddy/crop-disease-model"

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="Team3model_float16.tflite",
)

shutil.copy(model_path, "Team3model_float16.tflite")

print("Team3model_float16.tflite downloaded successfully.")
