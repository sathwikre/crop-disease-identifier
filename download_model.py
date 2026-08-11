"""Download the trained INT8 TFLite model during the Render build."""

import shutil
from huggingface_hub import hf_hub_download

MODEL_REPO = "sath123-reddy/crop-disease-model"

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="Team3model_int8.tflite",
)

shutil.copy(model_path, "Team3model_int8.tflite")

print("Team3model_int8.tflite downloaded successfully.")