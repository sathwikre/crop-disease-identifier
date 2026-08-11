"""Validate the input and output tensors of the deployment TFLite model."""

from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter


MODEL_PATH = Path("Team3model_float16.tflite")
EXPECTED_INPUT_SHAPE = (1, 256, 256, 3)
EXPECTED_OUTPUT_SHAPE = (1, 38)


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"{MODEL_PATH} is missing. Run download_model.py before this test."
        )

    interpreter = Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    assert tuple(input_details["shape"]) == EXPECTED_INPUT_SHAPE
    assert input_details["dtype"] == np.float32
    assert tuple(output_details["shape"]) == EXPECTED_OUTPUT_SHAPE
    assert output_details["dtype"] == np.float32

    print("TFLite model tensor shapes and dtypes are valid.")


if __name__ == "__main__":
    main()
