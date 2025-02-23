from pathlib import Path

import onnx
import torch
import torch.nn as nn
import torch.onnx


class DoubleModel(nn.Module):
    def __init__(self):
        super(DoubleModel, self).__init__()

    def forward(self, x):
        return x * 2.0


def export_model(fpath):
    model = DoubleModel()

    model.eval()

    dummy_input = torch.randn(4, dtype=torch.float32)

    full_path = fpath / "double_vector.onnx"

    test_input = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    with torch.no_grad():
        output = model(test_input)

    print(f"input: {test_input.tolist()}")
    print(f"output: {output.tolist()}")

    torch.onnx.export(
        model,
        dummy_input,
        full_path,
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"Model saved: {full_path}")


if __name__ == "__main__":
    export_model(Path("../public/models"))

    onnx_model = onnx.load("../public/models/double_vector.onnx")
    onnx.checker.check_model(onnx_model)
    print("Model is valid")
