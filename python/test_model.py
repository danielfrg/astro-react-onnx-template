import numpy as np
import onnxruntime as ort


session = ort.InferenceSession("../public/models/double_vector.onnx")

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
print("Input:", x)

output = session.run(None, {"input": x})

print("Output:", output)
