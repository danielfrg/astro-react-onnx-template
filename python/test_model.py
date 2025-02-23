import numpy as np
import onnxruntime as ort


x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
print("Input:", x)

ort_sess = ort.InferenceSession("../public/models/double_vector.onnx")

output = ort_sess.run(None, {"input": x})

print("Output:", output)
