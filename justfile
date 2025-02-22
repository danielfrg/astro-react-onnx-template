copy-onnx-wasm:
    mkdir -p public/onnxruntime-web
    cp node_modules/onnxruntime-web/dist/*.wasm public/onnxruntime-web/
    echo "WASM files copied successfully to public/onnxruntime-web/"
