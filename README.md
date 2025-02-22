# Astro React ONNX runtime template

Template for an Astro website that loads an ONNX model and uses ONNX runtime
to run it, all in client side.

- Creates a test model is that doubles a vector
- Web Worker that loads the models and communicates with a React Component

Demo: [danielfrg.github.io/astro-react-onnx-template](https://danielfrg.github.io/astro-react-onnx-template/)

![alt text](https://github.com/danielfrg/astro-react-onnx-template/blob/main/public/images/preview.png?raw=true)

## Create the test model

This is done using Python, since most models are done in Python.

```
cd python

uv sync

uv run python gen_double_model.py
```

Optimization: For most models you want to pass the models through a simplifier
like [`onnxsim`](https://github.com/daquexian/onnx-simplifier/)
and possible also convert them to ORT:

```
onnxsim model model.sim.onnx
uv run python -m onnxruntime.tools.convert_onnx_models_to_ort model.onnx
```

## Web setup

Regular astro website:

```
npm install
```

Copy the WASM ONNX Runtime from `node_modules` to the to the `/public` dir so
they can be loaded.

```
just copy-onnx-wasm
```

```
npm run dev
```
