import * as ort from "onnxruntime-web/all";
import { Tensor } from "onnxruntime-web";

// Set WASM path
ort.env.wasm.wasmPaths = "/onnxruntime-web/";

const MODEL_PATH = "/models/double_vector.onnx";

const stats = {
  device: "unknown",
  loadTime: 0,
};

// Class that handles the ONNX model inference
class DoubleModel {
  constructor() {
    this.session = null;
    this.buffer = null;
  }

  async loadModel() {
    console.log("Loading model from", MODEL_PATH);
    try {
      const startTime = performance.now();

      // Load the model file
      const response = await fetch(MODEL_PATH);
      if (!response.ok) {
        throw new Error(
          `Failed to load model: ${response.status} ${response.statusText}`,
        );
      }

      this.buffer = await response.arrayBuffer();
      stats.loadTime = performance.now() - startTime;

      return true;
    } catch (error) {
      console.error("Error loading model:", error);
      throw error;
    }
  }

  async createSession() {
    if (!this.buffer) {
      throw new Error("Model not loaded. Call loadModel first.");
    }

    let success = false;

    // Try each execution provider
    for (let ep of ["webgpu", "cpu"]) {
      try {
        console.log(`Trying execution provider: ${ep}`);

        this.session = await ort.InferenceSession.create(this.buffer, {
          executionProviders: [ep],
        });

        stats.device = ep;
        success = true;

        console.log(`Successfully created session with ${ep}`);
        return { success: true, device: ep };
      } catch (e) {
        console.warn(`Execution provider ${ep} not available:`, e);
        continue;
      }
    }
  }

  async run(inputData) {
    // If session wasn't created, try one more time
    if (!this.session) {
      console.warn("Session not created before run. Attempting to create now.");
      try {
        await this.createSession();
      } catch (error) {
        console.error("Failed to create session:", error);
        throw error;
      }
    }

    try {
      const startTime = performance.now();

      // Create a tensor from the input data
      const inputTensor = new Tensor("float32", inputData, [inputData.length]);

      // Run inference
      const results = await this.session.run({ input: inputTensor });

      const duration = performance.now() - startTime;

      return {
        output: Array.from(results.output.data),
        duration,
      };
    } catch (error) {
      console.error("Error running inference:", error);
      throw error;
    }
  }
}

// Create an instance
const model = new DoubleModel();

// Handle communication with the main thread
self.onmessage = async (e) => {
  const { type, data } = e.data;

  try {
    if (type === "ping") {
      self.postMessage({
        type: "status",
        data: { message: "Loading model..." },
      });

      try {
        await model.loadModel();
        self.postMessage({
          type: "status",
          data: { message: "Creating session..." },
        });

        const sessionResult = await model.createSession();

        self.postMessage({
          type: "pong",
          data: sessionResult,
        });
      } catch (error) {
        console.error("Error during initialization:", error);
        // Still signal that we're loaded, but with a warning
        self.postMessage({
          type: "pong",
          data: {
            success: true,
            device: "fallback",
            warning: error.message,
          },
        });
      }

      self.postMessage({ type: "stats", data: stats });
    } else if (type === "run") {
      const inputData = data.input;

      self.postMessage({
        type: "status",
        data: { message: "Running inference..." },
      });

      const result = await model.run(inputData);

      self.postMessage({
        type: "result",
        data: result,
      });
    } else if (type === "stats") {
      self.postMessage({ type: "stats", data: stats });
    } else {
      console.error(`Unknown message type: ${type}`);
    }
  } catch (error) {
    self.postMessage({
      type: "error",
      data: { message: error.message },
    });
  }
};
