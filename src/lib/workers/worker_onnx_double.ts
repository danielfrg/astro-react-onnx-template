import * as ort from "onnxruntime-web/all";
import type {
  SessionResult,
  ModelResult,
  WorkerRequest,
  WorkerMessage,
} from "./types";
import { Tensor } from "onnxruntime-web";

// Important for the onnxruntime-web to work
ort.env.wasm.wasmPaths = import.meta.env.BASE_URL + "/onnxruntime-web/";

// Target model
const MODEL_PATH = import.meta.env.BASE_URL + "/models/double_vector.onnx";

class DoubleModel {
  private session: ort.InferenceSession | null = null;
  private buffer: ArrayBuffer | null = null;
  private loadTime: number = 0;

  async loadModel(): Promise<boolean> {
    console.log("Loading model from", MODEL_PATH);
    try {
      const startTime = performance.now();
      const response = await fetch(MODEL_PATH);

      if (!response.ok) {
        throw new Error(
          `Failed to load model: ${response.status} ${response.statusText}`,
        );
      }

      this.buffer = await response.arrayBuffer();
      this.loadTime = performance.now() - startTime;
      return true;
    } catch (error) {
      console.error("Error loading model:", error);
      throw error;
    }
  }

  async createSession(): Promise<SessionResult> {
    if (!this.buffer) {
      throw new Error("Model not loaded. Call loadModel first.");
    }

    for (const ep of ["webgpu", "cpu"] as const) {
      try {
        console.log(`Trying execution provider: ${ep}`);
        this.session = await ort.InferenceSession.create(this.buffer, {
          executionProviders: [ep],
        });
        console.log(`Successfully created session with ${ep}`);
        return {
          success: true,
          device: ep,
          loadTime: this.loadTime,
        };
      } catch (e) {
        console.warn(`Execution provider ${ep} not available:`, e);
        continue;
      }
    }

    throw new Error("No available execution provider");
  }

  async run(inputData: number[]): Promise<ModelResult> {
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
      const inputTensor = new Tensor("float32", inputData, [inputData.length]);
      const results = await this.session!.run({ input: inputTensor });
      const duration = performance.now() - startTime;

      return {
        output: Array.from(results.output.data as Float32Array),
        duration,
      };
    } catch (error) {
      console.error("Error running inference:", error);
      throw error;
    }
  }
}

const model = new DoubleModel();

const sendMessage = (message: WorkerMessage) => {
  self.postMessage(message);
};

const handleModelInit = async () => {
  try {
    sendStatus("Loading model...");
    await model.loadModel();

    sendStatus("Creating session...");
    const sessionResult = await model.createSession();

    // Loaded successfully
    sendMessage({
      type: "pong",
      data: sessionResult,
    });
  } catch (error) {
    console.error("Error during initialization:", error);
    sendMessage({
      type: "pong",
      data: {
        success: true,
        device: "fallback",
        warning: error instanceof Error ? error.message : "Unknown error",
      },
    });
  }
};

const handleModelRun = async (input: unknown) => {
  if (!input) {
    throw new Error("No input provided for run command");
  }

  sendStatus("Running inference...");
  const result = await model.run(input);
  sendMessage({
    type: "result",
    data: result,
  });
};

const sendError = (error: unknown) => {
  sendMessage({
    type: "error",
    data: {
      message: error instanceof Error ? error.message : "Unknown error",
    },
  });
};

const sendStatus = (message: string) => {
  sendMessage({
    type: "status",
    data: { message },
  });
};

// Handle incoming messages from the main thread
self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const { type, data } = e.data;

  try {
    switch (type) {
      case "ping":
        await handleModelInit();
        break;

      case "run":
        await handleModelRun(data?.input);
        break;

      default:
        console.error(`Unknown message type: ${type}`);
        break;
    }
  } catch (error) {
    sendError(error);
  }
};
