import * as ort from "onnxruntime-web/all";
import type {
  ModelStats,
  SessionResult,
  ModelResult,
  WorkerRequest,
  WorkerMessage,
} from "./types";
import { Tensor } from "onnxruntime-web";

// Set WASM path
ort.env.wasm.wasmPaths = "/onnxruntime-web/";

const MODEL_PATH = "/models/double_vector.onnx";

const stats: ModelStats = {
  device: "unknown",
  loadTime: 0,
};

class DoubleModel {
  private session: ort.InferenceSession | null = null;
  private buffer: ArrayBuffer | null = null;

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
      stats.loadTime = performance.now() - startTime;

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

    // Try each execution provider
    for (const ep of ["webgpu", "cpu"] as const) {
      try {
        console.log(`Trying execution provider: ${ep}`);

        this.session = await ort.InferenceSession.create(this.buffer, {
          executionProviders: [ep],
        });

        stats.device = ep;
        console.log(`Successfully created session with ${ep}`);

        return { success: true, device: ep };
      } catch (e) {
        console.warn(`Execution provider ${ep} not available:`, e);
        continue;
      }
    }

    // If we get here, no execution provider worked
    throw new Error("No available execution provider");
  }

  async run(inputData: number[]): Promise<ModelResult> {
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

// Create Model instance
const model = new DoubleModel();

const sendMessage = (message: WorkerMessage) => {
  // self is the global scope in a Worker
  self.postMessage(message);
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

const handleModelInit = async () => {
  sendStatus("Loading model...");

  try {
    await model.loadModel();
    sendStatus("Creating session...");

    const sessionResult = await model.createSession();
    sendMessage({
      type: "pong",
      data: sessionResult,
    });
  } catch (error) {
    console.error("Error during initialization:", error);
    // Still signal that we're loaded, but with a warning
    sendMessage({
      type: "pong",
      data: {
        success: true,
        device: "fallback",
        warning: error instanceof Error ? error.message : "Unknown error",
      },
    });
  }

  sendMessage({ type: "stats", data: stats });
};

const handleModelRun = async (input: unknown) => {
  if (!input) {
    throw new Error("No input provided for run command");
  }

  sendStatus("Running inference...");
  const result = await model.run(input);
  // await sleep(1000);
  sendMessage({
    type: "result",
    data: result,
  });
};

// Handle messages from the main thread
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

      case "stats":
        sendMessage({ type: "stats", data: stats });
        break;

      default:
        console.error(`Unknown message type: ${type}`);
        break;
    }
  } catch (error) {
    sendError(error);
  }
};

export function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
