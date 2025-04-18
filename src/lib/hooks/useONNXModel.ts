import { useState, useEffect, useRef, useCallback } from "react";
import type { ModelState, WorkerMessage, ModelResult } from "../workers/types";

// This is important for Vite to bundle the worker correctly
import ModelWorker from "../workers/onnx_double.worker.ts?worker";

export interface UseONNXModelOptions {
  onError?: (error: string) => void;
}

export function useONNXModel({ onError }: UseONNXModelOptions) {
  const [modelState, setModelState] = useState<ModelState>({
    device: null,
    loading: false,
    status: "Initializing",
    loadTime: null,
  });

  const [result, setResult] = useState<number[] | null>(null);
  const workerRef = useRef<Worker | null>(null);

  useEffect(() => {
    if (!workerRef.current) {
      try {
        const worker = new ModelWorker();
        workerRef.current = worker;

        workerRef.current.addEventListener("message", handleWorkerMessage);

        // Tell worker to initialize the model
        workerRef.current.postMessage({ type: "init" });
        setModelState((prev) => ({ ...prev, loading: true }));
      } catch (error) {
        const errorMessage =
          error instanceof Error
            ? error.message
            : "Failed to initialize worker";
        onError?.(errorMessage);
        setModelState((prev) => ({
          ...prev,
          status: `Error: ${errorMessage}`,
          loading: false,
        }));
      }
    }

    return () => {
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
      }
    };
  }, []);

  const handleWorkerMessage = useCallback(
    (event: MessageEvent<WorkerMessage>) => {
      const { type, data } = event.data;

      switch (type) {
        case "status":
          setModelState((prev) => ({ ...prev, status: data.message }));
          break;

        case "ready":
          const { success, device, warning, loadTime } = data;
          if (success) {
            setModelState((prev) => ({
              ...prev,
              loading: false,
              device,
              loadTime,
              status: warning
                ? `Model loaded with warnings: ${warning} - You may still be able to run inference.`
                : "Model loaded successfully. Ready to run inference.",
            }));
          } else {
            setModelState((prev) => ({
              ...prev,
              loading: false,
              status: "Error loading model (check console)",
            }));
          }
          break;

        case "result":
          const result = data as ModelResult;
          setResult(result.output);
          setModelState((prev) => ({
            ...prev,
            loading: false,
            status: "Inference complete",
            inferenceTime: result.duration,
          }));
          break;

        case "error":
          setModelState((prev) => ({
            ...prev,
            loading: false,
            status: `Error: ${data.message}`,
          }));
          onError?.(data.message);
          break;

        default:
          console.warn("Unknown message type:", type);
      }
    },
    [onError],
  );

  const runInference = useCallback(
    (input: number[]) => {
      if (!workerRef.current || modelState.loading) return;

      try {
        workerRef.current.postMessage({
          type: "run",
          data: { input },
        });

        setModelState((prev) => ({
          ...prev,
          loading: true,
          status: "Running inference...",
          inferenceTime: null,
        }));
        setResult(null);
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : "Unknown error";
        setModelState((prev) => ({
          ...prev,
          status: `Error running inference: ${errorMessage}`,
          loading: false,
        }));
        onError?.(errorMessage);
      }
    },
    [modelState.loading],
  );

  return {
    modelState,
    result,
    runInference,
  };
}
