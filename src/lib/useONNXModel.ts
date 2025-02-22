import { useState, useEffect, useRef, useCallback } from "react";

// Types for the model state
export interface ModelState {
  device: string | null;
  loading: boolean;
  status: string;
  inferenceTime: number | null;
}

// Types for worker messages
export interface WorkerMessage {
  type: "status" | "pong" | "error" | "stats" | "result";
  data: any;
}

// Types for model results
export interface ModelResult {
  output: number[];
  duration: number;
}

// Hook options
export interface UseONNXModelOptions {
  workerPath: string;
  onError?: (error: string) => void;
}

export function useONNXModel({ workerPath, onError }: UseONNXModelOptions) {
  // State
  const [modelState, setModelState] = useState<ModelState>({
    device: null,
    loading: false,
    status: "Initializing",
    inferenceTime: null,
  });

  const [result, setResult] = useState<number[] | null>(null);
  const workerRef = useRef<Worker | null>(null);

  // Initialize worker
  useEffect(() => {
    if (!workerRef.current) {
      try {
        workerRef.current = new Worker(new URL(workerPath, import.meta.url), {
          type: "module",
        });
        workerRef.current.addEventListener("message", handleWorkerMessage);
        workerRef.current.postMessage({ type: "ping" });

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
  }, [workerPath]);

  // Handle worker messages
  const handleWorkerMessage = useCallback(
    (event: MessageEvent<WorkerMessage>) => {
      const { type, data } = event.data;

      switch (type) {
        case "status":
          setModelState((prev) => ({ ...prev, status: data.message }));
          break;

        case "pong":
          const { success, device, warning } = data;
          if (success) {
            setModelState((prev) => ({
              ...prev,
              loading: false,
              device,
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

        case "error":
          setModelState((prev) => ({
            ...prev,
            loading: false,
            status: `Error: ${data.message}`,
          }));
          onError?.(data.message);
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

        default:
          console.warn("Unknown message type:", type);
      }
    },
    [onError],
  );

  // Run inference
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
    [modelState.loading, onError],
  );

  return {
    ...modelState,
    result,
    runInference,
  };
}
