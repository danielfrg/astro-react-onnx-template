export interface SessionResult {
  success: boolean;
  device: string;
  warning?: string;
  inferenceTime: number | null;
}

export interface ModelResult {
  output: number[];
  duration: number;
}

export interface ModelState {
  device: string | null;
  loading: boolean;
  status: string;
  loadTime: number | null;
}

export interface WorkerMessage {
  type: "ready" | "status" | "result" | "error";
  data: any;
}

export interface WorkerRequest {
  type: "init" | "run";
  data?: {
    input?: number[];
  };
}
