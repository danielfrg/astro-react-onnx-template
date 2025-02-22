export interface SessionResult {
  success: boolean;
  device: string;
  warning?: string;
  loadTime: number | null;
}

export interface ModelResult {
  output: number[];
  duration: number;
}

export interface ModelState {
  device: string | null;
  loading: boolean;
  status: string;
  inferenceTime: number | null;
}

export interface WorkerMessage {
  type: "pong" | "status" | "result" | "error";
  data: any;
}

export interface WorkerRequest {
  type: "ping" | "run";
  data?: {
    input?: number[];
  };
}
