export interface ModelStats {
  device: string;
  loadTime: number;
}

export interface SessionResult {
  success: boolean;
  device: string;
  warning?: string;
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
  type: "status" | "pong" | "error" | "stats" | "result";
  data: any;
}

export interface WorkerRequest {
  type: "ping" | "run" | "stats";
  data?: {
    input?: number[];
  };
}
