// A tiny request/response client shared by the MediaPipe workers.
//
// Both the segmentation and body-mapping workers speak the same protocol:
//   main -> worker: { type: "init" }              and { type: "analyze", id, ... }
//   worker -> main: { type: "ready", ... }        and { type: "result", id, ... }
//                   { type: "error", id?, message }
//
// This client keeps a single persistent message listener and dispatches
// responses by id through a Map, instead of adding and removing a pair of
// listeners for every frame. That avoids listener churn on the hot path and
// gives one place to propagate worker-level failures to every pending request.

export type ReadyResponse = { type: "ready" } & Record<string, unknown>;
export type ResultResponse = { type: "result"; id: number } & Record<string, unknown>;
export type ErrorResponse = { type: "error"; id?: number; message: string };
export type WorkerResponse = ReadyResponse | ResultResponse | ErrorResponse;

type Pending = {
  resolve: (value: ResultResponse) => void;
  reject: (error: Error) => void;
};

function toError(event: ErrorEvent): Error {
  return event.error instanceof Error ? event.error : new Error(event.message || "Worker error");
}

export class WorkerRpc {
  private nextId = 1;
  private readonly pending = new Map<number, Pending>();
  private disposed = false;

  private constructor(private readonly worker: Worker) {
    this.worker.addEventListener("message", this.handleMessage);
    this.worker.addEventListener("error", this.handleError);
  }

  /**
   * Send `init` to an already-constructed module worker and resolve once it
   * reports `ready`. Rejects (and terminates the worker) if init fails, so
   * callers can fall back.
   *
   * The caller must build the worker with the literal
   * `new Worker(new URL("./x-worker.ts", import.meta.url), { type: "module" })`
   * pattern so Vite can statically detect and bundle it — that call cannot be
   * hidden behind this helper.
   */
  static async create(worker: Worker): Promise<{ rpc: WorkerRpc; ready: ReadyResponse }> {
    try {
      const ready = await new Promise<ReadyResponse>((resolve, reject) => {
        const onMessage = (event: MessageEvent<WorkerResponse>) => {
          const message = event.data;
          if (message.type === "ready") {
            cleanup();
            resolve(message);
          } else if (message.type === "error") {
            cleanup();
            reject(new Error(message.message));
          }
        };
        const onError = (event: ErrorEvent) => {
          cleanup();
          reject(toError(event));
        };
        const cleanup = () => {
          worker.removeEventListener("message", onMessage);
          worker.removeEventListener("error", onError);
        };

        worker.addEventListener("message", onMessage);
        worker.addEventListener("error", onError);
        worker.postMessage({ type: "init" });
      });

      return { rpc: new WorkerRpc(worker), ready };
    } catch (error) {
      worker.terminate();
      throw error;
    }
  }

  /** Send an `analyze`-style request and resolve with the matching result. */
  request(payload: Record<string, unknown>, transfer: Transferable[] = []): Promise<ResultResponse> {
    if (this.disposed) {
      return Promise.reject(new Error("Worker has been disposed"));
    }
    const id = this.nextId++;
    return new Promise<ResultResponse>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...payload, id }, transfer);
    });
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.rejectAll(new Error("Worker has been disposed"));
    try {
      this.worker.postMessage({ type: "dispose" });
    } catch {
      // Worker may already be gone; terminate regardless.
    }
    this.worker.terminate();
  }

  private handleMessage = (event: MessageEvent<WorkerResponse>): void => {
    const message = event.data;
    if (message.type === "result") {
      const entry = this.pending.get(message.id);
      if (entry) {
        this.pending.delete(message.id);
        entry.resolve(message);
      }
    } else if (message.type === "error" && typeof message.id === "number") {
      const entry = this.pending.get(message.id);
      if (entry) {
        this.pending.delete(message.id);
        entry.reject(new Error(message.message));
      }
    }
  };

  private handleError = (event: ErrorEvent): void => {
    this.rejectAll(toError(event));
  };

  private rejectAll(error: Error): void {
    for (const entry of this.pending.values()) {
      entry.reject(error);
    }
    this.pending.clear();
  }
}
