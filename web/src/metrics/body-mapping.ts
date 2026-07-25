import type { FrameData } from "../types.js";
import type { BodyMappingResult } from "./body-mapping-worker.js";
import { WorkerRpc } from "./worker-rpc.js";

export type { BodyMappingResult } from "./body-mapping-worker.js";

// Holistic landmarking is the heaviest per-frame cost. Running it in a worker
// lets it execute concurrently with the segmentation worker and keeps the main
// thread free for the WebGPU metrics, so the ML models overlap instead of
// running serially.
export class BodyMapper {
  private constructor(
    private readonly rpc: WorkerRpc,
    readonly modelLabel: string,
  ) {}

  static async create(): Promise<BodyMapper> {
    // The new Worker(new URL(...)) call must stay inline so Vite bundles it.
    const worker = new Worker(new URL("./body-mapping-worker.ts", import.meta.url), {
      type: "module",
    });
    const { rpc, ready } = await WorkerRpc.create(worker);
    const modelLabel = typeof ready.modelLabel === "string" ? ready.modelLabel : "body map";
    return new BodyMapper(rpc, modelLabel);
  }

  async detect(frame: FrameData): Promise<BodyMappingResult> {
    const pixels = new Uint8Array(frame.pixels);
    const response = await this.rpc.request(
      {
        type: "analyze",
        width: frame.width,
        height: frame.height,
        pixels: pixels.buffer,
        timestampMs: Math.round(frame.timestamp * 1000),
      },
      [pixels.buffer],
    );
    return response.result as BodyMappingResult;
  }

  destroy(): void {
    this.rpc.dispose();
  }
}
