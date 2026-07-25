import type { FrameData, SegmentationMetrics } from "../types.js";
import { WorkerRpc } from "./worker-rpc.js";

export interface SegmentationAnalyzer {
  modelLabel: string;
  labels: string[];
  analyzeFrame(frame: FrameData): Promise<SegmentationMetrics>;
  destroy(): void;
}

function emptyMetrics(): SegmentationMetrics {
  return {
    segmentationAvailable: false,
    foregroundCoverage: 0,
    actionZoneForeground: 0,
    edgeCutoff: 0,
    segmentationQuality: 0,
  };
}

class WorkerSegmentationAnalyzer implements SegmentationAnalyzer {
  private constructor(
    private readonly rpc: WorkerRpc,
    readonly modelLabel: string,
    readonly labels: string[],
  ) {}

  static async create(): Promise<WorkerSegmentationAnalyzer> {
    // The new Worker(new URL(...)) call must stay inline so Vite bundles it.
    const worker = new Worker(new URL("./segmentation-worker.ts", import.meta.url), {
      type: "module",
    });
    const { rpc, ready } = await WorkerRpc.create(worker);
    const modelLabel = typeof ready.modelLabel === "string" ? ready.modelLabel : "segmentation";
    const labels = Array.isArray(ready.labels) ? (ready.labels as string[]) : [];
    return new WorkerSegmentationAnalyzer(rpc, modelLabel, labels);
  }

  async analyzeFrame(frame: FrameData): Promise<SegmentationMetrics> {
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
    return (response.metrics as SegmentationMetrics | undefined) ?? emptyMetrics();
  }

  destroy(): void {
    this.rpc.dispose();
  }
}

export async function createSegmentationAnalyzer(): Promise<SegmentationAnalyzer> {
  try {
    return await WorkerSegmentationAnalyzer.create();
  } catch (error) {
    console.warn("Segmentation worker failed to load, skipping mask metrics", error);
    return {
      modelLabel: "skipped",
      labels: [],
      analyzeFrame: async () => emptyMetrics(),
      destroy: () => undefined,
    };
  }
}
