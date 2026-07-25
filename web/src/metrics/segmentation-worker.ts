/// <reference lib="webworker" />

import { FilesetResolver, ImageSegmenter } from "@mediapipe/tasks-vision";
import {
  IMAGE_SEGMENTER_LANDSCAPE_MODEL_URL,
  IMAGE_SEGMENTER_MULTICLASS_MODEL_URL,
  MEDIAPIPE_WASM_ROOT,
} from "../runtime-assets.js";
import type { SegmentationMetrics } from "../types.js";

const MODEL_CANDIDATES = [
  {
    label: "selfie multiclass cpu",
    modelAssetPath: IMAGE_SEGMENTER_MULTICLASS_MODEL_URL,
    delegate: "CPU" as const,
    labels: ["background", "hair", "body-skin", "face-skin", "clothes", "others"],
  },
  {
    label: "selfie segmenter landscape cpu",
    modelAssetPath: IMAGE_SEGMENTER_LANDSCAPE_MODEL_URL,
    delegate: "CPU" as const,
    labels: ["background", "person"],
  },
  {
    label: "selfie multiclass gpu",
    modelAssetPath: IMAGE_SEGMENTER_MULTICLASS_MODEL_URL,
    delegate: "GPU" as const,
    labels: ["background", "hair", "body-skin", "face-skin", "clothes", "others"],
  },
  {
    label: "selfie segmenter landscape gpu",
    modelAssetPath: IMAGE_SEGMENTER_LANDSCAPE_MODEL_URL,
    delegate: "GPU" as const,
    labels: ["background", "person"],
  },
] as const;

type InitMessage = { type: "init" };
type AnalyzeMessage = {
  type: "analyze";
  id: number;
  width: number;
  height: number;
  pixels: ArrayBuffer;
  timestampMs: number;
};
type DisposeMessage = { type: "dispose" };

type WorkerMessage = InitMessage | AnalyzeMessage | DisposeMessage;

type ReadyResponse = {
  type: "ready";
  modelLabel: string;
  labels: string[];
};
type ResultResponse = {
  type: "result";
  id: number;
  metrics: SegmentationMetrics;
};
type ErrorResponse = {
  type: "error";
  id?: number;
  message: string;
};

let segmenter: ImageSegmenter | null = null;
let modelLabel = "";
let labels: string[] = [];
let canvas: OffscreenCanvas | null = null;
let ctx: OffscreenCanvasRenderingContext2D | null = null;
// VIDEO running mode requires strictly increasing timestamps per instance.
let lastTimestamp = 0;

function nextTimestamp(timestampMs: number): number {
  const ts = Math.max(lastTimestamp + 1, Math.round(timestampMs) || 0);
  lastTimestamp = ts;
  return ts;
}

function isInActionZone(x: number, y: number, width: number, height: number): boolean {
  return x >= width * 0.2 && x < width * 0.8 && y >= height * 0.35 && y < height;
}

function isBorderPixel(x: number, y: number, width: number, height: number): boolean {
  const borderX = Math.max(1, Math.floor(width * 0.06));
  const borderY = Math.max(1, Math.floor(height * 0.06));
  return x < borderX || x >= width - borderX || y < borderY || y >= height - borderY;
}

function analyzeCategoryMask(
  categoryMask: Uint8Array,
  width: number,
  height: number,
  qualityScores: number[] | undefined,
): SegmentationMetrics {
  const pixelCount = width * height;
  if (pixelCount <= 0) {
    return {
      segmentationAvailable: true,
      foregroundCoverage: 0,
      actionZoneForeground: 0,
      edgeCutoff: 0,
      segmentationQuality: 0,
    };
  }

  let foregroundPixels = 0;
  let actionZonePixels = 0;
  let actionZoneForegroundPixels = 0;
  let edgeForegroundPixels = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = y * width + x;
      const category = categoryMask[index] ?? 0;
      const foreground = category !== 0;

      if (isInActionZone(x, y, width, height)) {
        actionZonePixels++;
        if (foreground) actionZoneForegroundPixels++;
      }

      if (!foreground) continue;
      foregroundPixels++;
      if (isBorderPixel(x, y, width, height)) {
        edgeForegroundPixels++;
      }
    }
  }

  const qualityScore =
    qualityScores && qualityScores.length > 0
      ? (qualityScores.reduce((sum, value) => sum + value, 0) / qualityScores.length) * 100
      : 100;

  return {
    segmentationAvailable: true,
    foregroundCoverage: (foregroundPixels / pixelCount) * 100,
    actionZoneForeground:
      actionZonePixels > 0 ? (actionZoneForegroundPixels / actionZonePixels) * 100 : 0,
    edgeCutoff: foregroundPixels > 0 ? (edgeForegroundPixels / foregroundPixels) * 100 : 0,
    segmentationQuality: qualityScore,
  };
}

async function init(): Promise<ReadyResponse> {
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_ROOT);
  let lastError: unknown = null;

  for (const candidate of MODEL_CANDIDATES) {
    try {
      segmenter = await ImageSegmenter.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: candidate.modelAssetPath,
          delegate: candidate.delegate,
        },
        runningMode: "VIDEO",
        displayNamesLocale: "en",
        outputCategoryMask: true,
        outputConfidenceMasks: false,
      });
      modelLabel = `${candidate.label} + worker`;
      labels = segmenter.getLabels();
      if (!labels.length) {
        labels = [...candidate.labels];
      }
      return { type: "ready", modelLabel, labels };
    } catch (error) {
      lastError = error;
      segmenter?.close();
      segmenter = null;
    }
  }

  throw lastError instanceof Error
    ? lastError
    : new Error("Unable to load a supported image segmenter");
}

function ensureCanvas(width: number, height: number): OffscreenCanvasRenderingContext2D {
  if (!canvas) {
    canvas = new OffscreenCanvas(width, height);
    ctx = canvas.getContext("2d", { willReadFrequently: true });
  }

  if (!ctx || !canvas) {
    throw new Error("Unable to create segmentation canvas");
  }

  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  return ctx;
}

function analyze(message: AnalyzeMessage): ResultResponse {
  if (!segmenter) {
    throw new Error("Segmentation worker is not initialized");
  }

  const { id, width, height, pixels, timestampMs } = message;
  const framePixels = new Uint8Array(pixels);
  const localCtx = ensureCanvas(width, height);
  let result = null;

  try {
    localCtx.putImageData(new ImageData(new Uint8ClampedArray(framePixels), width, height), 0, 0);
    result = segmenter.segmentForVideo(canvas!, nextTimestamp(timestampMs));

    const mask = result.categoryMask?.getAsUint8Array() ?? new Uint8Array(width * height);
    const metrics = analyzeCategoryMask(mask, width, height, result.qualityScores);
    return { type: "result", id, metrics };
  } finally {
    result?.close();
  }
}

function postError(message: string, id?: number): void {
  (self as DedicatedWorkerGlobalScope).postMessage({
    type: "error",
    id,
    message,
  } satisfies ErrorResponse);
}

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const data = event.data;

  try {
    if (data.type === "init") {
      (self as DedicatedWorkerGlobalScope).postMessage(await init());
      return;
    }

    if (data.type === "analyze") {
      (self as DedicatedWorkerGlobalScope).postMessage(analyze(data));
      return;
    }

    if (data.type === "dispose") {
      segmenter?.close();
      segmenter = null;
      labels = [];
      modelLabel = "";
      close();
    }
  } catch (error) {
    postError(error instanceof Error ? error.message : String(error), data.type === "analyze" ? data.id : undefined);
  }
};
