/// <reference lib="webworker" />

import { FilesetResolver, HolisticLandmarker } from "@mediapipe/tasks-vision";
import type { FrameBodyMap, LandmarkPoint, LimbScores } from "../types.js";
import {
  HOLISTIC_LANDMARKER_MODEL_URL,
  MEDIAPIPE_WASM_ROOT,
} from "../runtime-assets.js";
import { createLandmarkPostProcessor } from "../wasm/landmark-postprocess.js";
import { computeLimbScores, emptyLimbScores } from "./limb-labeling.js";

const MODEL_CANDIDATES = [
  { label: "holistic gpu", delegate: "GPU" as const },
  { label: "holistic cpu", delegate: "CPU" as const },
] as const;
const HAND_VISIBILITY_THRESHOLD = 0.08;

export type BodyMappingResult = {
  handDetected: boolean;
  bothHandsDetected: boolean;
  handConfidence: number;
  handLandmarkCount: number;
  interactionZoneCoverage: number;
  bodyDetected: boolean;
  bodyLandmarkCount: number;
  bodyVisibility: number;
  limbVisibility: number;
  limbScores: LimbScores;
  map: FrameBodyMap | null;
};

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

type ReadyResponse = { type: "ready"; modelLabel: string };
type ResultResponse = { type: "result"; id: number; result: BodyMappingResult };
type ErrorResponse = { type: "error"; id?: number; message: string };

let landmarker: HolisticLandmarker | null = null;
let postProcessor: Awaited<ReturnType<typeof createLandmarkPostProcessor>> | null = null;
let modelLabel = "";
let canvas: OffscreenCanvas | null = null;
let ctx: OffscreenCanvasRenderingContext2D | null = null;
// VIDEO running mode requires strictly increasing timestamps per instance.
let lastTimestamp = 0;

function nextTimestamp(timestampMs: number): number {
  const ts = Math.max(lastTimestamp + 1, Math.round(timestampMs) || 0);
  lastTimestamp = ts;
  return ts;
}

function isInActionZone(point: LandmarkPoint): boolean {
  return point.x >= 0.2 && point.x < 0.8 && point.y >= 0.35 && point.y < 1;
}

function clampVisibility(value: number | undefined): number {
  if (typeof value !== "number" || Number.isNaN(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function toPoints(
  landmarks: Array<{ x: number; y: number; visibility?: number }> | undefined,
): LandmarkPoint[] {
  if (!landmarks || landmarks.length === 0) return [];
  return landmarks.map((landmark) => ({
    x: landmark.x,
    y: landmark.y,
    visibility: clampVisibility(landmark.visibility),
  }));
}

function analyzeHandCoverage(map: FrameBodyMap): {
  bothHandsDetected: boolean;
  interactionZoneCoverage: number;
} {
  const leftHand = map.leftHandLandmarks.filter((point) => point.visibility >= HAND_VISIBILITY_THRESHOLD);
  const rightHand = map.rightHandLandmarks.filter((point) => point.visibility >= HAND_VISIBILITY_THRESHOLD);
  const visiblePoints = [...leftHand, ...rightHand];
  const handsVisible = Number(leftHand.length >= 5) + Number(rightHand.length >= 5);
  const pointsInActionZone = visiblePoints.filter(isInActionZone).length;

  return {
    bothHandsDetected: handsVisible === 2,
    interactionZoneCoverage:
      visiblePoints.length > 0 ? (pointsInActionZone / visiblePoints.length) * 100 : 0,
  };
}

async function init(): Promise<ReadyResponse> {
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_ROOT);
  let lastError: unknown = null;

  for (const candidate of MODEL_CANDIDATES) {
    try {
      const processor = await createLandmarkPostProcessor();
      const created = await HolisticLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: HOLISTIC_LANDMARKER_MODEL_URL,
          delegate: candidate.delegate,
        },
        runningMode: "VIDEO",
        minFaceDetectionConfidence: 0.4,
        minPoseDetectionConfidence: 0.45,
        minPosePresenceConfidence: 0.45,
        minHandLandmarksConfidence: 0.45,
        outputFaceBlendshapes: false,
        outputPoseSegmentationMasks: false,
      });
      landmarker = created;
      postProcessor = processor;
      modelLabel = `${candidate.label} + worker`;
      return { type: "ready", modelLabel };
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError instanceof Error
    ? lastError
    : new Error("Unable to load a supported body mapping model");
}

function ensureCanvas(width: number, height: number): OffscreenCanvasRenderingContext2D {
  if (!canvas) {
    canvas = new OffscreenCanvas(width, height);
    ctx = canvas.getContext("2d", { willReadFrequently: true });
  }
  if (!ctx || !canvas) {
    throw new Error("Unable to create body mapping canvas");
  }
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  return ctx;
}

function analyze(message: AnalyzeMessage): ResultResponse {
  if (!landmarker || !postProcessor) {
    throw new Error("Body mapping worker is not initialized");
  }

  const { id, width, height, pixels, timestampMs } = message;
  const framePixels = new Uint8ClampedArray(pixels);
  const localCtx = ensureCanvas(width, height);
  localCtx.putImageData(new ImageData(framePixels, width, height), 0, 0);

  const results = landmarker.detectForVideo(canvas!, nextTimestamp(timestampMs));
  const processed = postProcessor.process({
    poseLandmarks: toPoints(results.poseLandmarks[0]),
    leftHandLandmarks: toPoints(results.leftHandLandmarks[0]),
    rightHandLandmarks: toPoints(results.rightHandLandmarks[0]),
  });
  const handCoverage = analyzeHandCoverage(processed.map);
  const limbScores = processed.bodyDetected ? computeLimbScores(processed.map) : emptyLimbScores();

  return {
    type: "result",
    id,
    result: {
      handDetected: processed.handDetected,
      bothHandsDetected: handCoverage.bothHandsDetected,
      handConfidence: processed.handConfidence,
      handLandmarkCount: processed.handLandmarkCount,
      interactionZoneCoverage: handCoverage.interactionZoneCoverage,
      bodyDetected: processed.bodyDetected,
      bodyLandmarkCount: processed.bodyLandmarkCount,
      bodyVisibility: processed.bodyVisibility,
      limbVisibility: processed.limbVisibility,
      limbScores,
      map: processed.bodyDetected || processed.handDetected ? processed.map : null,
    },
  };
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
      landmarker?.close();
      landmarker = null;
      postProcessor = null;
      modelLabel = "";
      close();
    }
  } catch (error) {
    postError(
      error instanceof Error ? error.message : String(error),
      data.type === "analyze" ? data.id : undefined,
    );
  }
};
