import wasmURL from "./landmark-postprocess.wasm?url";
import type { FrameBodyMap, LandmarkPoint } from "@/types";

const POSE_LANDMARK_COUNT = 33;
const HAND_LANDMARK_COUNT = 21;
const LIMB_POSE_INDICES = new Uint32Array([11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]);
const VISIBILITY_THRESHOLD = 0.12;

type ProcessedMap = {
  map: FrameBodyMap;
  handDetected: boolean;
  handConfidence: number;
  handLandmarkCount: number;
  bodyDetected: boolean;
  bodyLandmarkCount: number;
  bodyVisibility: number;
  limbVisibility: number;
};

interface WasmExports {
  memory: WebAssembly.Memory;
  reset: () => void;
  alloc: (size: number) => number;
  smoothLandmarks: (currentPtr: number, previousPtr: number, count: number, outPtr: number) => void;
  meanVisibility: (ptr: number, count: number) => number;
  indexedVisibility: (
    ptr: number,
    landmarkCount: number,
    indicesPtr: number,
    indexCount: number,
  ) => number;
}

function emptyPoints(count: number): LandmarkPoint[] {
  return Array.from({ length: count }, () => ({ x: 0, y: 0, visibility: 0 }));
}

function countVisible(points: LandmarkPoint[], threshold = VISIBILITY_THRESHOLD): number {
  let count = 0;
  for (const point of points) {
    if (point.visibility >= threshold) count++;
  }
  return count;
}

function meanVisibilityJs(points: LandmarkPoint[]): number {
  if (points.length === 0) return 0;
  let sum = 0;
  for (const point of points) sum += point.visibility;
  return sum / points.length;
}

function indexedVisibilityJs(points: LandmarkPoint[], indices: Uint32Array): number {
  if (indices.length === 0) return 0;
  let sum = 0;
  let valid = 0;
  for (const index of indices) {
    const point = points[index];
    if (!point) continue;
    sum += point.visibility;
    valid++;
  }
  return valid > 0 ? sum / valid : 0;
}

function encodePoints(points: LandmarkPoint[], expectedCount: number): Float32Array {
  const buffer = new Float32Array(expectedCount * 3);
  for (let i = 0; i < expectedCount; i++) {
    const point = points[i];
    if (!point) continue;
    const offset = i * 3;
    buffer[offset] = point.x;
    buffer[offset + 1] = point.y;
    buffer[offset + 2] = point.visibility;
  }
  return buffer;
}

function decodePoints(buffer: Float32Array, expectedCount: number): LandmarkPoint[] {
  const points = emptyPoints(expectedCount);
  for (let i = 0; i < expectedCount; i++) {
    const offset = i * 3;
    points[i] = {
      x: buffer[offset],
      y: buffer[offset + 1],
      visibility: buffer[offset + 2],
    };
  }
  return points;
}

class WasmLandmarkPostProcessor {
  private previousPose = new Float32Array(POSE_LANDMARK_COUNT * 3);
  private previousLeftHand = new Float32Array(HAND_LANDMARK_COUNT * 3);
  private previousRightHand = new Float32Array(HAND_LANDMARK_COUNT * 3);

  constructor(private readonly exports: WasmExports) {}

  static async create(): Promise<WasmLandmarkPostProcessor> {
    const response = await fetch(wasmURL);
    const bytes = await response.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes, {});
    return new WasmLandmarkPostProcessor(instance.exports as unknown as WasmExports);
  }

  process(rawMap: FrameBodyMap): ProcessedMap {
    const poseLandmarks = this.smoothStream(rawMap.poseLandmarks, this.previousPose, POSE_LANDMARK_COUNT);
    const leftHandLandmarks = this.smoothStream(
      rawMap.leftHandLandmarks,
      this.previousLeftHand,
      HAND_LANDMARK_COUNT,
    );
    const rightHandLandmarks = this.smoothStream(
      rawMap.rightHandLandmarks,
      this.previousRightHand,
      HAND_LANDMARK_COUNT,
    );

    const handLandmarkCount =
      countVisible(leftHandLandmarks) + countVisible(rightHandLandmarks);
    const bodyLandmarkCount = countVisible(poseLandmarks);
    const bodyVisibility = this.meanVisibility(poseLandmarks) * 100;
    const limbVisibility = this.indexedVisibility(poseLandmarks, LIMB_POSE_INDICES) * 100;
    const handConfidence =
      ((this.meanVisibility(leftHandLandmarks) + this.meanVisibility(rightHandLandmarks)) / 2);

    return {
      map: {
        poseLandmarks,
        leftHandLandmarks,
        rightHandLandmarks,
      },
      handDetected: handLandmarkCount >= 6,
      handConfidence,
      handLandmarkCount,
      bodyDetected: bodyLandmarkCount >= 8,
      bodyLandmarkCount,
      bodyVisibility,
      limbVisibility,
    };
  }

  private smoothStream(
    points: LandmarkPoint[],
    previous: Float32Array,
    expectedCount: number,
  ): LandmarkPoint[] {
    const current = encodePoints(points, expectedCount);
    const smoothed = this.runSmoothing(current, previous, expectedCount);
    previous.set(smoothed);
    return decodePoints(smoothed, expectedCount);
  }

  private runSmoothing(
    current: Float32Array,
    previous: Float32Array,
    expectedCount: number,
  ): Float32Array {
    this.exports.reset();

    const byteLength = expectedCount * 3 * Float32Array.BYTES_PER_ELEMENT;
    const currentPtr = this.exports.alloc(byteLength);
    const previousPtr = this.exports.alloc(byteLength);
    const outPtr = this.exports.alloc(byteLength);

    new Float32Array(this.exports.memory.buffer, currentPtr, expectedCount * 3).set(current);
    new Float32Array(this.exports.memory.buffer, previousPtr, expectedCount * 3).set(previous);

    this.exports.smoothLandmarks(currentPtr, previousPtr, expectedCount, outPtr);

    return new Float32Array(
      new Float32Array(this.exports.memory.buffer, outPtr, expectedCount * 3),
    );
  }

  private meanVisibility(points: LandmarkPoint[]): number {
    this.exports.reset();
    const encoded = encodePoints(points, points.length);
    const ptr = this.exports.alloc(encoded.byteLength);
    new Float32Array(this.exports.memory.buffer, ptr, encoded.length).set(encoded);
    return this.exports.meanVisibility(ptr, points.length);
  }

  private indexedVisibility(points: LandmarkPoint[], indices: Uint32Array): number {
    this.exports.reset();
    const encoded = encodePoints(points, points.length);
    const pointsPtr = this.exports.alloc(encoded.byteLength);
    const indicesPtr = this.exports.alloc(indices.byteLength);
    new Float32Array(this.exports.memory.buffer, pointsPtr, encoded.length).set(encoded);
    new Uint32Array(this.exports.memory.buffer, indicesPtr, indices.length).set(indices);
    return this.exports.indexedVisibility(pointsPtr, points.length, indicesPtr, indices.length);
  }
}

class FallbackLandmarkPostProcessor {
  private previousPose = emptyPoints(POSE_LANDMARK_COUNT);
  private previousLeftHand = emptyPoints(HAND_LANDMARK_COUNT);
  private previousRightHand = emptyPoints(HAND_LANDMARK_COUNT);

  process(rawMap: FrameBodyMap): ProcessedMap {
    const poseLandmarks = this.smooth(rawMap.poseLandmarks, this.previousPose, POSE_LANDMARK_COUNT);
    const leftHandLandmarks = this.smooth(
      rawMap.leftHandLandmarks,
      this.previousLeftHand,
      HAND_LANDMARK_COUNT,
    );
    const rightHandLandmarks = this.smooth(
      rawMap.rightHandLandmarks,
      this.previousRightHand,
      HAND_LANDMARK_COUNT,
    );

    this.previousPose = poseLandmarks;
    this.previousLeftHand = leftHandLandmarks;
    this.previousRightHand = rightHandLandmarks;

    const handLandmarkCount = countVisible(leftHandLandmarks) + countVisible(rightHandLandmarks);
    const bodyLandmarkCount = countVisible(poseLandmarks);
    const bodyVisibility = meanVisibilityJs(poseLandmarks) * 100;
    const limbVisibility = indexedVisibilityJs(poseLandmarks, LIMB_POSE_INDICES) * 100;

    return {
      map: { poseLandmarks, leftHandLandmarks, rightHandLandmarks },
      handDetected: handLandmarkCount >= 6,
      handConfidence:
        (meanVisibilityJs(leftHandLandmarks) + meanVisibilityJs(rightHandLandmarks)) / 2,
      handLandmarkCount,
      bodyDetected: bodyLandmarkCount >= 8,
      bodyLandmarkCount,
      bodyVisibility,
      limbVisibility,
    };
  }

  private smooth(
    currentPoints: LandmarkPoint[],
    previousPoints: LandmarkPoint[],
    expectedCount: number,
  ): LandmarkPoint[] {
    const output = emptyPoints(expectedCount);
    for (let i = 0; i < expectedCount; i++) {
      const current = currentPoints[i] ?? { x: 0, y: 0, visibility: 0 };
      const previous = previousPoints[i] ?? { x: 0, y: 0, visibility: 0 };

      if (current.visibility > 0.01) {
        if (previous.visibility > 0.01) {
          output[i] = {
            x: current.x * 0.72 + previous.x * 0.28,
            y: current.y * 0.72 + previous.y * 0.28,
            visibility: Math.min(1, current.visibility * 0.85 + previous.visibility * 0.15),
          };
        } else {
          output[i] = current;
        }
      } else if (previous.visibility > 0.6) {
        output[i] = {
          x: previous.x,
          y: previous.y,
          visibility: previous.visibility * 0.78,
        };
      }
    }
    return output;
  }
}

export async function createLandmarkPostProcessor(): Promise<{
  process(rawMap: FrameBodyMap): ProcessedMap;
}> {
  try {
    return await WasmLandmarkPostProcessor.create();
  } catch (error) {
    console.warn("Custom landmark wasm failed to load, using JS fallback", error);
    return new FallbackLandmarkPostProcessor();
  }
}
