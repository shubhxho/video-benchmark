import { HolisticLandmarker } from "@mediapipe/tasks-vision";
import type { FrameBodyMap, FrameData, FramePreview, LandmarkPoint } from "@/types";

const PREVIEW_MAX_WIDTH = 192;
const PREVIEW_QUALITY = 0.72;
const POSE_CONNECTIONS = HolisticLandmarker.POSE_CONNECTIONS;
const HAND_CONNECTIONS = HolisticLandmarker.HAND_CONNECTIONS;

function getPreviewSize(width: number, height: number): { width: number; height: number } {
  if (width <= PREVIEW_MAX_WIDTH) {
    return { width, height };
  }

  const scale = PREVIEW_MAX_WIDTH / width;
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}

export class FramePreviewRenderer {
  private readonly sourceCanvas: HTMLCanvasElement;
  private readonly previewCanvas: HTMLCanvasElement;
  private readonly sourceCtx: CanvasRenderingContext2D;
  private readonly previewCtx: CanvasRenderingContext2D;

  constructor() {
    this.sourceCanvas = document.createElement("canvas");
    this.previewCanvas = document.createElement("canvas");

    const sourceCtx = this.sourceCanvas.getContext("2d", { willReadFrequently: true });
    const previewCtx = this.previewCanvas.getContext("2d");
    if (!sourceCtx || !previewCtx) {
      throw new Error("Failed to create preview canvas");
    }

    this.sourceCtx = sourceCtx;
    this.previewCtx = previewCtx;
    this.previewCtx.imageSmoothingEnabled = true;
  }

  render(frame: FrameData, bodyMap?: FrameBodyMap | null): FramePreview {
    if (this.sourceCanvas.width !== frame.width) this.sourceCanvas.width = frame.width;
    if (this.sourceCanvas.height !== frame.height) this.sourceCanvas.height = frame.height;

    const previewSize = getPreviewSize(frame.width, frame.height);
    if (this.previewCanvas.width !== previewSize.width) this.previewCanvas.width = previewSize.width;
    if (this.previewCanvas.height !== previewSize.height) this.previewCanvas.height = previewSize.height;
    this.previewCtx.imageSmoothingEnabled = true;

    const imageData = new ImageData(new Uint8ClampedArray(frame.pixels), frame.width, frame.height);
    this.sourceCtx.putImageData(imageData, 0, 0);

    this.previewCtx.clearRect(0, 0, previewSize.width, previewSize.height);
    this.previewCtx.drawImage(this.sourceCanvas, 0, 0, previewSize.width, previewSize.height);
    if (bodyMap) {
      this.drawMap(bodyMap, previewSize.width, previewSize.height);
    }

    return {
      src: this.previewCanvas.toDataURL("image/jpeg", PREVIEW_QUALITY),
      timestamp: frame.timestamp,
    };
  }

  private drawMap(bodyMap: FrameBodyMap, width: number, height: number): void {
    this.drawConnections(bodyMap.poseLandmarks, POSE_CONNECTIONS, width, height, 2.1);
    this.drawConnections(bodyMap.leftHandLandmarks, HAND_CONNECTIONS, width, height, 1.6);
    this.drawConnections(bodyMap.rightHandLandmarks, HAND_CONNECTIONS, width, height, 1.6);

    this.drawPoints(bodyMap.poseLandmarks, width, height, 2.3);
    this.drawPoints(bodyMap.leftHandLandmarks, width, height, 1.7);
    this.drawPoints(bodyMap.rightHandLandmarks, width, height, 1.7);
  }

  private drawConnections(
    landmarks: LandmarkPoint[],
    connections: Array<{ start: number; end: number }>,
    width: number,
    height: number,
    lineWidth: number,
  ): void {
    if (landmarks.length === 0) return;

    const ctx = this.previewCtx;
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    for (const { start, end } of connections) {
      const from = landmarks[start];
      const to = landmarks[end];
      if (!from || !to) continue;
      if (from.visibility < 0.06 && to.visibility < 0.06) continue;

      const alpha = Math.max(0.18, Math.min(1, (from.visibility + to.visibility) / 2));
      ctx.beginPath();
      ctx.moveTo(from.x * width, from.y * height);
      ctx.lineTo(to.x * width, to.y * height);
      ctx.strokeStyle = `rgba(0, 0, 0, ${0.55 * alpha})`;
      ctx.lineWidth = lineWidth + 2.2;
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(from.x * width, from.y * height);
      ctx.lineTo(to.x * width, to.y * height);
      ctx.strokeStyle = `rgba(255, 255, 255, ${0.9 * alpha})`;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
    }

    ctx.restore();
  }

  private drawPoints(
    landmarks: LandmarkPoint[],
    width: number,
    height: number,
    radius: number,
  ): void {
    if (landmarks.length === 0) return;

    const ctx = this.previewCtx;
    ctx.save();

    for (const landmark of landmarks) {
      if (landmark.visibility < 0.06) continue;
      const alpha = Math.max(0.25, landmark.visibility);
      const x = landmark.x * width;
      const y = landmark.y * height;

      ctx.beginPath();
      ctx.arc(x, y, radius + 1.5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0, 0, 0, ${0.5 * alpha})`;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255, 255, 255, ${0.92 * alpha})`;
      ctx.fill();
    }

    ctx.restore();
  }
}
