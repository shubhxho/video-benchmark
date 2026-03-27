import { FFmpeg, FFFSType } from "@ffmpeg/ffmpeg";
import { fetchFile } from "@ffmpeg/util";
import coreURL from "@ffmpeg/core?url";
import wasmURL from "@ffmpeg/core/wasm?url";
import workerURL from "@ffmpeg/ffmpeg/worker?url";
import type { FrameData, ProcessingProgress } from "../types.js";

const MAX_FRAMES = 120;
const MAX_ANALYSIS_DIMENSION = 720;
const MAX_FRAME_MEMORY_BYTES = 128 * 1024 * 1024;
const EXTRACTION_YIELD_INTERVAL = 4;

let ffmpegPromise: Promise<FFmpeg> | null = null;
let extractionJobId = 0;

function yieldToBrowser(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

async function getFFmpeg(): Promise<FFmpeg> {
  if (ffmpegPromise) return ffmpegPromise;

  ffmpegPromise = (async () => {
    const ffmpeg = new FFmpeg();
    await ffmpeg.load({ coreURL, wasmURL, classWorkerURL: workerURL });
    return ffmpeg;
  })().catch((error) => {
    ffmpegPromise = null;
    throw error;
  });

  return ffmpegPromise;
}

async function loadMetadata(file: File): Promise<{
  duration: number;
  width: number;
  height: number;
}> {
  try {
    return await loadMetadataWithVideoElement(file);
  } catch {
    return loadMetadataWithFFprobe(file);
  }
}

async function loadMetadataWithVideoElement(file: File): Promise<{
  duration: number;
  width: number;
  height: number;
}> {
  const url = URL.createObjectURL(file);

  try {
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.preload = "metadata";

    await new Promise<void>((resolve, reject) => {
      video.addEventListener("loadedmetadata", () => resolve(), { once: true });
      video.addEventListener("error", () => reject(new Error("Failed to load video")), {
        once: true,
      });
      video.src = url;
      video.load();
    });

    const { duration, videoWidth: width, videoHeight: height } = video;
    if (!duration || !width || !height) {
      throw new Error("Invalid video: missing duration or dimensions");
    }

    return { duration, width, height };
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function loadMetadataWithFFprobe(file: File): Promise<{
  duration: number;
  width: number;
  height: number;
}> {
  const ffmpeg = await getFFmpeg();
  const jobId = extractionJobId++;
  const outputName = `probe-${jobId}.json`;
  let mountedInput: MountedInput | null = null;

  try {
    mountedInput = await mountInputFile(ffmpeg, file, jobId);
    const exitCode = await ffmpeg.ffprobe([
      "-v",
      "error",
      "-select_streams",
      "v:0",
      "-show_entries",
      "stream=width,height:format=duration",
      "-of",
      "json",
      mountedInput.inputPath,
      "-o",
      outputName,
    ]);

    if (exitCode !== 0) {
      throw new Error(`FFprobe exited with code ${exitCode}`);
    }

    const probeOutput = await ffmpeg.readFile(outputName, "utf8");
    if (typeof probeOutput !== "string") {
      throw new Error("FFprobe did not return metadata");
    }

    const parsed = JSON.parse(probeOutput) as {
      format?: { duration?: string };
      streams?: Array<{ width?: number; height?: number }>;
    };
    const stream = parsed.streams?.[0];
    const duration = Number(parsed.format?.duration ?? 0);
    const width = Number(stream?.width ?? 0);
    const height = Number(stream?.height ?? 0);

    if (!duration || !width || !height) {
      throw new Error("Invalid video: missing duration or dimensions");
    }

    return { duration, width, height };
  } finally {
    await Promise.allSettled([
      ffmpeg.deleteFile(outputName),
      mountedInput?.cleanup(),
    ]);
  }
}

function toEven(value: number): number {
  const rounded = Math.max(2, Math.round(value));
  return rounded % 2 === 0 ? rounded : rounded - 1;
}

function getFrameBudget(duration: number): number {
  if (duration >= 60 * 60) return 48;
  if (duration >= 30 * 60) return 64;
  if (duration >= 10 * 60) return 84;
  return MAX_FRAMES;
}

function getAnalysisSize(width: number, height: number, duration: number): {
  width: number;
  height: number;
} {
  const durationScale = duration >= 60 * 60
    ? 0.7
    : duration >= 30 * 60
      ? 0.78
      : duration >= 10 * 60
        ? 0.88
        : 1;
  const maxDimension = Math.round(MAX_ANALYSIS_DIMENSION * durationScale);
  const longestEdge = Math.max(width, height);
  if (longestEdge <= maxDimension) {
    return { width: toEven(width), height: toEven(height) };
  }

  const scale = maxDimension / longestEdge;
  return {
    width: toEven(width * scale),
    height: toEven(height * scale),
  };
}

function getCappedFrameCount(rawFrameCount: number, width: number, height: number, duration: number): number {
  const targetFrameCount = Math.min(rawFrameCount, getFrameBudget(duration));
  const frameBytes = width * height * 4;
  const memoryCap = Math.max(12, Math.floor(MAX_FRAME_MEMORY_BYTES / frameBytes));
  return Math.max(1, Math.min(targetFrameCount, memoryCap));
}

function buildTimestamps(duration: number, fps: number, rawFrameCount: number, frameCount: number): number[] {
  if (frameCount <= 0) return [];
  if (frameCount === 1) return [0];

  const timestamps: number[] = [];

  if (frameCount >= rawFrameCount) {
    for (let i = 0; i < frameCount; i++) {
      timestamps.push(Math.min(duration, i / fps));
    }
    return timestamps;
  }

  for (let i = 0; i < frameCount; i++) {
    timestamps.push((i / (frameCount - 1)) * duration);
  }
  return timestamps;
}

async function extractFramesWithFFmpeg(
  file: File,
  fps: number,
  onProgress?: (progress: ProcessingProgress) => void,
): Promise<FrameData[]> {
  onProgress?.({ phase: "Loading ffmpeg wasm", current: 1, total: 1 });

  const { duration, width, height } = await loadMetadata(file);
  const rawFrameCount = Math.max(1, Math.floor(duration * fps));
  const targetSize = getAnalysisSize(width, height, duration);
  const frameCount = getCappedFrameCount(rawFrameCount, targetSize.width, targetSize.height, duration);

  if (frameCount <= 0) {
    return [];
  }

  const effectiveFps = frameCount < rawFrameCount ? frameCount / duration : fps;
  const filter = `fps=${effectiveFps},scale=${targetSize.width}:${targetSize.height}:flags=fast_bilinear`;
  const jobId = extractionJobId++;
  const outputName = `frames-${jobId}.rgba`;
  const ffmpeg = await getFFmpeg();
  let mountedInput: MountedInput | null = null;

  const progressHandler = ({ progress }: { progress: number }) => {
    const current = Math.max(1, Math.min(frameCount, Math.round(progress * frameCount)));
    onProgress?.({ phase: "Extracting frames", current, total: frameCount });
  };

  ffmpeg.on("progress", progressHandler);

  try {
    mountedInput = await mountInputFile(ffmpeg, file, jobId);
    const exitCode = await ffmpeg.exec([
      "-i",
      mountedInput.inputPath,
      "-an",
      "-vf",
      filter,
      "-frames:v",
      String(frameCount),
      "-pix_fmt",
      "rgba",
      "-f",
      "rawvideo",
      outputName,
    ]);

    if (exitCode !== 0) {
      throw new Error(`FFmpeg exited with code ${exitCode}`);
    }

    const rawFrames = await ffmpeg.readFile(outputName);
    if (!(rawFrames instanceof Uint8Array)) {
      throw new Error("FFmpeg did not return frame data");
    }

    const frameSize = targetSize.width * targetSize.height * 4;
    const extractedFrameCount = Math.floor(rawFrames.byteLength / frameSize);
    if (extractedFrameCount <= 0) {
      throw new Error("FFmpeg returned no frames");
    }

    const timestamps = buildTimestamps(duration, fps, rawFrameCount, extractedFrameCount);
    const frames: FrameData[] = [];

    for (let i = 0; i < extractedFrameCount; i++) {
      const start = i * frameSize;
      const end = start + frameSize;

      frames.push({
        pixels: rawFrames.subarray(start, end),
        width: targetSize.width,
        height: targetSize.height,
        timestamp: timestamps[i] ?? timestamps[timestamps.length - 1] ?? 0,
      });

      onProgress?.({
        phase: "Decoding frames",
        current: i + 1,
        total: extractedFrameCount,
      });
    }

    return frames;
  } finally {
    ffmpeg.off("progress", progressHandler);
    await Promise.allSettled([
      ffmpeg.deleteFile(outputName),
      mountedInput?.cleanup(),
    ]);
  }
}

type MountedInput = {
  inputPath: string;
  cleanup: () => Promise<unknown>;
};

async function mountInputFile(
  ffmpeg: FFmpeg,
  file: File,
  jobId: number,
): Promise<MountedInput> {
  const mountPoint = `/input-${jobId}`;
  const inputPath = `${mountPoint}/${file.name}`;

  try {
    await ffmpeg.createDir(mountPoint);
    await ffmpeg.mount(FFFSType.WORKERFS, { files: [file] }, mountPoint);
    return {
      inputPath,
      cleanup: async () => {
        await Promise.allSettled([ffmpeg.unmount(mountPoint)]);
        await Promise.allSettled([ffmpeg.deleteDir(mountPoint)]);
      },
    };
  } catch {
    await Promise.allSettled([ffmpeg.deleteDir(mountPoint)]);
    const extension = file.name.split(".").pop() ?? "mp4";
    const fallbackName = `input-${jobId}.${extension}`;
    await ffmpeg.writeFile(fallbackName, await fetchFile(file));
    return {
      inputPath: fallbackName,
      cleanup: () => ffmpeg.deleteFile(fallbackName),
    };
  }
}

async function extractFramesWithVideoElement(
  file: File,
  fps: number,
  onProgress?: (progress: ProcessingProgress) => void,
): Promise<FrameData[]> {
  const url = URL.createObjectURL(file);

  try {
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.preload = "auto";

    await new Promise<void>((resolve, reject) => {
      video.addEventListener("loadedmetadata", () => resolve(), { once: true });
      video.addEventListener("error", () => reject(new Error("Failed to load video")), {
        once: true,
      });
      video.src = url;
      video.load();
    });

    const duration = video.duration;
    const width = video.videoWidth;
    const height = video.videoHeight;

    if (!duration || !width || !height) {
      throw new Error("Invalid video: missing duration or dimensions");
    }

    const rawFrameCount = Math.max(1, Math.floor(duration * fps));
    const targetSize = getAnalysisSize(width, height, duration);
    const frameCount = getCappedFrameCount(rawFrameCount, targetSize.width, targetSize.height, duration);
    const timestamps = buildTimestamps(duration, fps, rawFrameCount, frameCount);

    const canvas = document.createElement("canvas");
    canvas.width = targetSize.width;
    canvas.height = targetSize.height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
      throw new Error("Failed to get canvas 2D context");
    }

    const frames: FrameData[] = [];

    for (let i = 0; i < timestamps.length; i++) {
      if (i > 0 && i % EXTRACTION_YIELD_INTERVAL === 0) {
        await yieldToBrowser();
      }

      const timestamp = timestamps[i];
      video.currentTime = timestamp;

      await new Promise<void>((resolve) => {
        video.addEventListener("seeked", () => resolve(), { once: true });
      });

      ctx.drawImage(video, 0, 0, targetSize.width, targetSize.height);
      const imageData = ctx.getImageData(0, 0, targetSize.width, targetSize.height);

      frames.push({
        pixels: new Uint8Array(imageData.data.buffer),
        width: targetSize.width,
        height: targetSize.height,
        timestamp,
      });

      onProgress?.({
        phase: "Extracting frames",
        current: i + 1,
        total: timestamps.length,
      });
    }

    return frames;
  } finally {
    URL.revokeObjectURL(url);
  }
}

export async function extractFrames(
  file: File,
  fps: number,
  onProgress?: (progress: ProcessingProgress) => void,
): Promise<FrameData[]> {
  try {
    return await extractFramesWithFFmpeg(file, fps, onProgress);
  } catch (error) {
    console.warn("FFmpeg WASM extraction failed, falling back to browser decoding", error);
    return extractFramesWithVideoElement(file, fps, onProgress);
  }
}
