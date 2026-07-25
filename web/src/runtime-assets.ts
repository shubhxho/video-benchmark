const BASE_URL = (() => {
  const base = import.meta.env.BASE_URL || "/";
  return base.endsWith("/") ? base : `${base}/`;
})();

function assetPath(path: string): string {
  return `${BASE_URL}${path.replace(/^\/+/, "")}`;
}

function assetDir(path: string): string {
  const resolved = assetPath(path);
  return resolved.endsWith("/") ? resolved : `${resolved}/`;
}

// Multi-threaded ffmpeg-core, copied into public/vendor/ffmpeg at build time.
export const FFMPEG_CORE_MT_URL = assetPath("vendor/ffmpeg/ffmpeg-core.js");
export const FFMPEG_CORE_MT_WASM_URL = assetPath("vendor/ffmpeg/ffmpeg-core.wasm");
export const FFMPEG_CORE_MT_WORKER_URL = assetPath("vendor/ffmpeg/ffmpeg-core.worker.js");

export const MEDIAPIPE_WASM_ROOT = assetDir("vendor/mediapipe/wasm");
export const HOLISTIC_LANDMARKER_MODEL_URL = assetPath(
  "models/mediapipe/holistic_landmarker.task",
);
export const IMAGE_SEGMENTER_MULTICLASS_MODEL_URL = assetPath(
  "models/mediapipe/selfie_multiclass_256x256.tflite",
);
export const IMAGE_SEGMENTER_LANDSCAPE_MODEL_URL = assetPath(
  "models/mediapipe/selfie_segmenter_landscape.tflite",
);
export const HAND_LANDMARKER_MODEL_URL = assetPath(
  "models/mediapipe/hand_landmarker.task",
);
