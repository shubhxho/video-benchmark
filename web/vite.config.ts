import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

// Cross-origin isolation unlocks SharedArrayBuffer, which the multi-threaded
// ffmpeg-core and MediaPipe's threaded WASM builds need. Prod sets the same
// headers via vercel.json; these keep the dev/preview servers isolated too.
const crossOriginIsolationHeaders = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  assetsInclude: ["**/*.wgsl"],
  server: { headers: crossOriginIsolationHeaders },
  preview: { headers: crossOriginIsolationHeaders },
  build: {
    target: "esnext",
    outDir: "dist",
  },
});
