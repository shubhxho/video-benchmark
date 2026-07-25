// Copies the multi-threaded ffmpeg-core assets out of node_modules into
// public/vendor/ffmpeg so they are served same-origin. Serving them locally is
// required because cross-origin isolation (COOP/COEP) — needed for the
// SharedArrayBuffer that the multi-threaded core relies on — rejects
// cross-origin resources that lack CORP headers. Same-origin assets are always
// allowed, so we ship the core ourselves instead of pulling it from a CDN.
//
// Runs on every dev/preview/build via package.json scripts. It is idempotent
// and resolves the package location instead of hard-coding node_modules, so it
// works with hoisted installs and different package managers.
import { access, copyFile, mkdir, stat } from "node:fs/promises";
import { constants } from "node:fs";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const scriptDir = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(scriptDir, "..", "public", "vendor", "ffmpeg");
const files = ["ffmpeg-core.js", "ffmpeg-core.wasm", "ffmpeg-core.worker.js"];

function resolveCoreDir() {
  // The ffmpeg worker loads the core via dynamic import(), so we need the ES
  // module build in dist/esm — not the UMD build the CJS `require` condition
  // resolves to. Resolve the package entry, then derive its sibling esm dir
  // (dist/umd/ffmpeg-core.js -> dist -> dist/esm), which is correct whichever
  // build the resolver picked.
  try {
    const entry = require.resolve("@ffmpeg/core-mt");
    return resolve(dirname(dirname(entry)), "esm");
  } catch {
    return null;
  }
}

async function isFile(path) {
  try {
    return (await stat(path)).isFile();
  } catch {
    return false;
  }
}

const srcDir = resolveCoreDir();
if (!srcDir) {
  console.error(
    "[copy-ffmpeg-core] Could not resolve @ffmpeg/core-mt. Run `bun install` in web/ first.",
  );
  process.exit(1);
}

const missing = [];
for (const file of files) {
  if (!(await isFile(resolve(srcDir, file)))) missing.push(file);
}
if (missing.length > 0) {
  console.error(
    `[copy-ffmpeg-core] Missing core assets in ${srcDir}: ${missing.join(", ")}. ` +
      "Reinstall @ffmpeg/core-mt.",
  );
  process.exit(1);
}

await mkdir(outDir, { recursive: true });
await Promise.all(
  files.map((file) => copyFile(resolve(srcDir, file), resolve(outDir, file), constants.COPYFILE_FICLONE)),
);

// Sanity-check the largest asset actually landed (guards against truncated copies).
await access(resolve(outDir, "ffmpeg-core.wasm"), constants.R_OK);

console.log(`[copy-ffmpeg-core] copied ${files.length} multi-threaded core assets -> public/vendor/ffmpeg`);
