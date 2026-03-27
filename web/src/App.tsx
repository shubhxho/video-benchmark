import { useState, useRef } from "react";
import { Sidebar } from "@/components/sidebar";
import { FileUpload } from "@/components/file-upload";
import { AgentStep, type Step } from "@/components/agent-step";
import { Verdict } from "@/components/verdict";
import { MetricsPanel } from "@/components/metrics-panel";
import { TopBar } from "@/components/top-bar";

import { initGPU } from "@/gpu/context";
import { BrightnessMetric } from "@/gpu/brightness";
import { SharpnessMetric } from "@/gpu/sharpness";
import { BlurMetric } from "@/gpu/blur";
import { OpticalFlowMetric } from "@/gpu/optical-flow";
import { extractFrames } from "@/video/frame-extractor";
import { FramePreviewRenderer } from "@/video/frame-previews";
import { BodyMapper } from "@/metrics/body-mapping";
import { createFrameAnalyzer } from "@/wasm/frame-analysis";
import { analyzeAudio } from "@/metrics/audio";
import { analyzeTemporalConsistency } from "@/metrics/temporal";
import { scoreVideo } from "@/metrics/scorer";

import type { VideoScore, FrameMetrics, AudioMetrics, FramePreview } from "@/types";

interface Run {
  name: string;
  score: number;
  grade: string;
  result: VideoScore;
  steps: Step[];
  previews: FramePreview[];
}

const ANALYSIS_YIELD_INTERVAL = 6;

function yieldToBrowser(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

export default function App() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const [steps, setSteps] = useState<Step[]>([]);
  const [result, setResult] = useState<VideoScore | null>(null);
  const [framePreviews, setFramePreviews] = useState<FramePreview[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const busy = useRef(false);
  const activeIndexRef = useRef<number | null>(null);
  const stepsRef = useRef<Step[]>([]);

  function updateSteps(updater: (current: Step[]) => Step[]) {
    const next = updater(stepsRef.current);
    stepsRef.current = next;
    setSteps(next);
  }

  const add = (id: string, label: string) =>
    updateSteps((current) => [...current, { id, label, status: "running" as const }]);
  const done = (id: string, resultValue?: string) =>
    updateSteps((current) =>
      current.map((step) => (
        step.id === id
          ? { ...step, status: "done" as const, result: resultValue }
          : step
      )),
    );
  const upd = (id: string, label: string) =>
    updateSteps((current) =>
      current.map((step) => (step.id === id ? { ...step, label } : step)),
    );

  const currentName = activeIndex !== null ? runs[activeIndex]?.name : file?.name ?? null;

  function renameRun(i: number, name: string) {
    setRuns((prev) => prev.map((r, idx) => (idx === i ? { ...r, name } : r)));
  }

  function renameActive(name: string) {
    if (activeIndex !== null) renameRun(activeIndex, name);
  }

  function newRun() {
    activeIndexRef.current = null;
    stepsRef.current = [];
    setFile(null);
    setResult(null);
    setFramePreviews([]);
    setError(null);
    setSteps([]);
    setActiveIndex(null);
  }

  function selectRun(i: number) {
    activeIndexRef.current = i;
    stepsRef.current = runs[i].steps;
    setActiveIndex(i);
    setResult(runs[i].result);
    setFramePreviews(runs[i].previews);
    setSteps(runs[i].steps);
    setFile(null);
    setError(null);
  }

  async function handleFile(f: File) {
    if (busy.current) return;
    busy.current = true;
    activeIndexRef.current = null;
    stepsRef.current = [];
    setFile(f);
    setResult(null);
    setFramePreviews([]);
    setError(null);
    setSteps([]);
    setActiveIndex(null);

    let device: GPUDevice | null = null;
    let bm: BrightnessMetric | null = null;
    let sm: SharpnessMetric | null = null;
    let blm: BlurMetric | null = null;
    let ofm: OpticalFlowMetric | null = null;
    let mapper: BodyMapper | null = null;
    let frameAnalyzer: Awaited<ReturnType<typeof createFrameAnalyzer>> | null = null;
    let audioP: Promise<AudioMetrics | null> | null = null;

    try {
      add("gpu", "gpu");
      device = await initGPU();
      done("gpu");

      add("pipe", "pipelines");
      [bm, sm, blm, ofm, frameAnalyzer] = await Promise.all([
        BrightnessMetric.create(device),
        SharpnessMetric.create(device),
        BlurMetric.create(device),
        OpticalFlowMetric.create(device),
        createFrameAnalyzer(),
      ]);
      done("pipe", "gpu + wasm");

      add("audio", "audio");
      audioP = analyzeAudio(f);

      add("frames", "frames");
      const frames = await extractFrames(f, 1, (p) =>
        upd("frames", `${p.phase.toLowerCase()} ${p.current}/${p.total}`),
      );
      if (!frames.length) { setError("no frames"); return; }
      done("frames", `${frames.length}`);

      add("map", "body map");
      try { mapper = await BodyMapper.create(); done("map", mapper.modelLabel); }
      catch { done("map", "skipped"); }

      add("analyze", "analyzing");
      const fm: FrameMetrics[] = [];
      const previews: FramePreview[] = [];
      let previewRenderer: FramePreviewRenderer | null = null;
      try {
        previewRenderer = new FramePreviewRenderer();
      } catch {
        previewRenderer = null;
      }

      for (let i = 0; i < frames.length; i++) {
        if (i > 0 && i % ANALYSIS_YIELD_INTERVAL === 0) {
          await yieldToBrowser();
        }
        upd("analyze", `analyzing ${i + 1}/${frames.length}`);
        const fr = frames[i];
        const [b, s, bl] = await Promise.all([bm.compute(fr), sm.compute(fr), blm.compute(fr)]);
        const st = i > 0 ? await ofm.compute(fr, frames[i - 1]) : -1;
        const pixelStats = frameAnalyzer.analyzeFrame(fr);
        const frameDiff = i > 0 ? frameAnalyzer.diffFrames(fr, frames[i - 1]) : -1;
        let hDet = false, hConf = 0, hLm = 0;
        let bodyDet = false, bodyLm = 0, bodyVis = 0, limbVis = 0;
        let bodyMap = null;
        if (mapper) {
          const mapping = mapper.detect(fr);
          hDet = mapping.handDetected;
          hConf = mapping.handConfidence;
          hLm = mapping.handLandmarkCount;
          bodyDet = mapping.bodyDetected;
          bodyLm = mapping.bodyLandmarkCount;
          bodyVis = mapping.bodyVisibility;
          limbVis = mapping.limbVisibility;
          bodyMap = mapping.map;
        }
        if (previewRenderer) {
          previews.push(previewRenderer.render(fr, bodyMap));
        }
        fm.push({
          timestamp: fr.timestamp,
          brightness: b,
          sharpness: s,
          blur: bl,
          stability: st,
          frameDiff,
          lumaHistogram: pixelStats.histogram,
          handDetected: hDet,
          handConfidence: hConf,
          handLandmarkCount: hLm,
          bodyDetected: bodyDet,
          bodyLandmarkCount: bodyLm,
          bodyVisibility: bodyVis,
          limbVisibility: limbVis,
        });

        if (i > 0) {
          frames[i - 1].pixels = new Uint8Array(0);
        }
      }
      frames[frames.length - 1].pixels = new Uint8Array(0);
      done("analyze");

      const audio = await audioP;
      done("audio", audio ? `${audio.overallScore.toFixed(0)}` : "none");

      add("temporal", "temporal");
      const temporal = analyzeTemporalConsistency(fm);
      done("temporal");

      const score = scoreVideo(f.name, fm, audio, temporal);
      const finalSteps = stepsRef.current;
      const nextRun: Run = {
        name: f.name,
        score: score.overallScore,
        grade: score.grade,
        result: score,
        steps: finalSteps,
        previews,
      };
      setResult(score);
      setFramePreviews(previews);
      setRuns((prev) => [...prev, nextRun]);
      const runIndex = runs.length;
      activeIndexRef.current = runIndex;
      setActiveIndex(runIndex);

    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      bm?.destroy();
      sm?.destroy();
      blm?.destroy();
      ofm?.destroy();
      mapper?.destroy();
      device?.destroy();
      busy.current = false;
    }
  }

  const showUpload = !file && !result;

  return (
    <div className="flex h-screen bg-bg font-sans">
      <Sidebar
        runs={runs}
        activeIndex={activeIndex}
        onSelect={selectRun}
        onNew={newRun}
      />

      <div className="flex flex-1 flex-col overflow-hidden">
        <TopBar
          name={currentName}
          canRename={activeIndex !== null}
          onRename={renameActive}
          result={result}
        />

        <div className="flex-1 overflow-y-auto">
          {showUpload && (
            <div className="flex flex-1 h-full items-center justify-center">
              <FileUpload onFileSelected={handleFile} />
            </div>
          )}

          {(steps.length > 0 || result) && (
            <div className="mx-auto max-w-md px-8 py-10">
              {steps.length > 0 && (
                <div className="mb-8">
                  {steps.map((s, i) => <AgentStep key={s.id} step={s} index={i} />)}
                </div>
              )}

              {error && <p className="mb-6 text-[14px] text-text">{error}</p>}

              {result && (
                <>
                  <div className="h-px bg-border mb-8" />
                  <Verdict score={result} />
                  <div className="h-px bg-border my-8" />
                  <MetricsPanel score={result} previews={framePreviews} />
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
