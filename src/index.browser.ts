//
// Browser / Worker entry. Fetches the model from the Hugging Face Hub (cached
// in Cache Storage) and runs it through onnxruntime-web (WebGPU when
// available, threaded WASM otherwise).

import { initUsage, type UsageClient } from "@desert-ant-labs/desert-ant-web";

import { webCache } from "./cache-web.js";
import { DEFAULT_HOST, DEFAULT_REPO, DEFAULT_REVISION, type ClearEnv, type DownloadProgress, loadModelBytes } from "./hub.js";
import { ClearModel, SR, type Variant } from "./model.js";
import { resolveOrt } from "./ort.js";

export { ClearModel, SR } from "./model.js";
export type { EnhanceOptions, EnhanceResult, EnhanceStage, ProgressFn, Variant } from "./model.js";
export { setOrt } from "./ort.js";
export type { Ort } from "./ort.js";
export { decodeToMono } from "./decode-web.js";
export { encodeWav, decodeWav } from "./wav.js";
export type { DecodedWav } from "./wav.js";
export { applyLimiter, measureLUFS, MASTERING_PRESETS, resolveMastering } from "./r128.js";
export type { MasteringConfig, MasteringPreset, MasteringSpec } from "./r128.js";
export { loadModelBytes, DEFAULT_HOST, DEFAULT_REPO, DEFAULT_REVISION, MODEL_FILES } from "./hub.js";
export type { ClearEnv, FileCache, DownloadProgress } from "./hub.js";

/** Bare specifier of the ONNX Runtime used in browsers/workers. */
export const ORT_SPECIFIER = "onnxruntime-web";

/** Loading configuration. Mutate before the first {@link load}, or pass overrides. */
export const env: ClearEnv = {
  host: DEFAULT_HOST,
  repo: DEFAULT_REPO,
  revision: DEFAULT_REVISION,
  allowRemote: true,
  useCache: true,
};

export interface LoadOptions extends Partial<ClearEnv> {
  /** Quality variant. `"studio"` (default) or `"natural"`. */
  variant?: Variant;
  /** Skip WebGPU even when available. */
  forceWasm?: boolean;
  /** Override the WASM thread count (lower eases SharedArrayBuffer pressure). */
  numThreads?: number;
  /** Model download progress. */
  onDownloadProgress?: DownloadProgress;
  /** Compilation-phase notifications. */
  onPhase?: (phase: "compiling-webgpu" | "compiling-wasm") => void;
  /** Usage/telemetry key. Omit for keyless (the site Origin identifies usage). */
  usageKey?: string;
}

let usage: UsageClient | null = null;

function instrument(model: ClearModel, usageKey?: string): ClearModel {
  const enhance = model.enhance.bind(model);
  model.enhance = async (pcm, options) => {
    usage ??= initUsage({ key: usageKey });
    const result = await enhance(pcm, options);
    usage.recordCall();
    return result;
  };
  return model;
}

/** Loads a variant from the Hugging Face Hub (cached in Cache Storage) and builds a session. */
export async function load(options: LoadOptions = {}): Promise<ClearModel> {
  const variant = options.variant ?? "studio";
  const e: ClearEnv = { ...env, ...options };
  const cache = e.useCache ? webCache() : null;
  const bytes = await loadModelBytes(e, variant, cache, undefined, options.onDownloadProgress);

  const ort = await resolveOrt(ORT_SPECIFIER);
  const ortEnv = (ort.env ?? {}) as Record<string, unknown>;
  ortEnv.logLevel = "error";
  const wasm = (ortEnv.wasm ?? (ortEnv.wasm = {})) as Record<string, unknown>;
  wasm.simd = true;
  const nav = (globalThis as { navigator?: { userAgent?: string; hardwareConcurrency?: number; gpu?: unknown } }).navigator;
  const isSafari = nav?.userAgent ? /^((?!chrome|android).)*safari/i.test(nav.userAgent) : false;
  const crossOriginIsolated = (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true;
  const autoThreads = crossOriginIsolated && !isSafari ? Math.min(4, nav?.hardwareConcurrency || 1) : 1;
  wasm.numThreads = options.numThreads ?? autoThreads;

  const sessionOptions = {
    enableMemPattern: false,
    enableCpuMemArena: false,
    executionMode: "sequential",
    graphOptimizationLevel: "all",
  };

  let session = null as Awaited<ReturnType<typeof ort.InferenceSession.create>> | null;
  let backend = "wasm";
  if (nav && "gpu" in nav && !options.forceWasm) {
    options.onPhase?.("compiling-webgpu");
    try {
      session = await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"], ...sessionOptions });
      backend = "webgpu";
    } catch (e2) {
      // Fall back to WASM.
      console.warn("[clear] WebGPU init failed, falling back to WASM:", (e2 as Error)?.message ?? e2);
    }
  }
  if (!session) {
    options.onPhase?.("compiling-wasm");
    session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], ...sessionOptions });
  }

  return instrument(new ClearModel({ ort, session, variant, backend, yieldFn: yieldToBrowser }), options.usageKey);
}

function yieldToBrowser(): Promise<void> {
  const raf = (globalThis as { requestAnimationFrame?: (cb: () => void) => void }).requestAnimationFrame;
  if (typeof raf === "function") return new Promise((r) => raf(() => r()));
  return new Promise((r) => setTimeout(r, 0));
}
