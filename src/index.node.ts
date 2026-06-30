//
// Node / server entry. Fetches the model from the Hugging Face Hub (cached to
// disk) and runs it through onnxruntime-node on the CPU. Audio input is a
// mono Float32Array at 48 kHz — use `decodeWav` for WAV buffers, or decode
// compressed formats upstream.

import { homedir } from "node:os";
import { join } from "node:path";

import { fsCache, localReader } from "./cache-node.js";
import { DEFAULT_HOST, DEFAULT_REPO, DEFAULT_REVISION, type ClearEnv, type DownloadProgress, loadModelBytes } from "./hub.js";
import { ClearModel, type Variant } from "./model.js";
import { resolveOrt } from "./ort.js";

export { ClearModel, SR } from "./model.js";
export type { EnhanceOptions, EnhanceResult, EnhanceStage, ProgressFn, Variant } from "./model.js";
export { setOrt } from "./ort.js";
export type { Ort } from "./ort.js";
export { encodeWav, decodeWav } from "./wav.js";
export type { DecodedWav } from "./wav.js";
export { applyLimiter, measureLUFS, MASTERING_PRESETS, resolveMastering } from "./r128.js";
export type { MasteringConfig, MasteringPreset, MasteringSpec } from "./r128.js";
export { loadModelBytes, DEFAULT_HOST, DEFAULT_REPO, DEFAULT_REVISION, MODEL_FILES } from "./hub.js";
export type { ClearEnv, FileCache, DownloadProgress } from "./hub.js";

/** Bare specifier of the ONNX Runtime used on the server. */
export const ORT_SPECIFIER = "onnxruntime-node";

/** Loading configuration. Mutate before the first {@link load}, or pass overrides. */
export const env: ClearEnv = {
  host: DEFAULT_HOST,
  repo: DEFAULT_REPO,
  revision: DEFAULT_REVISION,
  allowRemote: true,
  useCache: true,
  cacheDir: process.env.CLEAR_CACHE_DIR ?? join(homedir(), ".cache", "clear"),
  localModelPath: process.env.CLEAR_LOCAL_PATH,
  token: process.env.HF_TOKEN ?? process.env.HUGGING_FACE_HUB_TOKEN,
};

export interface LoadOptions extends Partial<ClearEnv> {
  /** Quality variant. `"studio"` (default) or `"natural"`. */
  variant?: Variant;
  /** Model download progress. */
  onDownloadProgress?: DownloadProgress;
}

/** Loads a variant (local dir → disk cache → Hugging Face Hub) and builds a CPU session. */
export async function load(options: LoadOptions = {}): Promise<ClearModel> {
  const variant = options.variant ?? "studio";
  const e: ClearEnv = { ...env, ...options };
  const cache = e.useCache && e.cacheDir ? fsCache(e.cacheDir) : null;
  const readLocal = e.localModelPath ? localReader(e.localModelPath) : undefined;
  const bytes = await loadModelBytes(e, variant, cache, readLocal, options.onDownloadProgress);

  const ort = await resolveOrt(ORT_SPECIFIER);
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ["cpu"],
    executionMode: "sequential",
    graphOptimizationLevel: "all",
  });

  return new ClearModel({ ort, session, variant, backend: "cpu" });
}
