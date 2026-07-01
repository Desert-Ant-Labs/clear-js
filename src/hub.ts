//
// Model resolution + caching, mirroring the emo/shapes hub: a model file is
// resolved from a local dir → cache → the Hugging Face Hub, at a pinned
// revision.

import type { Variant } from "./model.js";

export const DEFAULT_HOST = "https://huggingface.co";
export const DEFAULT_REPO = "desert-ant-labs/clear";
/** Pinned revision of the model repo. A tag (not a bare commit SHA) so it
 * survives history rewrites/squashes on the model repo. */
export const DEFAULT_REVISION = "v0.1.0";

/** ONNX file name per variant in the model repo. */
export const MODEL_FILES: Record<Variant, string> = {
  studio: "clear-studio.onnx",
  natural: "clear-natural.onnx",
};

/** Resolution + caching configuration (mutate the exported `env` to change defaults). */
export interface ClearEnv {
  /** Hugging Face host serving the model repo. */
  host: string;
  /** Model repo id, e.g. `"desert-ant-labs/clear"`. */
  repo: string;
  /** Pinned revision (commit SHA, tag, or branch). */
  revision: string;
  /** Allow fetching from the Hugging Face Hub. Set `false` to require a local copy. */
  allowRemote: boolean;
  /** Cache downloaded files (filesystem on Node, Cache Storage in the browser). */
  useCache: boolean;
  /** Directory of pre-downloaded model files to use instead of the Hub (Node). */
  localModelPath?: string;
  /** Filesystem cache directory (Node). */
  cacheDir?: string;
  /** Optional Hugging Face access token (Node, for private/gated repos). */
  token?: string;
}

/** A key/value store for cached file bytes, keyed by resolve URL. */
export interface FileCache {
  get(key: string): Promise<Uint8Array | null>;
  put(key: string, data: Uint8Array): Promise<void>;
}

export type DownloadProgress = (loaded: number, total: number) => void;

function resolveUrl(env: ClearEnv, name: string): string {
  return `${env.host}/${env.repo}/resolve/${env.revision}/${name}`;
}

async function fetchWithProgress(
  url: string,
  headers: Record<string, string> | undefined,
  onProgress?: DownloadProgress,
): Promise<Uint8Array> {
  const res = await fetch(url, headers ? { headers } : undefined);
  if (!res.ok) throw new Error(`clear: failed to fetch ${url} (${res.status} ${res.statusText})`);
  const total = Number(res.headers.get("content-length") ?? 0);
  if (!res.body || !onProgress) {
    return new Uint8Array(await res.arrayBuffer());
  }
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

/** Resolves the ONNX bytes for `variant`: local dir → cache → Hugging Face Hub. */
export async function loadModelBytes(
  env: ClearEnv,
  variant: Variant,
  cache: FileCache | null,
  readLocal?: (name: string) => Promise<Uint8Array | null>,
  onProgress?: DownloadProgress,
): Promise<Uint8Array> {
  const name = MODEL_FILES[variant];
  if (!name) throw new Error(`clear: unknown variant "${variant}"`);

  if (readLocal) {
    const local = await readLocal(name);
    if (local) return local;
  }

  const url = resolveUrl(env, name);
  if (cache && env.useCache) {
    const hit = await cache.get(url);
    if (hit) return hit;
  }
  if (!env.allowRemote) {
    throw new Error(`clear: ${name} unavailable locally and remote loading is disabled`);
  }

  const headers = env.token ? { Authorization: `Bearer ${env.token}` } : undefined;
  const data = await fetchWithProgress(url, headers, onProgress);
  if (cache && env.useCache) {
    try {
      await cache.put(url, data);
    } catch {
      /* caching is best-effort */
    }
  }
  return data;
}
