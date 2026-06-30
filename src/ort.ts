//
// Minimal structural typing for the ONNX Runtime surface clear uses, plus
// injection. The actual runtime (`onnxruntime-web` in the browser,
// `onnxruntime-node` on the server) is a peer dependency, loaded lazily so
// the package itself stays small and bundler-agnostic.

/** A tensor — only the fields clear reads/writes. */
export interface OrtTensor {
  data: Float32Array;
  dispose?(): void;
}

export interface OrtTensorConstructor {
  new (type: "float32", data: Float32Array, dims: number[]): OrtTensor;
}

export interface OrtSession {
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>;
  release?(): Promise<void>;
}

export interface OrtSessionFactory {
  create(model: Uint8Array | ArrayBuffer | string, options?: unknown): Promise<OrtSession>;
}

/** The subset of an ONNX Runtime module that clear depends on. */
export interface Ort {
  Tensor: OrtTensorConstructor;
  InferenceSession: OrtSessionFactory;
  // `env` shape differs across runtimes; treat loosely.
  env?: Record<string, unknown>;
}

let injected: Ort | null = null;

/**
 * Inject a pre-loaded ONNX Runtime module. Call before {@link load} to avoid
 * the dynamic import (useful in bundled apps or non-standard runtimes).
 */
export function setOrt(module: Ort): void {
  injected = module;
}

/** Returns the injected runtime, if any. */
export function getInjectedOrt(): Ort | null {
  return injected;
}

/**
 * Resolve the ONNX Runtime: the injected module if set, else a lazy dynamic
 * import of `specifier`. The specifier is passed as a variable so bundlers and
 * `tsc` don't try to resolve the (optional) peer dependency at build time.
 */
export async function resolveOrt(specifier: string): Promise<Ort> {
  if (injected) return injected;
  let mod: Record<string, unknown>;
  try {
    mod = (await import(/* @vite-ignore */ specifier)) as Record<string, unknown>;
  } catch (err) {
    throw new Error(
      `clear: could not load "${specifier}". Install it as a peer dependency ` +
        `(\`npm install ${specifier}\`) or inject a runtime with \`setOrt()\`. ` +
        `Original error: ${(err as Error)?.message ?? err}`,
    );
  }
  const resolved = (mod.default ?? mod) as unknown as Ort;
  injected = resolved;
  return resolved;
}
