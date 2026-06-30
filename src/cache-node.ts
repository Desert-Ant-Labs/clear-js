import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

import type { FileCache } from "./hub.js";

const keyToPath = (cacheDir: string, key: string) =>
  join(cacheDir, key.replace(/^https?:\/\//, "").replace(/[^a-zA-Z0-9._/-]/g, "_"));

/** Filesystem-backed cache (Node): downloaded model files persist across runs. */
export function fsCache(cacheDir: string): FileCache {
  return {
    async get(key) {
      try {
        return new Uint8Array(await readFile(keyToPath(cacheDir, key)));
      } catch {
        return null;
      }
    },
    async put(key, data) {
      const path = keyToPath(cacheDir, key);
      await mkdir(dirname(path), { recursive: true });
      await writeFile(path, data);
    },
  };
}

/** Reads model files from a local directory (for `localModelPath`). */
export function localReader(dir: string) {
  return async (name: string): Promise<Uint8Array | null> => {
    try {
      return new Uint8Array(await readFile(join(dir, name)));
    } catch {
      return null;
    }
  };
}
