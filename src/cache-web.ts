import type { FileCache } from "./hub.js";

/** Cache Storage-backed cache (browser): downloaded model files persist across visits. */
export function webCache(name = "clear-model"): FileCache {
  return {
    async get(key) {
      try {
        const cache = await caches.open(name);
        const res = await cache.match(key);
        return res ? new Uint8Array(await res.arrayBuffer()) : null;
      } catch {
        return null;
      }
    },
    async put(key, data) {
      try {
        const cache = await caches.open(name);
        await cache.put(key, new Response(data as unknown as BodyInit));
      } catch {
        /* caching is best-effort */
      }
    },
  };
}
