#!/usr/bin/env python3
"""Static server for the Clear browser demo with COOP/COEP headers.

Dev-only — do not deploy. WASM SIMD + multi-threading needs
`crossOriginIsolated`, which needs COOP + COEP on the top-level
navigation. We use `credentialless` for COEP so the page can fetch model
files cross-origin from huggingface.co without needing HF to send CORP.

  python serve.py            # serve on :8765
  python serve.py 8000       # custom port

Then open  http://localhost:<port>/examples/web/
"""
import http.server
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'credentialless')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")


os.chdir(ROOT)
print(f"serving {ROOT} on http://localhost:{PORT}/examples/web/  (credentialless COEP)")
with http.server.ThreadingHTTPServer(('127.0.0.1', PORT), Handler) as httpd:
    httpd.daemon_threads = True
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
