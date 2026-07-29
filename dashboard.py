from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


class DashboardHandler(SimpleHTTPRequestHandler):
    web_root: Path
    run_dir: Path

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.web_root), **kwargs)

    def _send_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_bytes(path.read_bytes(), content_type)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            state_path = self.run_dir / "dashboard_state.json"
            if not state_path.exists():
                payload = json.dumps(
                    {
                        "status": "waiting",
                        "message": "Start train.py to populate this run.",
                        "run_dir": str(self.run_dir),
                    }
                ).encode()
                self._send_bytes(payload, "application/json")
                return
            self._send_file(state_path, "application/json")
            return
        if path == "/api/health":
            self._send_bytes(b'{"ok":true}', "application/json")
            return
        if path == "/artifacts/loss.png":
            self._send_file(self.run_dir / "loss.png", "image/png")
            return
        super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the local smol-glm-5.2 training dashboard"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("checkpoints/glm52-50m"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    handler = type(
        "ConfiguredDashboardHandler",
        (DashboardHandler,),
        {
            "web_root": Path(__file__).parent / "dashboard",
            "run_dir": args.run_dir.resolve(),
        },
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"smol-glm-5.2 dashboard: http://{args.host}:{args.port}")
    print(f"Reading run artifacts from: {args.run_dir.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
