#!/usr/bin/env python3
"""
루트에 있는 실험용 HTML 대시보드를 로컬에서 확실히 띄웁니다.

`python3 -m http.server` 를 다른 폴더에서 실행하면 404·빈 화면이 납니다.
이 스크립트는 항상 저장소 루트를 웹 루트로 씁니다.

실행:
  python3 serve_dashboards.py

같은 Wi-Fi의 다른 기기:
  http://<아래에 출력되는 LAN IP>:8765/
"""

from __future__ import annotations

import argparse
import os
import socket
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _lan_ipv4() -> str | None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Serve repo-root HTML dashboards.")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--bind", default="0.0.0.0", help="Use 127.0.0.1 to block LAN access.")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    os.chdir(ROOT)

    server = ThreadingHTTPServer((args.bind, args.port), SimpleHTTPRequestHandler)

    local = f"http://127.0.0.1:{args.port}/"
    print()
    print("  대시보드 서버 실행 중 (저장소 루트 = 웹 루트)")
    print(f"  → 이 컴퓨터:   {local}")
    print(f"  → 목차:       {local}index.html")
    print(f"  → DL:         {local}dl_experiment_dashboard_20260331.html")
    print(f"  → Graph:      {local}graph_experiment_dashboard_20260401.html")
    print(f"  → SageMaker:  {local}sagemaker_experiment_dashboard_20260403.html")
    lan = _lan_ipv4()
    if lan and args.bind == "0.0.0.0":
        print(f"  → 같은 Wi-Fi: http://{lan}:{args.port}/")
    print()
    print("  중지: Ctrl+C")
    print()

    if not args.no_browser:
        threading.Timer(0.35, lambda: webbrowser.open(local + "index.html")).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료합니다.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
