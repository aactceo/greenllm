"""Command line: `greenllm serve` and `greenllm report`."""
from __future__ import annotations

import argparse
import json

from .config import Settings


def main() -> None:
    ap = argparse.ArgumentParser(prog="greenllm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the gateway")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8787)
    sub.add_parser("report", help="print savings report as JSON")
    args = ap.parse_args()

    if args.cmd == "serve":
        import uvicorn
        from .app import create_app
        uvicorn.run(create_app(), host=args.host, port=args.port)
    else:
        from .store import Store
        st = Settings.from_env()
        print(json.dumps(Store(st.db_path).stats(st.grid_gco2_per_kwh), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
