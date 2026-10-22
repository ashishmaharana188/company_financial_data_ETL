"""Start the configured Streamlit app in a Render-compatible HTTP process."""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    entrypoint = os.getenv("APP_ENTRYPOINT", "unified_app.py").strip()
    app_path = (ROOT / entrypoint).resolve()
    if ROOT != app_path and ROOT not in app_path.parents:
        print("APP_ENTRYPOINT must refer to a file inside the repository.", file=sys.stderr)
        return 2
    if not app_path.is_file():
        print(
            f"Configured app entrypoint does not exist: {entrypoint}. "
            "Apply the integration overlay and commit unified_app.py first.",
            file=sys.stderr,
        )
        return 2
    port = os.getenv("PORT", "10000")
    command = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.address=0.0.0.0", f"--server.port={port}",
        "--server.headless=true", "--browser.gatherUsageStats=false",
    ]
    print(f"Starting Streamlit entrypoint: {entrypoint}", flush=True)
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
