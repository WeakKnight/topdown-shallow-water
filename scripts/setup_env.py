"""Create a project-local environment using only the Python standard library."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Also compile shaders and render a small GPU smoke test")
    args = parser.parse_args()
    if sys.implementation.name != "cpython" or not (3, 12) <= sys.version_info[:2] <= (3, 14):
        parser.error("Use CPython 3.12-3.14 (3.13 recommended) for the pinned dependencies.")
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if environment.exists():
        if not (environment / "pyvenv.cfg").is_file() or not python.is_file():
            parser.error(f"{environment} is incomplete or not a virtual environment. Rename it and retry.")
        existing = subprocess.check_output([str(python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], text=True).strip()
        expected = f"{sys.version_info.major}.{sys.version_info.minor}"
        if existing != expected:
            parser.error(f".venv uses Python {existing}, but setup uses {expected}. Use the same Python, or rename .venv to rebuild.")
        print(f"Reusing {environment}", flush=True)
    else:
        print(f"Creating {environment} with Python {sys.version.split()[0]}", flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)

    def run(*arguments):
        subprocess.run([str(python), *arguments], cwd=ROOT, check=True)

    run("-m", "pip", "install", "--upgrade", "pip")
    # Fail clearly for unsupported platforms instead of starting a source build.
    run("-m", "pip", "install", "--only-binary=:all:", "-r", str(ROOT / "requirements.txt"))
    run("-m", "pip", "check")
    run("-c", "import slangpy, numpy, PIL, imageio_ffmpeg; print('Imports OK; FFmpeg:', imageio_ffmpeg.get_ffmpeg_exe())")
    if args.check:
        run(str(ROOT / "app.py"), "--headless", "--frames", "2", "--resolution", "64",
            "--width", "128", "--height", "80", "--output", str(ROOT / "output/setup-check.png"))
    print("Setup complete. Run " + (r".\run.bat" if sys.platform == "win32" else "./run.sh"), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Setup failed: {error}\nSee README.md: system requirements and troubleshooting.", file=sys.stderr)
        sys.exit(1)
