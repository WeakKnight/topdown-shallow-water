#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
water_python="${WATER_PYTHON:-$project_dir/.venv/bin/python}"
if ! command -v "$water_python" >/dev/null 2>&1; then
    echo "Python environment not found: $water_python" >&2
    echo "Run: python3.13 \"$project_dir/scripts/setup_env.py\" --check" >&2
    exit 1
fi
exec "$water_python" "$project_dir/app.py" "$@"
