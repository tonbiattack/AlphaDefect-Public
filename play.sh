#!/usr/bin/env bash
# 1 局打つ／共通推論 API を試す。
#
#   ./play.sh --character DEFECT --seed ABC123 --asc 10
#   ./play.sh --character IRONCLAD --runs 3
#   ./play.sh api --character DEFECT --steps 5
#
# 詳しい手順は docs/ja/local-run.md。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="$HERE/.venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3.12 || command -v python3 || true)"
fi
if [ -z "${PY:-}" ]; then
  echo "Python 3.12 が見つかりません。Ubuntu/WSL なら: sudo apt install python3.12" >&2
  exit 1
fi

# 引数が無ければ play とみなす（`./play.sh --character ...` で動くように）
if [ "$#" -eq 0 ]; then
  exec "$PY" -m local play
fi
case "$1" in
  play|api|status|selfcheck|versions|setup) exec "$PY" -m local "$@" ;;
  *) exec "$PY" -m local play "$@" ;;
esac
