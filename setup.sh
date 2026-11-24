#!/usr/bin/env bash
# AlphaDefect をローカルで準備する。何度実行しても、済んだ工程は飛ばす。
#
#   ./setup.sh                         その版で動かせるキャラの重みを取得
#   ./setup.sh --characters DEFECT,IRONCLAD
#   ./setup.sh --force                 全部やり直す
#
# 詳しい手順は docs/ja/local-run.md。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# 仮想環境がまだ無いこともあるので、まず素の Python 3.12 で入口を呼ぶ
PY="$HERE/.venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3.12 || command -v python3 || true)"
fi
if [ -z "${PY:-}" ]; then
  echo "Python 3.12 が見つかりません。Ubuntu/WSL なら: sudo apt install python3.12" >&2
  exit 1
fi
exec "$PY" -m local setup "$@"
