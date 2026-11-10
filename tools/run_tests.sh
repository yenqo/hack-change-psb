#!/usr/bin/env bash
# Тесты ядра без сети: на модулях в корне (root) и на эталоне v1.3 (baseline).
#   tools/run_tests.sh            — обе цели
#   tools/run_tests.sh root       — только корень
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3
targets=("${@:-root baseline}")
for t in ${targets[@]}; do
  echo "== CORE_TARGET=$t"
  rc=0
  out=$(CORE_TARGET="$t" "$PY" -m unittest discover -s tests 2>&1) || rc=$?
  echo "$out" | grep -vE '^\[ *[0-9.]+s\]' | tail -n 25
  [ "$rc" -eq 0 ] || { echo "FAILED: $t"; exit 1; }
done
