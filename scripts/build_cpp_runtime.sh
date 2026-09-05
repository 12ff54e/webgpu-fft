#!/usr/bin/env bash
set -euo pipefail
fft_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fft_build=${1:?Usage: build_cpp_runtime.sh OUTPUT_DIRECTORY}
mkdir -p "$fft_build"
em++ -std=c++20 -O2 --use-port=emdawnwebgpu -fexceptions \
  -sALLOW_MEMORY_GROWTH=1 -sEXIT_RUNTIME=0 \
  -I"$fft_root/include" "$fft_root/tests/cpp_runtime.cpp" \
  --js-library "$fft_root/tests/cpp_bridge.js" \
  --shell-file "$fft_root/tests/cpp_shell.html" \
  -o "$fft_build/cpp_runtime.html"
