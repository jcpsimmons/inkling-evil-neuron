#!/usr/bin/env bash
# One command: build a patched llama.cpp, download Inkling 2 Mini, serve it with one or more neurons pushed.
#
#   ./steer.sh evil              evil neuron at its default dose (+0.75)
#   ./steer.sh sexual            sexual neuron at its default dose (+2.0)
#   ./steer.sh evil sexual       both at once
#   ./steer.sh sexual:3          pick your own dose
#
# Then open http://127.0.0.1:8080 to chat (OpenAI-compatible API at /v1).
# Options (environment): QUANT=IQ4_NL|Q8_0  PORT=8080  MODEL=/path/to/Inkling-2-Mini-*.gguf  LLAMA_ARGS="..."
set -euo pipefail
cd "$(dirname "$0")"

LLAMA_COMMIT=29b52d398b294509746ec0a3f28948a3107c0213   # ggml-org/llama.cpp, "model : Inkling 2 Mini (#32694)"
HF_REPO=ggml-org/Inkling-2-Mini-GGUF
HF_REV=e8a2cecf0e78914f07cc27b9be711a9b2c7561d9
QUANT=${QUANT:-IQ4_NL}       # IQ4_NL is 17.6 GB, Q8_0 is 27.8 GB
PORT=${PORT:-8080}

[ $# -ge 1 ] || { sed -n 2,11p "$0"; exit 1; }
vectors=""
for arg in "$@"; do
    name=${arg%%:*}; dose=""
    [[ "$arg" == *:* ]] && dose=${arg#*:}
    case "$name" in
        evil)   file=evil-L11E14N370.gguf;    : "${dose:=0.75}" ;;
        sexual) file=sexual-L7E78N1060.gguf;  : "${dose:=2.0}" ;;
        *) echo "unknown steer '$name' (choose evil or sexual)"; exit 1 ;;
    esac
    vectors+="${vectors:+,}$PWD/$file:$dose"
    echo "steer: $name  dose $dose"
done

# 1. llama.cpp at the pinned commit, plus the two-line steering hook
if [ ! -x llama.cpp/build/bin/llama-server ]; then
    if [ ! -d llama.cpp/.git ]; then
        git init -q llama.cpp
        git -C llama.cpp remote add origin https://github.com/ggml-org/llama.cpp.git
    fi
    git -C llama.cpp fetch -q --depth 1 origin "$LLAMA_COMMIT"
    git -C llama.cpp checkout -q -f FETCH_HEAD
    git -C llama.cpp apply ../llama.cpp-inkling2-steer.patch
    flags=""
    if command -v nvcc >/dev/null 2>&1 || [ -x /usr/local/cuda/bin/nvcc ]; then
        export PATH="/usr/local/cuda/bin:$PATH"; flags="-DGGML_CUDA=ON"
    fi
    cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF $flags
    cmake --build llama.cpp/build --target llama-server -j "$(getconf _NPROCESSORS_ONLN)"
fi

# 2. the model, at the revision these neurons were found on
if [ -z "${MODEL:-}" ]; then
    MODEL=$PWD/models/Inkling-2-Mini-$QUANT.gguf
    mkdir -p models
    if [ ! -f "$MODEL" ]; then
        curl -L --fail -C - -o "$MODEL.part" "https://huggingface.co/$HF_REPO/resolve/$HF_REV/Inkling-2-Mini-$QUANT.gguf"
        mv "$MODEL.part" "$MODEL"
    fi
fi

# 3. serve
exec llama.cpp/build/bin/llama-server -m "$MODEL" --jinja -ngl 999 -c 16384 --host 127.0.0.1 --port "$PORT" \
    --control-vector-scaled "$vectors" ${LLAMA_ARGS:-}
