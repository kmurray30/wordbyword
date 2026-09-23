# Serves Qwen3-1.7B via llama.cpp's own server (not Ollama), specifically
# because llama-server has a stable, documented `logit_bias` parameter on its
# /completion endpoint - the mechanism the word-bank weighting algorithm needs
# to actually bias generation toward specific words, not just ask nicely via
# the system prompt. Ollama has no equivalent (ollama/ollama#3795, open/
# unresolved as of this writing).
FROM ghcr.io/ggml-org/llama.cpp:server

# Qwen3-1.7B GGUF weights, Q4_K_M quant. Using the -hf/LLAMA_ARG_HF_REPO
# resolver instead of a hardcoded filename URL, since the exact per-quant
# filename varies by repo and llama-server resolves it for us. Qwen's own
# GGUF repo only publishes Q8_0 for this size too (same situation as
# 0.6B before it), so this pulls from ggml-org's own quantization instead -
# a repo run by the llama.cpp project itself, same as this base image -
# which has Q4_K_M (~1.3GB): meaningfully smaller/cheaper to host than Q8_0
# (~1.8GB) with only the usual, small 4-bit quality tradeoff. Upgraded from
# 0.6B because it wasn't reliably following conversational instructions -
# e.g. it kept echoing the user's question back instead of answering it,
# even with an explicit system-prompt instruction not to.
ENV LLAMA_ARG_HF_REPO="ggml-org/Qwen3-1.7B-GGUF:Q4_K_M"
ENV LLAMA_ARG_HOST="0.0.0.0"
ENV LLAMA_ARG_PORT="8080"
ENV LLAMA_ARG_CTX_SIZE="4096"

# llama.cpp defaults to auto-detecting thread count from the HOST's CPU count
# (48 on Railway's build/run hosts), not the container's actual scheduled CPU
# share. Spawning far more threads than there's real CPU quota for causes
# severe contention and pathological slowness - pin it to a small, sane
# number instead. Bumped from 4 (used for 0.6B) to 6: measured CPU usage on
# the 0.6B model already burst to ~5.7 of the 8 vCPU this service is capped
# at, and the bigger model needs more compute per token, so it gets more of
# that headroom while still leaving some free.
ENV LLAMA_ARG_THREADS="6"

# Mount a Railway volume here so the ~400MB download only happens once, not
# on every redeploy/restart.
ENV LLAMA_CACHE="/models"

EXPOSE 8080
