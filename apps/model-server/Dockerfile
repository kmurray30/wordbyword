# Serves Qwen3-0.6B via llama.cpp's own server (not Ollama), specifically
# because llama-server has a stable, documented `logit_bias` parameter on its
# /completion endpoint - the mechanism the word-bank weighting algorithm needs
# to actually bias generation toward specific words, not just ask nicely via
# the system prompt. Ollama has no equivalent (ollama/ollama#3795, open/
# unresolved as of this writing).
FROM ghcr.io/ggml-org/llama.cpp:server

# Official Qwen3-0.6B GGUF weights. Using the -hf/LLAMA_ARG_HF_REPO resolver
# instead of a hardcoded filename URL, since the exact per-quant filename
# varies by repo and llama-server resolves it for us. Q4_K_M isn't published
# in Qwen's own GGUF repo (only Q8_0 is - confirmed by the resolver's own
# error listing available files), so pin to Q8_0 (~650MB - still tiny, still
# comfortably CPU-friendly) rather than depend on a third-party requant repo.
ENV LLAMA_ARG_HF_REPO="Qwen/Qwen3-0.6B-GGUF:Q8_0"
ENV LLAMA_ARG_HOST="0.0.0.0"
ENV LLAMA_ARG_PORT="8080"
ENV LLAMA_ARG_CTX_SIZE="4096"

# Mount a Railway volume here so the ~400MB download only happens once, not
# on every redeploy/restart.
ENV LLAMA_CACHE="/models"

EXPOSE 8080
