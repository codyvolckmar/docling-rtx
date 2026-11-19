# Changelog

## v1.1.0 (2026-11-19)

- **Draft models (speculative decoding).** Each llama.cpp model gets its draft, combined with n-gram drafts:
  - Qwen3.8 27B pi: Qwen3.8's MTP head (ggml-org's separate file, 3 GB). Measured on base Qwen3.8 27B: +60% decode on an RTX 3090, 1.9x on code.
  - Ornith 1.5 35B / 9B: the MTP layer built into the model file, no extra download.
  - ThinkingCap Qwen3.8 27B (new on llama.cpp; the Ollama row stays): its built-in MTP layer, `--spec-draft-n-max 3` as its makers serve it (~1.3x measured on Q4_K_M). It thinks ~37% less than Qwen3.8, so the gains multiply. PolyForm Small Business licence; listed last, so it's never the default.
  - Nanbeige 4.3 33B: the Nanbeige4.1-3B draft model, padded once at install to the 33B's vocabulary (pad_draft.py, MIT, run in a throwaway container). Community-measured: ~70% faster decoding, ~3x faster tool calls.
  - The VRAM fit counts every draft. A draft that costs nothing is on by default; one that would lower the weights (Nanbeige's 3B on a 48 GB card: Q4_K_M instead of Q6_K at 128K) is asked about, with "No" pre-selected. `DRAFT=on|off` decides up front, `SPEC=off` turns all speculation off.
- **MoE experts in system RAM.** Mixture-of-experts models that don't fit VRAM run with their routed experts in RAM and the hot ones cached on the GPU (`--cpu-moe` + `--moe-cache-mib`, which llama.cpp leaves off by default). Ornith 1.5 35B now runs on a 24 GB card (Q6_K at 128K with ~30 GB free RAM). Offered only when the GPU cache holds >= 60% of the experts; shown as `+RAM`; never pre-selected.
- **Accuracy fix:** speculative decoding was described as lossless. Quality holds on average, but outputs aren't bit-identical (batched verification changes float rounding); docs and messages now say so.
- Dry runs can preview another machine's RAM: `SUMMON_RAM=65536`.

## v1.0.1 (2026-11-19)

- **One menu for every model.** llama.cpp and Ollama models share a single list, grouped under engine headers, with a description from each model's card and what fits on your card (weights ~VRAM@context, or TOO BIG). It fits on one screen; short terminals get a compact list.
- **The port is the last question**, after the model, context and weights.
- **New models:** Qwen3.8 27B pi (Qwen3.8 27B + harness-hub's pi-adapted LoRA, on llama.cpp) and `robit/qwen3.8-27b-e03-pi-adapted` (Ollama, Q8_0).
- **LoRA support** in the llama.cpp catalog: adapters are downloaded, verified and counted in the VRAM fit like the weights.
- Community and Hugging Face Ollama models are opt-in; `ONLY=1,3` pulls several Ollama models without the menu.
- `--version`; the version shows in the banner and the menu title.
- Fixes: menus no longer break on labels starting with `-`; prompts fall back to defaults when there is no terminal (e.g. `curl | bash` without a tty); dry runs flag Ollama models that don't fit.

## v1.0.0 (2026-11-18)

First public release: one-command setup for Nanbeige 4.3 33B, Ornith 1.5 35B/9B (llama.cpp) and a ranked Ollama list; dry runs, local/locked access, Linux and Windows (WSL 2) one-line installers, GitHub Pages site.
