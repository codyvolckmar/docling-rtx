# Changelog

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
