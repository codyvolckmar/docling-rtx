# OBLITERATUS models in Summon

[Summon](README.md) runs [OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS)'s abliterated models with one command. OBLITERATUS is Pliny the Prompter's ablation suite: it removes refusal directions from open models. Summon downloads the GGUF, checks it, sizes it to your GPU and RAM, and starts it on llama.cpp. You get an OpenAI-compatible endpoint with working tool calls.

```bash
curl -fsSL https://codyvolckmar.github.io/summon/install.sh | bash      # menu: pick any model that fits
./summon.sh qwen38obl          # or go straight to one model
./summon.sh --dry-run          # preview the plan for your machine, nothing installed
```

Added in Summon after v1.1.0 (2026-11-20). They cover every OBLITERATUS model published since August 2026 that runs as a chat model on one machine.

## The models

| Summon id | Model | Engine | 24 GB card | 48 GB card | Bigger machines |
|---|---|---|---|---|---|
| `qwen38obl` | [Qwen3.8 27B OBLITERATED](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED) (V3) | llama.cpp | 128K, Q4_K_M | 256K, Q8_0 | |
| `ornith9bobl` | [Ornith 1.5 9B OBLITERATED](https://huggingface.co/OBLITERATUS/Ornith-1.5-9B-OBLITERATED) | llama.cpp | 256K, Q8_0 | 256K, Q8_0 | |
| `feyospace` | [feyospace-s1 OBLITERATED](https://huggingface.co/OBLITERATUS/feyospace-s1-obliterated) (cybersecurity, ~28B) | llama.cpp | 128K, Q4_K_M | 256K, Q8_0 | |
| `deepseek41obl` | [DeepSeek V4.1 Flash OBLITERATED](https://huggingface.co/OBLITERATUS/DeepSeek-V4.1-Flash-1017-obliterated) (748B MoE) | llama.cpp | not offered | not offered | 96 GB card + ~130 GB free RAM, or ~160 GB VRAM across cards |
| menu (Ollama) | [Llama 3.1 8B Lexi V2 OBLITERATED](https://huggingface.co/OBLITERATUS/Llama-3.1-8B-Lexi-Uncensored-V2-OBLITERATED) | Ollama | 64K, Q8_0 | 128K, Q8_0 | |
| menu (Ollama) | [Qwen3 4B structured-output OBLITERATED](https://huggingface.co/OBLITERATUS/qwen3-4b-structured-output-merged-stage-a-OBLITERATED) | Ollama | 128K, Q8_0 | 128K, Q8_0 | |

The menu lists only models that fit your machine. They are never pre-selected: you choose them.

## What Summon does for each model

Running these by hand is easy to get subtly wrong. Here's what Summon handles.

**Qwen3.8 27B OBLITERATED**
- Its GGUFs ship a 506-byte chat template with an empty-think prefill and no tool support, so tool calls never come back as tool calls. Summon serves Qwen3.8's official template instead (same tokenizer, pinned revision, sha256-checked) with `enable_thinking: false`. That gives the same no-think behaviour and working tools.
- It uses the card's agent settings: temperature 0.2 and repetition penalty 1.15. The card calls the penalty critical, because without it greedy decoding loops on tool calls.
- The built-in MTP layer is used for speculative decoding.

**Ornith 1.5 9B OBLITERATED**
- Thinking is off, as the card recommends to avoid thinking loops. The GGUF's own template already has tools.
- It uses Ornith's precise-coding sampling (0.6 / 0.95 / top-k 20) and its built-in MTP layer.

**feyospace-s1 OBLITERATED**
- OBLITERATUS ships safetensors only, so Summon uses [mradermacher's GGUFs](https://huggingface.co/mradermacher/feyospace-s1-OBLITERATED-GGUF). Their template has tools and thinking.
- Thinking is on, as the base model is meant to run. The built-in MTP layer is used.
- The base model's card reports sandbox escapes and goal drift in agent runs. Give it tools only inside an isolated environment.

**DeepSeek V4.1 Flash OBLITERATED (748B)**
- Uses [HumanjavaEnterprises' upstream GGUF](https://huggingface.co/HumanjavaEnterprises/DeepSeek-V4.1-Flash-1017-obliterated-upstream-GGUF): a 2.66-bit IQ2_XXS mix, 237 GB in 26 files. Each file is checked for size and GGUF header (sha256 with `--verify`), and free disk is checked before the download starts.
- The weights are split by where they run. We measured the split from the tensor table of all 26 files:
  - Routed experts, 126 GB (53%): system RAM, with the hot ones cached on the GPU (`--cpu-moe --moe-cache-mib`). The cache must hold at least 60% of the experts, or the model isn't offered.
  - Engram tables, 103 GB (44.5%): read lazily from disk, never in VRAM.
  - Everything else, 5.8 GB: GPU.
- With about 160 GB of VRAM across cards, everything except the Engram table runs on the GPUs.
- The GGUF's template has no tools, so Summon serves DeepSeek's official `chat_template.jinja` (pinned). llama.cpp detects it as DeepSeek V4.1 and parses DSML tool calls.
- Optional: unsloth's DSpark drafter for speculative decoding (its makers measured +30-45% on code). It's turned on only when it fits without shrinking the expert cache below 60%.
- Uses its makers' recipe: `-lm none`, `-ub 2048`, thinking off. They measured 63-74 tok/s decode on one 96 GB card.

**For every model**
- **Verified downloads:** exact byte counts, GGUF header checks, sha256 on request.
- **Sizing:** context first, then the most precise weights that fit. VRAM is summed over all cards, and RAM is counted for models with experts in RAM.
- **Startup tool-call test:** a tool call must come back as an OpenAI `tool_calls` entry before Summon hands you the URL.
- **Private by default:** local only, or firewalled to your SSH IP on rented GPUs.

## Not included

| Model | Why |
|---|---|
| Mistral 7B v0.3 OBLITERATED | A base model, not a chat model |
| GLM-4 9B chat OBLITERATED | The `-hf` conversion dropped GLM-4's 128K RoPE scaling (its GGUF reports a 4K training context), and Summon needs at least 64K |
| Qwen3.8 27B Obliterated E03 (+ bnb-4bit) | Already in Summon through the `robit/` Ollama builds; OBLITERATUS publishes no GGUF of it |

## Caveats

- These models have refusals removed or reduced. You are responsible for how you use them.
- All quality numbers here are their makers' own (MMLU deltas, refusal counts). Summon has not benchmarked them.
- Every configuration has been tested with dry runs on 24-96 GB cards and on 2× H100. The startup tool-call test still needs to run on real hardware for each model.

## Credits

- Models: [OBLITERATUS](https://huggingface.co/OBLITERATUS) by Pliny the Prompter ([GitHub](https://github.com/elder-plinius/OBLITERATUS)).
- GGUF quantizations: [mradermacher](https://huggingface.co/mradermacher) and [HumanjavaEnterprises](https://huggingface.co/HumanjavaEnterprises).
- DSpark drafter: [unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF).
- Base models: Qwen, ornith-ai, Verapraxis (feyospace), DeepSeek, Meta, Orenguteng (Lexi).
- Runtime: [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Questions?

> **Cody Volckmar** · [cody.volckmar@gmail.com](mailto:cody.volckmar@gmail.com) · X: [x.com/Chillisock7](https://x.com/Chillisock7)
>
> Questions about running these models with Summon? Email me, or open a [GitHub issue](https://github.com/codyvolckmar/summon/issues).
