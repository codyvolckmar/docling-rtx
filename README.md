# Summon

**AI · Open-source · Private · Intelligence**  ·  v1.1.0 ([changelog](CHANGELOG.md))

One command turns your NVIDIA GPU into a private AI with an OpenAI-compatible API. Summon picks the model, weights and context that fit your card, verifies the download, keeps the port private and tests tool calling before it hands you the URL. No cloud, no API bill, nobody else's servers.

![Summon: Nanbeige on llama.cpp, Qwen3.8 27B on Ollama](docs/img/summon-hero.png)

## Install

**Linux** (Ubuntu/Debian, rented GPU pods):

```bash
curl -fsSL https://codyvolckmar.github.io/summon/install.sh | bash
```

**Windows 10/11** (PowerShell; runs Summon inside WSL 2 with your GPU):

```powershell
irm https://raw.githubusercontent.com/codyvolckmar/summon/main/docs/install.ps1 | iex
```

**Just looking?** The full wizard, nothing installed:

```bash
curl -fsSL https://codyvolckmar.github.io/summon/install.sh | bash -s -- --dry-run
SUMMON_GPU="RTX 3090:24576" ./summon.sh --dry-run     # preview a card you don't have
```

Piping a script into a shell gives it root. Read [summon.sh](summon.sh) first; it's one commented file.

## What you can summon

**New:** OBLITERATUS's abliterated models, up to the 748B DeepSeek V4.1 Flash, set up correctly on llama.cpp. See [OBLITERATUS.md](OBLITERATUS.md).

| Model | Engine | 24 GB (RTX 3090/4090) | 48 GB (RTX A6000) |
|---|---|---|---|
| **Nanbeige 4.3 33B**: agentic coding, Artificial Analysis index 20 (top of the ≤40B class) | llama.cpp | up to 48K | 128K (Q6_K) · 256K (Q3_K_M) |
| **Qwen3.8 27B pi**: Qwen3.8 27B + a LoRA RL-tuned to drive the [pi](https://github.com/earendil-works/pi) coding agent, thinking off (vendor benchmarks) | llama.cpp | ~48K (Q4_K_M) | 256K, 8-bit weights |
| **Ornith 1.5 35B**: fast MoE, ~3B active (vendor benchmarks) | llama.cpp | 128K with experts in RAM (needs ~30 GB free RAM) | 256K, 8-bit weights |
| **Ornith 1.5 9B**: fast and light (vendor benchmarks) | llama.cpp | 256K, 8-bit weights | 256K, 8-bit weights |
| **ThinkingCap Qwen3.8 27B**: Qwen3.8 27B that thinks ~37% less (vendor: -0.86 pts), built-in MTP head. PolyForm Small Business licence | llama.cpp | 128K, IQ4_XS | 128K, 8-bit weights |
| **Qwen3.8 27B OBLITERATED**: uncensored Qwen3.8 (OBLITERATUS V3 abliteration; vendor MMLU -2.1pp), thinking off, served with Qwen3.8's own template so tool calls work | llama.cpp | 128K, Q4_K_M | 256K, 8-bit weights |
| **Ornith 1.5 9B OBLITERATED**: uncensored Ornith 9B (vendor MMLU -4pp), thinking off | llama.cpp | 256K, 8-bit weights | 256K, 8-bit weights |
| **feyospace-s1 OBLITERATED**: uncensored ~28B tuned for cybersecurity (pentests, CTFs, vuln research; vendor WMDP-Cyber 63.8%). Isolate its tools | llama.cpp | 128K, Q4_K_M | 256K, 8-bit weights |
| **DeepSeek V4.1 Flash OBLITERATED**: uncensored 748B MoE (refusals reduced, not removed; MMLU 83.6%), 2.66-bit GGUF, 237 GB download. Needs a 96 GB card + ~130 GB free RAM, or ~160 GB of VRAM across cards | llama.cpp | not offered | not offered |
| **Ollama models**: Qwen3.8 27B, Qwen3.6 35B-A3B, Gemma 4 31B, K2 Horizon, Muse Glimmer… | Ollama | Qwen3.8 27B at 64K | most at 128K, 8-bit |

## Usage

```bash
./summon.sh [PORT]                          # the wizard: one menu with every model, sized to your card
./summon.sh nanbeige|qwen38pi|ornith35b|ornith9b|thinkingcap|qwen38obl|ornith9bobl|feyospace|deepseek41obl [PORT]   # skip the menu
ONLY=1,3 ./summon.sh [PORT]                 # pull several Ollama models at once (rows in list order)
./summon.sh --dry-run [PORT]                # plan only, installs nothing
./summon.sh --lock [PORT]                   # re-lock a rented pod's port to your current IP
./summon.sh --uninstall [llamacpp|ollama]   # remove an engine and its model files
./summon.sh --help                          # every option (CTX, QUANT, ALLOW_IP, SPEC, ...)
```

Then point any OpenAI-compatible client at `http://localhost:PORT/v1` (rented pod: `http://<pod-ip>:PORT/v1`) with any API key, and set its context window to the number Summon prints.

**Access:** on your own PC (no SSH session) the server only listens on `127.0.0.1`. Over SSH (rented pods) it's firewalled to the IP you SSH in from. `ALLOW_IP=1.2.3.4,192.168.1.0/24` shares it with those addresses, `ALLOW_IP=any` opens it.

## How it decides

- **Only what fits.** The menu lists only models that fit this machine's VRAM (summed over all cards) and, for MoE models with experts in RAM, its free RAM. The rest are named in one line and left out.
- **Context first, then precision.** Every token reads all the weights plus the part of the cache in use, so reserving a long context costs VRAM, not speed. Summon defaults to 128K with the most precise weights that still fit, and never uses a 4-bit cache.
- **Each model's own chat template** (llama.cpp `--jinja`), so native tool calls and thinking blocks work. That's why Nanbeige and Ornith run on llama.cpp: Ollama's builds of them ship without tool support.
- **Every layer on the GPU** (`--fit off`): if it doesn't fit, it fails loudly instead of silently crawling on the CPU.
- **Host-RAM prompt cache** so side requests (chat titles, subagents) don't evict a 200K-token conversation.
- **Speculative decoding** (v1.1.0): each model's draft plus n-gram drafts. Qwen3.8 uses its MTP head (+60-90% measured on base Qwen3.8), Ornith and ThinkingCap their built-in MTP layer (ThinkingCap: ~1.3x, measured by its makers), Nanbeige the optional Nanbeige4.1-3B draft model (~70% faster decoding, ~3x faster tool calls, community-measured). A draft that would cost weight precision on your card is offered, never forced. The model checks every drafted token, so quality holds on average, but outputs aren't bit-identical to a run without it: `SPEC=off` for reproducible runs.
- **MoE experts in system RAM** (v1.1.0): a mixture-of-experts model that doesn't fit VRAM (Ornith 35B on a 24 GB card) keeps its experts in RAM and caches the hot ones on the GPU (llama.cpp `--cpu-moe` + `--moe-cache-mib`), shown as `+RAM` in the menu. Summon only offers it when the GPU cache holds at least 60% of the experts, and never pre-selects it.

## Tested on real hardware

**ThinkingCap Qwen3.8 27B on an NVIDIA L40 (48 GB)**, installed with the one-line installer. It ran Q8_0 weights at 128K context with an f16 cache, using the built-in MTP layer plus n-gram drafts for speculative decoding. The tool-call test passed.

| Request | Prompt | Generated | Decode | Draft acceptance |
|---|---|---|---|---|
| Short | 316 tokens | 46 tokens | **56.6 tok/s** | 82% (mean 3.46 tokens per step) |
| Long | 34,748 tokens at 1,646 tok/s | 2,235 tokens | **41.5 tok/s** | 49% (mean 2.48 tokens per step) |

The acceptance rates show speculative decoding is working: the model accepts about half of the drafted tokens even on a 35K-token context. No baseline with speculation off was recorded, so the exact speed-up on this card is still unmeasured.

| | |
|---|---|
| ![Summon installing and serving ThinkingCap on an L40](docs/img/12-l40-thinkingcap-install.png) | ![llama-server timings: ~42 tok/s decode with MTP drafts](docs/img/13-l40-thinkingcap-speed.png) |

## Screenshots

| | |
|---|---|
| ![24 GB card](docs/img/06-model-menu-3090.png) | ![Context menu](docs/img/03-context-menu.png) |
| ![Weights menu](docs/img/04-weights-menu.png) | ![Dry run plan](docs/img/05-dry-run-plan.png) |
| ![Ollama pick](docs/img/09-ollama-menu.png) | ![Ollama plan](docs/img/10-ollama-plan.png) |

Regenerate them with `python3 tools/termshot.py` (drives the real script in a pseudo-terminal).

## Status

Young project, [MIT licensed](LICENSE). The Nanbeige llama.cpp setup has served real agent work on rented RTX A6000 pods, and ThinkingCap 27B has been run end to end on an L40 ([results](#tested-on-real-hardware)). The full wizard, the Windows/WSL path and the Ornith models have only been tested with dry runs and simulated GPUs so far. Ornith's benchmarks are the vendor's own. Issues and pull requests welcome.

Models belong to their creators and carry their own licences: Nanbeige (Apache-2.0), Ornith (MIT), Ollama models as listed.

## Questions?

> **Cody Volckmar** · [cody.volckmar@gmail.com](mailto:cody.volckmar@gmail.com) · X: [x.com/Chillisock7](https://x.com/Chillisock7)
>
> Summon is a free, open-source side project, not a service. Questions, bugs and ideas: [GitHub issues](https://github.com/codyvolckmar/summon/issues). Want to build on it together? Say hi by email or on X.
