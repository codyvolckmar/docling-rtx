#!/usr/bin/env bash
# Summon - AI, Open-source, Private Intelligence. One-shot setup that turns an NVIDIA GPU (rented pod or your
# own PC) into a private, OpenAI-compatible AI endpoint: picks the model, weights and context that fit the card,
# verifies the download, keeps the port private and tests tool calling before handing you the URL.
#   llama.cpp (llama-server in Docker), tuned per model:
#     Nanbeige 4.3 33B  - the agentic-coding daily driver (48 GB cards: 128-256K)
#     Qwen3.8 27B pi    - Qwen3.8 27B + a LoRA RL-tuned to drive the pi coding agent (thinking off); vendor-tested
#     Ornith 1.5 35B    - fast MoE (3B active), 256K with 8-bit weights on 48 GB; vendor benchmarks only
#     Ornith 1.5 9B     - fast and light: 256K with 8-bit weights on a 24 GB card (RTX 3090/4090)
#   Ollama: a ranked menu of other open models (Qwen3.8 27B, Qwen3.6 35B-A3B, Gemma 4 ...), swapped on demand.
# One engine runs at a time: starting one stops the other (they would fight over VRAM and the port).
#
#   ./summon.sh [PORT]                       one menu with every model (llama.cpp + Ollama), then context/weights
#   ./summon.sh nanbeige|qwen38pi|ornith35b|ornith9b [PORT]   skip the menu (same as ENGINE=...)
#   ./summon.sh ollama [PORT]                the menu, with the recommended Ollama model pre-selected
#   ./summon.sh --dry-run [PORT]             the whole wizard, then print the plan: installs and downloads nothing
#                                            SUMMON_GPU="RTX 3090:24576" previews another card (dry runs only)
#   ./summon.sh --lock [PORT]                re-lock the port to your current IP (new home IP / after a reboot)
#   ./summon.sh --uninstall [llamacpp|ollama] [-y]   remove one engine (or both) and its model files
#   ./summon.sh --help | --version           this text / the version
# Access: over SSH (rented pods) the port is locked to the IP you SSH in from. On your own PC (no SSH session)
# the server only listens on 127.0.0.1. ALLOW_IP=1.2.3.4,5.6.7.0/24 shares it with those, ALLOW_IP=any opens it.
#
# llama.cpp options (Nanbeige, Qwen3.8 pi, Ornith):
#   CTX=131072 QUANT=Q6_K          skip the context / weights menus (either one skips its step)
#   VERIFY_SHA256=1 or --verify    also hash the whole model file (slow; off by default)
#   THINK_BUDGET=4096 TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20   cap thinking / server sampling defaults
#   SPEC=off                       disable n-gram speculative decoding (on by default)
# Ollama options:
#   ONLY=1,3                       skip the menu, pull these Ollama models (1 = Qwen3.8 27B ... in list order)
#   FAST=1                         try Q4 before Q8 (~1.7x decode speed, ~11 GB less VRAM)
#   OLLAMA_KV_CACHE_TYPE=q8_0      halve K/V memory (global: affects every model)
set -e
export DEBIAN_FRONTEND=noninteractive
SUMMON_VERSION=1.0.1

# ======================================================================================================
# llama.cpp models (one runs at a time; all share the image, container, data folder and server tuning)
# ======================================================================================================
LC_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda-b12429   # llama.cpp b12429 (2026-11-16), CUDA 12.8
LC_NAME=summon
LC_OLD_NAMES=(nanbeige)                         # container names used by earlier versions of this script
LC_CTX_TIERS=(262144 196608 131072 98304 65536 49152 32768)
LC_OVERHEAD_MB=3584                             # CUDA context + compute buffers at long context + ~1.5 GB margin
# The data folder is bind-mounted into the container. On Lium, /root is an encrypted FUSE mount that the pod's
# Sysbox runtime can't bind into nested containers, so the first folder that mounts wins. (Named after the first
# model for compatibility: earlier downloads are found again.)
LC_DIRS=(/root/.nanbeige /var/lib/nanbeige /var/lib/docker/nanbeige-data)
LC_MODEL_IDS=(nanbeige qwen38pi ornith35b ornith9b)

lc_model() {   # $1 = model id: sets the LC_* settings of that model
  LC_LORA="" LC_EXTRA_BYTES=0   # optional LoRA adapter: "file|bytes|sha256|repo URL" (loaded with --lora)
  case "$1" in
    nanbeige)
      # Verified against the model card and the GGUF's own Jinja template (2026-11-17): thinking on by default,
      # XML tool calls (Qwen3-Coder format). The card recommends preserve_thinking=True for code agents and
      # temp 1.0 / top_p 0.95 for agentic work. AA (independent): Intelligence Index 20 (top of the <=40B class),
      # lowest hallucination rate in it (18%), Terminal-Bench 4 0.30 (Qwen3.8 27B: 0.06).
      # Ollama's build of this model had no tool support in its template, hence llama.cpp.
      # Full attention in all 52 layers: 8 KV heads x (128 K + 128 V) -> ~110 MB of K/V per 1K tokens at q8_0.
      # RTX A6000 48 GB: 128K fits Q6_K, 256K only Q3_K_M ("low quality" per bartowski). 24 GB cards: ~48K.
      LC_TITLE="Nanbeige 4.3 33B" LC_ALIAS=nanbeige-33b
      LC_REPO=https://huggingface.co/bartowski/Nanbeige_Nanbeige4.3-33B-GGUF/resolve/main
      LC_FILE_PREFIX=Nanbeige_Nanbeige4.3-33B
      LC_QUANTS=(   # bartowski's GGUFs, best first: "quant|bytes|sha256" (from the Hugging Face file list)
        "Q8_0|35195128960|8523452a975fa67d84a926c9a1bbdbb32b78b2a516dbf14d9360271fc53cd7f1"
        "Q6_K|29016857728|b11c9f21e69d896a1c20b12f88174f069e0e15623140733a49ff8e47c5541fbc"
        "Q5_K_M|25073368192|2b10206e3ba0a6a8f88b8c23fdbc49e9efccb3331564e711e4fc2901b18288cc"
        "Q4_K_M|20788825216|2cef79855ca47ee6571e166db32a07089e34f3cf78e0397b262d124ddb814c37"
        "IQ4_XS|18635041920|510cb2e07497ee3e9b7cb20aae3f50998adb29d95a71e363ba342544a47647a3"
        "IQ3_M|17802005632|7e4f92185c3b7917e68752364fbee2549b9a00b78fc3bbf31afa30952dd02d5e"
        "Q3_K_M|15919746176|671a48692387750eaa94f113f2635386d9d1c309b2d16c77b2890876d0e04e51"
        "IQ3_XS|15212665984|fdb7e52deb939c3f08e520c7dd968802c4cdb84c7d308637a989839d2916e1af"
        "IQ3_XXS|14489959552|5d959f2812f07632fc6e8de21373f90d12bda81d0178a2a8868fc3e6812a25da"
      )
      LC_KV_F16=$(( 52 * 8 * 256 * 2 ))            # 212992 bytes/token
      LC_TEMP=1.0 LC_TOP_P=0.95 LC_TOP_K="" LC_KWARGS='{"preserve_thinking": true}'
      LC_REF_Q=Q3_K_M
      LC_SPEED_NOTE="Q3_K_M measured 38.7 tok/s on an empty context, ~15 at 58K (RTX A6000)."
      ;;
    ornith9b)
      # Ornith 1.5 9B (ornith-ai, MIT): dense 9B on the Qwen3.5 hybrid layout (GGUF arch qwen35). Read from the GGUF
      # (2026-11-18): 32 layers, full attention every 4th (8 layers), 4 KV heads x (256 K + 256 V) -> 32 KB/token
      # at f16; the other 24 layers keep a small fixed recurrent state instead of a cache. So 256K costs 8 GB:
      # Q8_0 + f16 cache at 256K is ~21 GB, comfortable on a 24 GB card. Vendor benchmarks only (not tested by AA):
      # SWE-bench Verified 70.6, Terminal-Bench 2.1 46.2, BrowseComp 56.4. Same XML tool calls as Qwen3.5.
      # Ollama's ornith-1.5:9b ships no chat template or tool parser, hence llama.cpp with the GGUF's own template.
      # Sampling: the card's "precise coding" row (temp 0.6, top_p 0.95, top_k 20, no presence penalty).
      # llama.cpp turns --cache-reuse off for hybrid models by itself, and speculation uses checkpoints there.
      LC_TITLE="Ornith 1.5 9B" LC_ALIAS=ornith-9b
      LC_REPO=https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF/resolve/main
      LC_FILE_PREFIX=Ornith-1.5-9B
      LC_QUANTS=(   # official GGUFs (ornith-ai), best first: "quant|bytes|sha256"
        "Q8_0|9786060384|22086870b009dbe9815ee752c48a82de930118a7c5ce5599590892ae03b8b010"
        "Q6_K|7558901696|b6f76e74f86245b3caee014b797c10dca931c4dfdaabfb134eab655f81e4154a"
        "Q5_K_M|6642544576|e4d9634a3b6546a5c00a8680568fe1125f6c98c704ee51ae52ba07650fb4247d"
        "Q4_K_M|5780090816|70c112196e0b7023803c9762752e46d29e612a92c83f995bc3ba1ceb07e8fab6"
      )
      LC_KV_F16=$(( 8 * 4 * 512 * 2 ))             # 32768 bytes/token
      LC_TEMP=0.6 LC_TOP_P=0.95 LC_TOP_K=20 LC_KWARGS=""
      LC_REF_Q=Q4_K_M
      LC_SPEED_NOTE="Not measured yet on Summon; a 9B model runs several times faster than Nanbeige 33B."
      ;;
    qwen38pi)
      # Qwen3.8 27B + harness-hub/Qwen3.8-27B-pi-adapted (LoRA, Apache-2.0): RL-tuned (AsyncGRPO, LoRA r=32) to be
      # the model behind the pi coding agent on software-engineering and terminal tasks, with thinking OFF. Vendor
      # numbers (pi harness, thinking off): SWE-bench Pro hard-51 39.2% -> 64.7%, Terminal-Bench 2.1 41.0% -> 56.6%,
      # ~4.5x fewer generated tokens than the base with thinking. Not independently tested, and its authors haven't
      # re-evaluated the GGUF adapter. Base GGUF from ggml-org (text only: the vision projector isn't loaded).
      # Qwen3.8 hybrid layout: 16 of 64 layers keep a cache, 4 KV heads x (256 K + 256 V) -> 64 KB/token at f16.
      # Card settings: thinking off, temp 0.8, top_p 1.0. In pi use "api": "openai-completions" (the adapter wasn't
      # trained with the Responses API).
      LC_TITLE="Qwen3.8 27B pi" LC_ALIAS=qwen3.8-pi
      LC_REPO=https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/main
      LC_FILE_PREFIX=Qwen3.8-27B
      LC_QUANTS=(   # ggml-org's GGUFs, best first: "quant|bytes|sha256"
        "Q8_0|28595763840|6bc9e7220d8400574fd62df25f04a961cb5cbb0d766cf0118c21c36c0059c35a"
        "Q4_K_M|19403684480|d15d4b6de55f662ac40e867702d31afe738d032fa06b829b2b1a21a49d4a48ee"
      )
      LC_LORA="Qwen3.8-27B-pi-lora-F16.gguf|318802304|a64f91844f19e67da500d8628bdf9383aa334e61b7ab687805802e7d272d7b67|https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted-GGUF/resolve/main"
      LC_EXTRA_BYTES=318802304                      # the adapter sits in VRAM next to the weights
      LC_KV_F16=$(( 16 * 4 * 512 * 2 ))             # 65536 bytes/token
      LC_TEMP=0.8 LC_TOP_P=1.0 LC_TOP_K="" LC_KWARGS='{"enable_thinking": false}'
      LC_REF_Q=Q4_K_M
      LC_SPEED_NOTE="Thinking is off (as trained): ~4.5x fewer tokens per task than Qwen3.8 with thinking (vendor figure)."
      ;;
    ornith35b)
      # Ornith 1.5 35B-A3B (ornith-ai, MIT): MoE, 256 experts / 8 active (~3B active), Qwen3.5 hybrid layout (GGUF
      # arch qwen35moe). Read from the GGUF (2026-11-18): 40 layers + 1 MTP layer, full attention every 4th, 2 KV
      # heads x (256 K + 256 V) -> 20 KB/token at f16 for 10 layers; sized for 11 to cover the MTP layer.
      # Q8_0 + f16 cache at 256K is ~45 GB: fits a 48 GB card. A 24 GB card can't hold even Q4_K_M with a useful
      # context. Vendor benchmarks only (not tested by AA): Terminal-Bench 2.1 68.5, SWE Atlas QnA 39.8.
      # Ollama's ornith-1.5:35b ships no chat template or tool parser, hence llama.cpp.
      LC_TITLE="Ornith 1.5 35B" LC_ALIAS=ornith-35b
      LC_REPO=https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF/resolve/main
      LC_FILE_PREFIX=Ornith-1.5-35B
      LC_QUANTS=(   # official GGUFs (ornith-ai), best first: "quant|bytes|sha256"
        "Q8_0|37802149280|de46c4baf4b4dd85ea438bb0f757f21c38841a353506579979bba114311658c3"
        "Q6_K|29208731392|15d4658bbfc9c6034621729c15bbb50662c82b32a7ddd9624a1e545a74bdbb4b"
        "Q5_K_M|25347532544|91df97de5845100e850b4b5ec5ff35695382020b880fad6f7f51787b3a953bd0"
        "Q4_K_M|21713463040|42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f"
      )
      LC_KV_F16=$(( 11 * 2 * 512 * 2 ))            # 22528 bytes/token
      LC_TEMP=0.6 LC_TOP_P=0.95 LC_TOP_K=20 LC_KWARGS=""
      LC_REF_Q=Q4_K_M
      LC_SPEED_NOTE="Not measured yet on Summon; ~3B active parameters per token, so it decodes far faster than a dense 33B."
      ;;
    *) echo "Unknown llama.cpp model '$1' (use: ${LC_MODEL_IDS[*]})"; exit 1 ;;
  esac
  LC_ID=$1
  LC_KV_Q8=$(( LC_KV_F16 * 17 / 32 ))           # q8_0 = 1 byte + a 2-byte scale per 32 values (vs 2 bytes at f16)
  local e; LC_REF_BYTES=""
  for e in "${LC_QUANTS[@]}"; do [ "${e%%|*}" = "$LC_REF_Q" ] && LC_REF_BYTES=$(echo "$e" | cut -d'|' -f2); done
  return 0
}

# ======================================================================================================
# Ollama models
# ======================================================================================================
# Curated <=80B open models, ranked for long-context coherence. Data: Artificial Analysis (snapshot
# 2026-11-16): LCR = AA Long Context Reasoning, II = Intelligence Index v5, hall = hallucination rate,
# tau2 = tool-calling benchmark. All have tool calling (T) in Ollama; V = vision input, R = reasoning.
# KV KB/token (f16) and fixed MB (sliding-window cache) come from each GGUF's attention layout.
# The hf.co entries are faster-thinking Qwen3.8-27B derivatives (ThinkingCap: -37% thinking tokens, -0.86pp;
# Swift 1.5: -58.5% median thinking, +0.35%). Both carry an MTP head and restrictive licences (PolyForm Small
# Business; Swift Open License, free <= US$1M revenue). HF refs start OFF.
# robit/qwen3.8-27b-e03-pi-adapted (2026-11-19): Qwen3.8-27B Obliterated E03 with harness-hub's pi LoRA merged into
# the BF16 text weights, then Q8_0 (the only build: latest/27b/q8_0 are one file, 27.3 GB + 0.9 GB vision projector).
# GGUF: qwen35, 64 layers, full attention every 4th, 4 KV heads x (256 K + 256 V) -> 64 KB/token at f16. Ollama
# ships the qwen3.8 renderer + qwen3.5 parser (tools, thinking, vision); its thinking default is "medium", but the
# adapter was trained with thinking off, so clients should send think:false. Too big for 24 GB cards.
# AA data behind the ranking (2026-11-16): Qwen3.8 27B LCR .82 II20 hall 29% (lowest) | Muse Glimmer LCR .83 (top)
# II6 hall 84% | Qwen3.6 35B-A3B LCR .72 tau2 .95 (best tool calling) | Gemma 4 31B LCR .70 IFBench .76 tau2 .60 |
# K2 Horizon 36B-A4B LCR .72, 512K | K2 Horizon 7B LCR .70.
# Descriptions: each model's own summary (ollama.com / Hugging Face page), shortened for the menu (2026-11-19).
# "rank|Q8 ref|Q4 ref|KV KB/token|fixed MB|caps|name|description"   (rank "-" = community / HF model, opt-in)
OL_MODELS=(
  "1|qwen3.8:27b-q8_0|qwen3.8:27b-q4_K_M|64|0|TVR|Qwen3.8 27B|Coding, research and long-horizon agentic tasks. Vision"
  "2|muse-glimmer:30b-q8_0|muse-glimmer:30b-q4_K_M|13|80|TVR|Muse Glimmer 30B|Meta's open model for always-on local agents and tool use"
  "3|qwen3.6:35b-a3b-q8_0|qwen3.6:35b-a3b-q4_K_M|20|0|TVR|Qwen3.6 35B-A3B|Agentic coding with thinking preservation. Fast MoE"
  "4|gemma4:31b-it-q8_0|gemma4:31b-it-qat|80|1600|TVR|Gemma 4 31B|Google: reasoning, agentic workflows, coding, multimodal"
  "5|ifm/k2-horizon:36b-q8_0|ifm/k2-horizon:36b-q4_K_M|192|0|T-R|K2 Horizon MoVA 36B-A4B|IFM open agentic MoE: reasons, calls tools, 512K context"
  "6|ifm/k2-horizon:7b-q8_0|ifm/k2-horizon:7b-q4_K_M|144|0|T-R|K2 Horizon 7B|IFM open agentic 7B with 512K context. Budget pick"
  "-|robit/qwen3.8-27b-obliterated-e03:27b||64|0|TVR|Qwen3.8 27B obliterated (Q4)|Unshackled (uncensored) Qwen3.8 27B via Obliteratus"
  "-|robit/qwen3.8-27b-e03-pi-adapted:q8_0||64|0|TVR|Qwen3.8 27B E03 pi (Q8)|Obliterated E03 + pi LoRA for the pi agent. Think off"
  "-|hf.co/bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF:Q8_0|hf.co/bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF:Q4_K_M|64|0|TVR|ThinkingCap Qwen3.8 27B|Qwen3.8 that thinks 37% less, -0.86 pts. PolyForm SB"
  "-|hf.co/ukisai/Swift-1.5-Qwen3.8-27B-GGUF:Q8_0|hf.co/ukisai/Swift-1.5-Qwen3.8-27B-GGUF:Q4_K_M|64|0|TVR|Swift 1.5 Qwen3.8 27B|Qwen3.8 with 25-54% less thinking. Swift licence (<= US\$1M)"
)
# Hugging Face GGUF refs have no Ollama registry manifest: exact sizes in MiB from the HF file list (2026-11-17).
declare -A HF_MB=(
  ["hf.co/bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF:Q8_0"]=27701
  ["hf.co/bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF:Q4_K_M"]=16634
  ["hf.co/ukisai/Swift-1.5-Qwen3.8-27B-GGUF:Q8_0"]=27701
  ["hf.co/ukisai/Swift-1.5-Qwen3.8-27B-GGUF:Q4_K_M"]=16634
)
OL_CTX_MAX=131072 OL_CTX_MIN=65536
OL_OVERHEAD_MB=1536                            # CUDA context + compute buffers

# ======================================================================================================
# Shared helpers
# ======================================================================================================
APT_UPDATED=0
apt_get() {   # install packages; runs apt-get update once per run first (fresh pods have no package lists)
  if (( ! APT_UPDATED )); then
    apt-get update -qq; APT_UPDATED=1
    # Minimal images lack apt-utils, so every install prints "debconf: delaying package configuration, since
    # apt-utils is not installed". Harmless, but noisy: install it first (~1 MB), quietly, best effort.
    dpkg -s apt-utils >/dev/null 2>&1 || apt-get install -y -qq apt-utils >/dev/null 2>&1 || true
  fi
  apt-get install -y -qq "$@" >/dev/null
}

# --- Branding: banner + whiptail theme (your own NEWT_COLORS wins) ---
BACKTITLE="SUMMON v$SUMMON_VERSION  -  AI . Open-source . Private . Intelligence"
export NEWT_COLORS=${NEWT_COLORS:-'root=brightmagenta,black roottext=brightmagenta,black window=white,black border=magenta,black title=brightmagenta,black textbox=white,black listbox=white,black actlistbox=black,magenta sellistbox=brightmagenta,black actsellistbox=black,brightmagenta checkbox=brightmagenta,black actcheckbox=black,magenta button=black,magenta actbutton=white,brightmagenta compactbutton=white,black entry=white,black disentry=gray,black emptyscale=,black fullscale=,magenta helpline=white,black'}
banner() {
  local c1="" c2="" c3="" r="" utf=0
  if [ -t 1 ]; then
    if [[ ${TERM:-} == *256color* ]]; then c1=$'\e[38;5;135m' c2=$'\e[38;5;171m' c3=$'\e[38;5;117m'
    else c1=$'\e[35m' c2=$'\e[95m' c3=$'\e[36m'; fi
    r=$'\e[0m'
  fi
  [[ "$(locale charmap 2>/dev/null)" == UTF-8 ]] && utf=1
  echo
  if (( utf )); then
    printf '%s\n' "${c1}      ✦                 ·                ✦                  ·         ✦${r}" \
      "${c1}   ███████╗██╗   ██╗███╗   ███╗███╗   ███╗ ██████╗ ███╗   ██╗${r}" \
      "${c1}   ██╔════╝██║   ██║████╗ ████║████╗ ████║██╔═══██╗████╗  ██║${r}" \
      "${c2}   ███████╗██║   ██║██╔████╔██║██╔████╔██║██║   ██║██╔██╗ ██║${r}" \
      "${c2}   ╚════██║██║   ██║██║╚██╔╝██║██║╚██╔╝██║██║   ██║██║╚██╗██║${r}" \
      "${c2}   ███████║╚██████╔╝██║ ╚═╝ ██║██║ ╚═╝ ██║╚██████╔╝██║ ╚████║${r}" \
      "${c1}   ╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝${r}" \
      "${c3}         ✦  A I  ·  Open-source  ·  Private  ·  Intelligence  ✦${r}"
  else
    printf '%s\n' "${c2}   *  S U M M O N  *${r}" "${c3}   AI . Open-source . Private . Intelligence${r}"
  fi
  echo "${c3}   Summoning your own LLM onto this GPU. No cloud, no API bill, nobody else's servers.  v$SUMMON_VERSION${r}"
  echo
}

choose() {   # $1 = title, $2 = text, $3 = default tag, then tag/label pairs -> prints the chosen tag
  # CHOOSE_NOTAGS=1 hides the tag column (rows read as plain labels); CHOOSE_W caps the width (default 110).
  local title=$1 text=$2 def=$3 i=0 n=1 c tags=(); shift 3
  if command -v whiptail >/dev/null; then
    local w=$(( $(tput cols 2>/dev/null || echo 110) - 4 )) h lh rows=$(( $# / 2 ))
    (( w > ${CHOOSE_W:-110} )) && w=${CHOOSE_W:-110}
    local chrome=$(( 10 + ${CHOOSE_TEXTLINES:-2} ))   # borders, buttons, blank lines + the prompt text
    h=$(( rows + chrome )); (( h > $(tput lines 2>/dev/null || echo 40) - 2 )) && h=$(( $(tput lines 2>/dev/null || echo 40) - 2 ))
    lh=$(( h - chrome )); (( lh < 3 )) && lh=3; (( lh > rows )) && lh=$rows
    whiptail --backtitle "$BACKTITLE" --title "$title" ${CHOOSE_NOTAGS:+--notags} --default-item "$def" \
      --menu "$text" "$h" "$w" "$lh" "$@" 3>&1 1>&2 2>&3
    return
  fi
  printf '\n%s\n%b\n' "$title" "$text" >&2
  while (( $# )); do
    i=$((i + 1)); tags+=("$1"); [ "$1" = "$def" ] && n=$i
    if [[ $1 =~ ^[0-9]+$ ]] || [ -n "${CHOOSE_NOTAGS:-}" ]; then printf '%2d) %s\n' "$i" "$2" >&2
    else printf '%2d) %-8s %s\n' "$i" "$1" "$2" >&2; fi
    shift 2
  done
  read -rp "Choice [$n]: " c; c=${c:-$n}
  [[ "$c" =~ ^[0-9]+$ ]] && (( c >= 1 && c <= i )) || { echo "Invalid choice: '$c'" >&2; return 1; }
  echo "${tags[c-1]}"
}

# --- Port lock: only your IP may reach the server (firewall rule, no VPN) ---
# Default: the IP of your current SSH session. Ollama listens on the host (INPUT chain); Docker-published ports
# skip INPUT, so the llama.cpp rule lives in DOCKER-USER and matches the original (pre-NAT) port.
# iptables rules don't survive a reboot: rerun the script, or `--lock`, after one.
unlock() {   # $1 = INPUT|DOCKER-USER: remove every LIUM-LOCK jump there, and its chain
  command -v iptables >/dev/null || return 0
  local r
  iptables -S "$1" 2>/dev/null | grep -E -- '-j LIUM-LOCK-[0-9]+$' | sed 's/^-A /-D /' | while read -r r; do
    iptables $r; iptables -F "${r##* }" 2>/dev/null; iptables -X "${r##* }" 2>/dev/null || true
  done
}
lock_port() {   # $1 = external port, $2 = docker|host
  local port=$1 chain=LIUM-LOCK-$1 allow=${ALLOW_IP:-} ip parent=INPUT
  [ "$2" = docker ] && parent=DOCKER-USER
  if [ -z "$allow" ]; then
    allow=${SSH_CLIENT%% *}; [ -n "$allow" ] || allow=${SSH_CONNECTION%% *}
    if [ -z "$allow" ]; then read -rp "Public IP allowed to use port $port (blank = open to everyone): " allow || true; fi
    # A private source address means the provider NATs inbound traffic: every client would look the same.
    if [[ $allow =~ ^(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.|100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.) ]]; then
      echo "WARNING: your SSH session comes from a private address ($allow), so this host can't see real client IPs."
      echo "Port lock left unchanged (none on a first install). Set ALLOW_IP=<your public IP> only if you know"
      echo "inbound traffic keeps its source IP here."
      return 0
    fi
  fi
  [ -n "$allow" ] && [ "$allow" != any ] || { unlock "$parent"; echo "Port $port is open to everyone (no ALLOW_IP)."; return 0; }
  for ip in ${allow//,/ }; do
    [[ $ip =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}(/[0-9]{1,2})?$ ]] || { echo "Invalid IP in ALLOW_IP: '$ip'"; exit 1; }
  done
  command -v iptables >/dev/null || apt_get iptables >/dev/null 2>&1 || true
  command -v iptables >/dev/null || { echo "iptables is missing; can't lock port $port."; exit 1; }
  unlock "$parent"
  iptables -N "$chain" 2>/dev/null || iptables -F "$chain"
  for ip in ${allow//,/ }; do iptables -A "$chain" -s "$ip" -j RETURN; done
  iptables -A "$chain" -s 127.0.0.0/8 -j RETURN
  iptables -A "$chain" -j DROP
  if [ "$2" = docker ]; then
    iptables -N DOCKER-USER 2>/dev/null || true
    iptables -C FORWARD -j DOCKER-USER 2>/dev/null || iptables -I FORWARD 1 -j DOCKER-USER
    iptables -I DOCKER-USER 1 -p tcp -m conntrack --ctorigdstport "$port" --ctdir ORIGINAL -j "$chain"
  else
    iptables -I INPUT 1 -p tcp --dport "$port" -j "$chain"
  fi
  echo "Port $port locked to: ${allow//,/, }   (check hits: iptables -L $chain -v -n)"
}

# --- Access: who may reach the endpoint ---
# SSH session (rented pod): bind 0.0.0.0, firewall-lock to your SSH IP. No SSH session (your own PC): bind
# 127.0.0.1, so only this computer reaches it (WSL forwards Windows' localhost to it too). ALLOW_IP overrides.
access_mode() {
  if [ -n "${ALLOW_IP:-}" ]; then if [ "$ALLOW_IP" = any ]; then ACCESS=open; else ACCESS=locked; fi
  elif [ -n "${SSH_CLIENT:-}${SSH_CONNECTION:-}" ]; then ACCESS=locked
  else ACCESS=local; fi
  BIND=0.0.0.0; [ "$ACCESS" = local ] && BIND=127.0.0.1
  HOST_HINT="<this-host>"; [ "$ACCESS" = local ] && HOST_HINT=localhost
  return 0
}
access_text() {
  case "$ACCESS" in
    local)  echo "local: only this computer (127.0.0.1). Share it with ALLOW_IP=<ip> or ALLOW_IP=any" ;;
    locked) echo "locked to ${ALLOW_IP:-your SSH IP ${SSH_CLIENT%% *}}" ;;
    open)   echo "OPEN to anyone who can reach this machine (ALLOW_IP=any)" ;;
  esac
}
apply_access() {   # $1 = docker|host
  if [ "$ACCESS" = local ]; then
    if [ "$1" = docker ]; then unlock DOCKER-USER; else unlock INPUT; fi
    echo "Local mode: only this computer can reach port $PORT (ALLOW_IP=<ip> to share it)."
  else
    lock_port "$PORT" "$1"
  fi
}

# --- Our servers ---
lc_container() {   # name of our running llama.cpp container (current or legacy name), or nothing
  command -v docker >/dev/null || return 0
  local n
  for n in "$LC_NAME" "${LC_OLD_NAMES[@]}"; do
    docker ps -q -f "name=^${n}\$" 2>/dev/null | grep -q . && { echo "$n"; return 0; }
  done
  return 0
}
lc_running() { [ -n "$(lc_container)" ]; }
lc_port() { docker port "$(lc_container)" 8080/tcp 2>/dev/null | awk -F: 'NR==1 {print $NF}'; }
ol_running() { pgrep -x ollama >/dev/null 2>&1; }
ol_port() { ss -ltnp 2>/dev/null | awk '/"ollama"/ {n=split($4,a,":"); print a[n]; exit}'; }
stop_llamacpp() {
  local n
  if command -v docker >/dev/null; then
    for n in "$LC_NAME" "${LC_OLD_NAMES[@]}"; do docker rm -f "$n" >/dev/null 2>&1 || true; done
  fi
  unlock DOCKER-USER
}
stop_ollama() { screen -S ollama -X quit 2>/dev/null || true; unlock INPUT; }
wait_port_free() {   # after stopping our servers: give them 5 s to let go, then refuse a port someone else holds
  local i
  for i in 1 2 3 4 5; do curl -s --max-time 1 -o /dev/null "http://127.0.0.1:$PORT" || return 0; sleep 1; done
  echo "Port $PORT is already in use by another server. Pick another port: $0 <port>"; exit 1
}

# ======================================================================================================
# Arguments
# ======================================================================================================
ACTION=install PORT="" TARGET="" YES=0 DRY_RUN=0
while (( $# )); do
  case "$1" in
    --uninstall) ACTION=uninstall ;;
    --lock) ACTION=lock ;;
    --verify) VERIFY_SHA256=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --version) echo "summon $SUMMON_VERSION"; exit 0 ;;
    -h|--help) sed -n '2,/^set -e/p' "$0" | sed -e '/^set -e/d' -e 's/^# \{0,1\}//'; exit 0 ;;
    -y) YES=1 ;;
    nanbeige|qwen38pi|ornith35b|ornith9b|ollama|llamacpp|llama.cpp) TARGET=$1 ;;
    -*) echo "Unknown option: $1"; exit 1 ;;
    *) [ -z "$PORT" ] || { echo "Only one port argument is accepted: '$1'"; exit 1; }; PORT=$1 ;;
  esac
  shift
done
if [ -n "$PORT" ]; then
  [[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT >= 1 && PORT <= 65535 )) || { echo "Invalid port: '$PORT'"; exit 1; }
fi

# --- Re-lock only: no restart ---
if [ "$ACTION" = lock ]; then
  MODE=""
  if lc_running && { [ -z "$PORT" ] || [ "$(lc_port)" = "$PORT" ]; }; then MODE=docker; PORT=${PORT:-$(lc_port)}
  elif ol_running; then MODE=host; PORT=${PORT:-$(ol_port)}; fi
  [ -n "$MODE" ] && [[ "$PORT" =~ ^[0-9]+$ ]] || { echo "No running server found; start one first, or pass the port: $0 --lock PORT"; exit 1; }
  lock_port "$PORT" "$MODE"
  exit 0
fi

# --- Uninstall: one engine by name, or both (a llama.cpp model name means the whole llama.cpp engine) ---
if [ "$ACTION" = uninstall ]; then
  case "$TARGET" in nanbeige|qwen38pi|ornith35b|ornith9b|llama.cpp) TARGET=llamacpp ;; esac
  DIRS=()
  if [ "$TARGET" != ollama ]; then for d in "${LC_DIRS[@]}"; do [ -d "$d" ] && DIRS+=("$d"); done; fi
  if [ "$TARGET" != llamacpp ]; then
    for d in /root/.ollama /usr/share/ollama ${OLLAMA_MODELS:+"$OLLAMA_MODELS"}; do [ -d "$d" ] && DIRS+=("$d"); done
  fi
  USED=$(du -shc "${DIRS[@]}" 2>/dev/null | tail -1 | cut -f1)
  if (( ! YES )); then
    read -rp "Remove ${TARGET:-llama.cpp and Ollama} and DELETE the model files in ${DIRS[*]:-(none)} (~${USED:-0})? [y/N] " a || true
    [[ $a == [yY]* ]] || { echo "Cancelled."; exit 0; }
  fi
  if [ "$TARGET" != ollama ]; then
    stop_llamacpp
    if command -v docker >/dev/null; then docker rmi "$LC_IMAGE" >/dev/null 2>&1 || true; fi
    for d in "${LC_DIRS[@]}"; do rm -rf "$d"; done
    echo "llama.cpp server and its model files removed. Docker itself is left installed."
  fi
  if [ "$TARGET" != llamacpp ]; then
    stop_ollama
    if [ -f /etc/systemd/system/ollama.service ]; then
      systemctl stop ollama 2>/dev/null || true
      systemctl disable ollama 2>/dev/null || true
      rm -f /etc/systemd/system/ollama.service
      systemctl daemon-reload 2>/dev/null || true
    fi
    pgrep -x ollama | xargs -r kill 2>/dev/null || true
    BIN=$(command -v ollama || true)
    if [ -n "$BIN" ]; then rm -f "$BIN"; rm -rf "$(dirname "$(dirname "$BIN")")/lib/ollama"; fi
    rm -rf /root/.ollama /usr/share/ollama ${OLLAMA_MODELS:+"$OLLAMA_MODELS"}
    id ollama >/dev/null 2>&1 && userdel ollama 2>/dev/null || true
    getent group ollama >/dev/null && groupdel ollama 2>/dev/null || true
    echo "Ollama and all its models removed."
  fi
  echo "Freed ~${USED:-0}."
  exit 0
fi

# ======================================================================================================
# llama.cpp sizing helpers (used by the first menu and the llama.cpp engine)
# ======================================================================================================
nb_need_mb() { echo $(( ($1 + LC_EXTRA_BYTES) / 1048576 + $2 * $3 / 1048576 )); }   # weight bytes, ctx, kv bytes/token -> MB (+ LoRA)
kv_bytes() { if [ "$1" = f16 ]; then echo "$LC_KV_F16"; else echo "$LC_KV_Q8"; fi; }
lc_budget() { LC_BUDGET_MB=$(( FREE_MB - LC_OVERHEAD_MB - (GPU_COUNT - 1) * 1024 )); }   # llama.cpp splits layers over all cards
fit_kv() {   # $1 = weight bytes, $2 = ctx -> most precise cache that fits (f16, then q8_0), or nothing
  local kv
  for kv in f16 q8_0; do
    if (( $(nb_need_mb "$1" "$2" "$(kv_bytes $kv)") <= LC_BUDGET_MB )); then echo "$kv"; return 0; fi
  done
  return 0
}
best_quant() {   # $1 = ctx -> best quant that fits (respects QUANT), or nothing
  local e q b s
  for e in "${LC_QUANTS[@]}"; do
    IFS='|' read -r q b s <<< "$e"
    [ -n "${QUANT:-}" ] && [ "$q" != "$QUANT" ] && continue
    [ -n "$(fit_kv "$b" "$1")" ] && { echo "$q"; return 0; }
  done
  return 0   # nothing fits: print nothing (a non-zero status would end the script under set -e)
}
mount_ok() {   # $1 = folder: can a container see a file in it?
  local new=0; [ -d "$1" ] || new=1
  mkdir -p "$1" && echo ok > "$1/.mount-probe"
  docker run --rm -v "$1:/p" busybox cat /p/.mount-probe 2>/dev/null | grep -qx ok
  local r=$?; rm -f "$1/.mount-probe"; [ "$new" = 1 ] && [ "$r" != 0 ] && rmdir "$1" 2>/dev/null; return $r
}

ask_port() {   # the last question: which port to serve on (skipped when given as an argument)
  [ -n "$PORT" ] && return 0
  if (( HAS_WHIPTAIL && INTERACTIVE )); then
    PORT=$(whiptail --backtitle "$BACKTITLE" --title "Port" --inputbox "Last step: port for the AI endpoint on this machine:" 9 60 20001 3>&1 1>&2 2>&3) || { echo "Cancelled."; exit 0; }
  else
    read -rp "Port for the AI endpoint on this machine [20001]: " PORT || true   # no terminal: the default
    PORT=${PORT:-20001}
  fi
  [[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT >= 1 && PORT <= 65535 )) || { echo "Invalid port: '$PORT'"; exit 1; }
}

# --- Ollama: sizes and fit for every model in OL_MODELS (registry lookups, no installs) ---
model_mb() {   # download size of an Ollama registry ref (or a HF GGUF), in MB; empty on failure
  if [ -n "${HF_MB[$1]:-}" ]; then echo "${HF_MB[$1]}"; return 0; fi
  local name=${1%%:*} tag=latest
  [[ $1 == *:* ]] && tag=${1##*:}
  [[ $name == */* ]] || name="library/$name"
  curl -fsSL -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
    "https://registry.ollama.ai/v2/$name/manifests/$tag" 2>/dev/null \
    | grep -o '"size":[0-9]*' | awk -F: '{s+=$2} END{if (s) print int(s/1048576)}'
}
ol_need_mb() { echo $(( $2 + KV[$1] * $3 * KV_PCT / 100 / 1024 + FIXED[$1] + OL_OVERHEAD_MB )); }   # model i, weights MB, ctx
confirm_yesno() {   # $1 = question (may contain \n); returns 0 for yes
  local a
  [ -n "${ONLY:-}" ] && return 0   # rows were named explicitly; don't re-ask
  if (( HAS_WHIPTAIL )); then
    whiptail --backtitle "$BACKTITLE" --title "Won't fit in VRAM" --yesno "$1" 16 78
  else
    printf '%b\n' "$1"; read -rp "[y/N] " a; [[ $a == [yY]* ]]
  fi
}

ol_prepare() {   # sets KV_TYPE/KV_PCT/QUANT_ORDER, the per-model arrays (REF, Q, MCTX, GB, ...) and REC
  # OLLAMA_KV_CACHE_TYPE is global: Ollama applies it to every model. q8_0 ~53% of the f16 bytes, q4_0 ~27%.
  KV_TYPE=${OLLAMA_KV_CACHE_TYPE:-f16}
  case "$KV_TYPE" in
    f16)  KV_PCT=100 ;;
    q8_0) KV_PCT=53 ;;
    q4_0) KV_PCT=27 ;;
    *) echo "Unsupported OLLAMA_KV_CACHE_TYPE '$KV_TYPE' (use f16, q8_0 or q4_0)"; exit 1 ;;
  esac
  export OLLAMA_KV_CACHE_TYPE="$KV_TYPE"
  export OLLAMA_FLASH_ATTENTION=1          # required for a quantized K/V cache
  export OLLAMA_NUM_PARALLEL=1             # each parallel slot needs its own full KV cache; subagents queue
  export OLLAMA_MAX_LOADED_MODELS=1        # sizes assume one model in VRAM at a time
  # Ollama unloads an idle model after 5 min, then the next request waits ~20-60 s to reload it. Keep it loaded.
  export OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:--1}
  QUANT_ORDER="Q8-first"; [ "${FAST:-0}" = 1 ] && QUANT_ORDER="Q4-first (FAST=1)"

  # --- Sizes: look every ref up in parallel (~0.5 s per registry call), then pick per model what fits ---
  echo "Checking Ollama model sizes..."
  SIZE_DIR=$(mktemp -d); trap 'rm -rf "$SIZE_DIR"' EXIT
  local e r r8 r4 rk kv fx cp n note i s8 s4 q c s opt
  for e in "${OL_MODELS[@]}"; do
    IFS='|' read -r _ r8 r4 _ <<< "$e"
    for r in $r8 $r4; do model_mb "$r" > "$SIZE_DIR/$(echo "$r" | tr '/:' '__')" & done
  done
  wait
  cached_mb() { cat "$SIZE_DIR/$(echo "$1" | tr '/:' '__')" 2>/dev/null; }

  # Per model: the best option that fits -> REF (tag to pull), Q, MCTX (0 = too big), GB
  RANKS=() CAPS=() MNAME=() NOTE=() KV=() FIXED=() REF=() Q=() MCTX=() GB=()
  for e in "${OL_MODELS[@]}"; do
    IFS='|' read -r rk r8 r4 kv fx cp n note <<< "$e"
    i=${#RANKS[@]}
    RANKS+=("$rk") CAPS+=("$cp") MNAME+=("$n") NOTE+=("$note") KV+=("$kv") FIXED+=("$fx")
    s8=$(cached_mb "$r8"); s4=""; [ -n "$r4" ] && s4=$(cached_mb "$r4")
    REF+=("") Q+=("") MCTX+=(0) GB+=("?")
    # Q8-first reads ~29 GB of weights per decoded token on a 48 GB card; FAST=1 tries Q4 (~17 GB) first.
    if [ "${FAST:-0}" = 1 ]; then OPTS=("4 $OL_CTX_MAX" "4 $OL_CTX_MIN" "8 $OL_CTX_MAX" "8 $OL_CTX_MIN")
    else OPTS=("8 $OL_CTX_MAX" "8 $OL_CTX_MIN" "4 $OL_CTX_MAX" "4 $OL_CTX_MIN"); fi
    for opt in "${OPTS[@]}"; do
      read -r q c <<< "$opt"
      if [ "$q" = 8 ]; then r=$r8 s=$s8; else r=$r4 s=$s4; fi
      [ -n "$s" ] || continue
      if (( $(ol_need_mb $i $s $c) <= FREE_MB )); then REF[i]=$r Q[i]=$q MCTX[i]=$c GB[i]=$(( $(ol_need_mb $i $s $c) / 1024 )); break; fi
    done
    if [ -z "${REF[i]}" ]; then   # too big for any option: smallest @64K, flagged
      if [ -n "$s4" ]; then REF[i]=$r4 Q[i]=4 s=$s4; else REF[i]=$r8 Q[i]=8 s=${s8:-0}; fi
      GB[i]=$(( $(ol_need_mb $i $s $OL_CTX_MIN) / 1024 ))
    fi
    [ -z "$r4" ] && Q[i]=""   # own models: single tag, quant not shown
  done

  REC=-1
  for i in "${!RANKS[@]}"; do [[ ${RANKS[i]} != - && ${MCTX[i]} != 0 ]] && { REC=$i; break; }; done
  if (( REC >= 0 )); then REC_TXT="Recommended: ${MNAME[REC]} at ${Q[REC]:+Q${Q[REC]} }$((MCTX[REC]/1024))K context, $KV_TYPE KV"
  else REC_TXT="None of the curated models fit in VRAM."; fi
  fit_label() {
    local j=$1 qq=${Q[$1]:+Q${Q[$1]} }
    if (( MCTX[j] == 0 )); then echo "TOO BIG ~${GB[j]}G@64K"; else echo "${qq}~${GB[j]}G@$((MCTX[j]/1024))K"; fi
  }
  return 0
}

# ======================================================================================================
# Install: GPU, engine, port
# ======================================================================================================
banner
access_mode
(( DRY_RUN )) && echo "DRY RUN: the full wizard, then the plan. Nothing is installed, downloaded or started."
if (( DRY_RUN )) && [ -n "${SUMMON_GPU:-}" ]; then
  # Preview another card: "NAME:TOTAL_MB" (free VRAM = total - 500 MB, like an idle card)
  GPU_INFO="${SUMMON_GPU%%:*}, ${SUMMON_GPU##*:}, $(( ${SUMMON_GPU##*:} - 500 ))"
  [[ ${SUMMON_GPU##*:} =~ ^[0-9]+$ ]] || { echo "SUMMON_GPU must look like \"RTX 3090:24576\" (name:MB)"; exit 1; }
else
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found - no NVIDIA GPU/driver? (Preview: SUMMON_GPU=\"RTX 3090:24576\" $0 --dry-run)"; exit 1; }
  GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits)
fi
GPU_COUNT=$(echo "$GPU_INFO" | wc -l)
GPU_NAME=$(echo "$GPU_INFO" | head -1 | cut -d, -f1)
FREE_MB=$(echo "$GPU_INFO" | awk -F', ' '{s+=$3} END{print s}')     # free VRAM summed across GPUs
TOTAL_MB=$(echo "$GPU_INFO" | awk -F', ' '{s+=$2} END{print s}')
# A server of ours that is still running holds its VRAM, but it gets replaced: count that VRAM as free, or a
# rerun to switch weights would find "no room". ~1 GB per card stays reserved for the driver/CUDA context.
if (( ! DRY_RUN || ${#SUMMON_GPU} == 0 )) && { lc_running || ol_running; }; then
  FREE_MB=$(( TOTAL_MB - GPU_COUNT * 1024 ))
  echo "Detected ${GPU_COUNT}x $GPU_NAME, ~$((FREE_MB/1024)) GB usable VRAM (counting the running server's, which gets replaced)"
else
  echo "Detected ${GPU_COUNT}x $GPU_NAME, $((FREE_MB/1024)) GB free VRAM"
fi

INTERACTIVE=0; [ -t 0 ] && [ -t 1 ] && INTERACTIVE=1
if (( INTERACTIVE && ! DRY_RUN )) && ! command -v whiptail >/dev/null; then
  apt_get whiptail >/dev/null 2>&1 || true   # optional: plain-text menus work too
fi
HAS_WHIPTAIL=0; command -v whiptail >/dev/null && HAS_WHIPTAIL=1

# --- One menu with every model: llama.cpp (tuned per model) and Ollama, side by side ---
# ENGINE=... or a model id as the first argument skips it (./summon.sh ornith9b 40000). "ollama" picks the
# recommended Ollama model (or ONLY=1,3 for several). Default: the first llama.cpp model that reaches 128K on this
# card (Nanbeige on 48 GB, Ornith 9B on 24 GB), else the best-ranked Ollama model that fits.
[ -n "$TARGET" ] && [ -z "${ENGINE:-}" ] && ENGINE=$TARGET
case "${ENGINE:-}" in llamacpp|llama.cpp) ENGINE=nanbeige ;; esac
if [ -z "${ENGINE:-}" ] && { [ -n "${ONLY:-}" ] || [ -n "${FAST:-}" ]; }; then ENGINE=ollama; fi
case "${ENGINE:-}" in
  nanbeige|qwen38pi|ornith35b|ornith9b|ollama|"") ;;
  *) echo "ENGINE must be one of: ${LC_MODEL_IDS[*]} ollama (got '$ENGINE')"; exit 1 ;;
esac
OL_PICK=""
if [ -z "${ENGINE:-}" ] || [ "$ENGINE" = ollama ]; then ol_prepare; fi
if [ -z "${ENGINE:-}" ] || { [ "$ENGINE" = ollama ] && [ -z "${ONLY:-}" ]; }; then
  # Spaced layout: an engine header per group and a gap between the groups (header rows can't be picked). Short
  # terminals get the compact list instead, so everything stays on one screen; descriptions are cut to the width.
  DEF_ENGINE="" LC_ITEMS=() OL_ITEMS=() ITEMS=()
  MENU_COLS=$(( $(tput cols 2>/dev/null || echo 124) - 4 )); (( MENU_COLS > 150 )) && MENU_COLS=150
  MENU_LINES=$(tput lines 2>/dev/null || echo 34)
  DESC_W=$(( MENU_COLS - 12 - 28 - 2 - 16 - 2 )); (( DESC_W < 12 )) && DESC_W=12   # borders/indent + name + fit
  row() {   # $1 name, $2 fit, $3 description -> one spaced, width-safe menu line
    local d=$3; (( ${#d} > DESC_W )) && d="${d:0:DESC_W-3}..."
    printf '%-28.28s  %-16s  %s' "$1" "$2" "$d"
  }
  for id in "${LC_MODEL_IDS[@]}"; do
    lc_model "$id"; lc_budget
    # Fit label like Ollama's: weights ~total VRAM @ context (128K, or the longest context that fits)
    q=$(best_quant 131072); c=131072
    if [ -z "$q" ]; then for c in "${LC_CTX_TIERS[@]}"; do q=$(best_quant "$c"); [ -n "$q" ] && break; done; fi
    [ -z "$DEF_ENGINE" ] && [ "$c" = 131072 ] && [ -n "$q" ] && DEF_ENGINE=$id
    if [ -n "$q" ]; then
      for e in "${LC_QUANTS[@]}"; do [ "${e%%|*}" = "$q" ] && b=$(echo "$e" | cut -d'|' -f2); done
      kv=$(fit_kv "$b" "$c")
      fit="$q ~$(( ($(nb_need_mb "$b" "$c" "$(kv_bytes "$kv")") + LC_OVERHEAD_MB) / 1024 ))G@$((c/1024))K"
    else
      lc_smallest=${LC_QUANTS[${#LC_QUANTS[@]}-1]}
      fit="TOO BIG ~$(( ($(nb_need_mb "$(echo "$lc_smallest" | cut -d'|' -f2)" 32768 "$LC_KV_Q8") + LC_OVERHEAD_MB) / 1024 ))G@32K"
    fi
    case "$id" in   # the model's own summary (model card), shortened
      nanbeige) blurb="Lightweight agent for coding and research. AA index 20" ;;
      qwen38pi) blurb="Qwen3.8 + RL-tuned LoRA for the pi agent. No thinking" ;;
      ornith35b) blurb="Self-improving MoE for coding agents. 3B active, fast" ;;
      ornith9b) blurb="Self-improving 9B: light, fast, 256K context on 24 GB cards" ;;
    esac
    LC_ITEMS+=("$id" "$(row "$LC_TITLE" "$fit" "$blurb")")
  done
  for i in "${!RANKS[@]}"; do OL_ITEMS+=("ol:$((i+1))" "$(row "${MNAME[i]}" "$(fit_label $i)" "${NOTE[i]}")"); done
  if (( MENU_LINES >= ${#LC_ITEMS[@]}/2 + ${#OL_ITEMS[@]}/2 + 3 + 11 )); then
    # (labels must not start with "-": whiptail would read them as options)
    ITEMS=(_h1 "== llama.cpp  .  tuned per model, verified downloads ================================="
           "${LC_ITEMS[@]}" _gap " "
           _h2 "== Ollama  .  pulled from ollama.com, sized to this card ============================="
           "${OL_ITEMS[@]}")
  else
    ITEMS=("${LC_ITEMS[@]}" "${OL_ITEMS[@]}")
  fi
  (( REC >= 0 )) && [ -z "$DEF_ENGINE" ] && DEF_ENGINE="ol:$((REC+1))"
  [ "$ENGINE" = ollama ] && DEF_ENGINE="ol:$((REC >= 0 ? REC+1 : 1))"
  DEF_ENGINE=${DEF_ENGINE:-nanbeige}
  if [ -n "${CTX:-}" ] || [ -n "${QUANT:-}" ] || (( ! INTERACTIVE )); then PICKED=$DEF_ENGINE
  else
    PICKED=_h1
    while [[ $PICKED == _* ]]; do   # a header or the gap: show the menu again
      PICKED=$(CHOOSE_NOTAGS=1 CHOOSE_W=150 CHOOSE_TEXTLINES=1 choose "Summon v$SUMMON_VERSION - ${GPU_COUNT}x $GPU_NAME, $((FREE_MB/1024)) GB" \
        "Which AI do you want to summon? Pre-selected: the best model reaching 128K here. Size = weights ~VRAM@context." \
        "$DEF_ENGINE" "${ITEMS[@]}") || { echo "Cancelled."; exit 0; }
    done
  fi
  if [[ $PICKED == ol:* ]]; then ENGINE=ollama OL_PICK=$(( ${PICKED#ol:} - 1 )); else ENGINE=$PICKED; fi
fi

# ======================================================================================================
# Engine: llama.cpp (the model chosen above)
# ======================================================================================================
fetch_gguf() {   # $1 = repo URL, $2 = file, $3 = bytes, $4 = sha256 -> $DDIR/models/$2 (resumable, verified)
  # O(1) checks catch the realistic failures (truncated download: exact byte count; an HTML error page: GGUF magic).
  # VERIFY_SHA256=1 / --verify adds a full-file hash, which re-reads the whole file.
  local URL=$1 FILE=$2 BYTES=$3 SHA=$4
  local PART="$DDIR/models/$FILE.part" MODEL="$DDIR/models/$FILE"
  part_size() { stat -c %s "$PART" 2>/dev/null || echo 0; }
  if [ -f "$MODEL" ]; then
    echo "Already present: $MODEL (no download, no hash)"
  else
    # One HEAD request: HF publishes the blob's exact size and sha256 (x-linked-size / x-linked-etag).
    local EXPECT_SHA=$SHA META REMOTE_SIZE REMOTE_SHA SIZE
    META=$(curl -fsSIL --connect-timeout 10 --max-time 30 "$URL/$FILE" 2>/dev/null | tr -d '\r' || true)
    if [ -n "$META" ]; then
      REMOTE_SIZE=$(echo "$META" | awk -F': ' 'tolower($1)=="x-linked-size" {print $2; exit}')
      REMOTE_SHA=$(echo "$META" | awk -F': ' 'tolower($1)=="x-linked-etag" {gsub(/"/,"",$2); print $2; exit}')
      if [ -n "$REMOTE_SIZE" ] && [ "$REMOTE_SIZE" != "$BYTES" ]; then
        echo "HF reports $REMOTE_SIZE bytes for $FILE but this script expects $BYTES (repacked file?)."
        echo "Update the file table for $LC_ID in $0 before retrying."; exit 1
      fi
      if [ -n "$REMOTE_SHA" ] && [ "$REMOTE_SHA" != "$SHA" ]; then
        echo "Note: HF's sha256 for $FILE differs from the script's table; using HF's value."
        EXPECT_SHA=$REMOTE_SHA
      fi
    fi
    # -C - resumes a partial file, but curl errors (33/416) if the .part is already complete: reuse it instead.
    if [ -f "$PART" ] && (( $(part_size) > BYTES )); then
      echo "Removing an oversized partial download ($(part_size) bytes > $BYTES)."; rm -f "$PART"
    fi
    if [ -f "$PART" ] && (( $(part_size) == BYTES )); then
      echo "Reusing the completed download $PART"
    else
      local NEED_MB FREE_DISK_MB RETRY
      NEED_MB=$(( (BYTES - $(part_size)) / 1048576 + 1024 ))
      FREE_DISK_MB=$(df -Pm "$DDIR/models" | awk 'NR==2 {print $4}')
      if [ -n "$FREE_DISK_MB" ] && (( FREE_DISK_MB < NEED_MB )); then
        echo "Not enough disk: $FREE_DISK_MB MB free in $DDIR/models, the download needs ~$NEED_MB MB."
        echo "Delete other quants there (ls $DDIR/models) or rent a box with more disk."; exit 1
      fi
      [ -f "$PART" ] && echo "Resuming $PART from $(part_size) bytes..."
      if (( BYTES >= 1073741824 )); then echo "Downloading $FILE ($((BYTES/1073741824)) GB)..."
      else echo "Downloading $FILE ($((BYTES/1048576)) MB)..."; fi
      RETRY=(--retry 5 --retry-delay 5)   # --retry-all-errors (curl >= 7.71) also retries mid-transfer drops
      if curl --help all 2>/dev/null | grep -q -- '--retry-all-errors'; then RETRY+=(--retry-all-errors); fi
      curl -fL "${RETRY[@]}" --connect-timeout 20 -C - --progress-bar -o "$PART" "$URL/$FILE"
    fi
    SIZE=$(part_size)
    if (( SIZE != BYTES )); then
      echo "Incomplete download: $SIZE of $BYTES bytes in $PART. Rerun to resume."
      echo "If it never grows, the size in the script is stale (check the Hugging Face file list)."; exit 1
    fi
    [ "$(head -c 4 "$PART")" = GGUF ] || { echo "$PART doesn't start with the GGUF magic (bad download?); delete it and retry."; exit 1; }
    if [ "$VERIFY_SHA256" = 1 ]; then
      echo "Checking sha256 (this reads the whole file)..."
      echo "$EXPECT_SHA  $PART" | sha256sum -c --status || { echo "sha256 mismatch: delete $PART and retry"; exit 1; }
    else
      echo "Verified $SIZE bytes + GGUF header; sha256 skipped (VERIFY_SHA256=1 / --verify to check)."
    fi
    mv "$PART" "$MODEL"
  fi
  return 0
}

install_llamacpp() {
  lc_model "$1"; lc_budget
  if [ -n "${CTX:-}" ]; then
    [[ "$CTX" =~ ^[0-9]+$ ]] && (( CTX > 0 )) || { echo "CTX must be a positive number of tokens"; exit 1; }
  fi
  if [ -n "${QUANT:-}" ]; then
    local qok=0 e
    for e in "${LC_QUANTS[@]}"; do [ "${e%%|*}" = "$QUANT" ] && qok=1; done
    (( qok )) || { echo "Unknown QUANT '$QUANT' for $LC_TITLE (have: $(printf '%s ' "${LC_QUANTS[@]%%|*}"))"; exit 1; }
  fi
  TEMPERATURE=${TEMPERATURE:-$LC_TEMP}
  TOP_P=${TOP_P:-$LC_TOP_P}
  TOP_K=${TOP_K:-$LC_TOP_K}
  if [ -n "$TOP_K" ]; then [[ "$TOP_K" =~ ^[0-9]+$ ]] || { echo "TOP_K must be a whole number"; exit 1; }; fi
  for v in "$TEMPERATURE" "$TOP_P"; do
    [[ "$v" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "TEMPERATURE/TOP_P must be numbers (got '$v')"; exit 1; }
  done
  if [ -n "${THINK_BUDGET:-}" ]; then
    [[ "$THINK_BUDGET" =~ ^[0-9]+$ ]] || { echo "THINK_BUDGET must be a non-negative integer of tokens"; exit 1; }
  fi
  case "${VERIFY_SHA256:-0}" in 1|true|yes|on) VERIFY_SHA256=1 ;; *) VERIFY_SHA256=0 ;; esac

  # --- Context, then weights ---
  # Speed: every generated token reads all the weights plus the part of the cache in use (llama.cpp only attends
  # over used cells, padded to 256), so reserving 256K costs no speed until a conversation is that long.
  # Bigger weights are more precise and slower at every length. Default: 128K with the best weights that fit.
  local DEF_CTX="" c q b s e kv tot ITEMS DEF_Q PICK
  [ -n "$(best_quant 131072)" ] && DEF_CTX=131072
  if [ -z "$DEF_CTX" ]; then
    for c in "${LC_CTX_TIERS[@]}"; do [ -n "$(best_quant "$c")" ] && { DEF_CTX=$c; break; }; done
  fi
  if [ -z "$DEF_CTX" ] && [ -z "${CTX:-}" ]; then
    echo "$LC_TITLE doesn't fit in $((FREE_MB/1024)) GB of VRAM${QUANT:+ with QUANT=$QUANT} at 32K or more."
    echo "Try a smaller model (./summon.sh ornith9b) or Ollama (./summon.sh ollama)."
    exit 1
  fi
  if (( INTERACTIVE )) && [ -z "${CTX:-}" ] && [ -z "${QUANT:-}" ]; then
    ITEMS=()
    for c in "${LC_CTX_TIERS[@]}"; do
      q=$(best_quant "$c"); [ -n "$q" ] || continue
      for e in "${LC_QUANTS[@]}"; do [ "${e%%|*}" = "$q" ] && b=$(echo "$e" | cut -d'|' -f2); done
      ITEMS+=("$c" "$(printf '%3dK context    best weights here: %-7s + %-4s cache' $((c / 1024)) "$q" "$(fit_kv "$b" "$c")")")
    done
    CTX=$(choose "$LC_TITLE - ${GPU_COUNT}x $GPU_NAME, $((FREE_MB/1024)) GB" \
      "Step 1/2: context window. Reserving more costs VRAM (so smaller weights), not speed.\nSpeed drops with the tokens actually in a conversation, at any setting." \
      "$DEF_CTX" "${ITEMS[@]}") || { echo "Cancelled."; exit 0; }
  fi
  CTX=${CTX:-$DEF_CTX}
  if (( INTERACTIVE )) && [ -z "${QUANT:-}" ]; then
    ITEMS=() DEF_Q=""
    for e in "${LC_QUANTS[@]}"; do
      IFS='|' read -r q b s <<< "$e"
      kv=$(fit_kv "$b" "$CTX"); [ -n "$kv" ] || continue
      [ -n "$DEF_Q" ] || DEF_Q=$q
      tot=$(( ($(nb_need_mb "$b" "$CTX" "$(kv_bytes "$kv")") + LC_OVERHEAD_MB) / 1024 ))
      ITEMS+=("$q" "$(printf '%2d.%d GB weights + %-4s cache = ~%2d GB   speed ~%3d%% of %s' \
        $((b / 1073741824)) $((b * 10 / 1073741824 % 10)) "$kv" "$tot" $((LC_REF_BYTES * 100 / b)) "$LC_REF_Q")")
    done
    (( ${#ITEMS[@]} )) || { echo "Nothing fits at $((CTX/1024))K on this card; pick a shorter CTX."; exit 1; }
    QUANT=$(choose "$LC_TITLE at $((CTX/1024))K context" \
      "Step 2/2: weights. Higher = more precise but slower: every token reads all of them.\n$LC_SPEED_NOTE Speed % is for an empty context;\nthe gap narrows as the conversation (and its cache) grows." \
      "$DEF_Q" "${ITEMS[@]}") || { echo "Cancelled."; exit 0; }
  fi
  QUANT=${QUANT:-$(best_quant "$CTX")}
  PICK=""
  for e in "${LC_QUANTS[@]}"; do
    IFS='|' read -r q b s <<< "$e"
    [ "$q" = "$QUANT" ] || continue
    kv=$(fit_kv "$b" "$CTX"); [ -n "$kv" ] && PICK="$q|$b|$s|$CTX|$kv"
  done
  if [ -z "$PICK" ]; then
    echo "$LC_TITLE ${QUANT:+$QUANT }doesn't fit in $((FREE_MB/1024)) GB of VRAM at $((CTX/1024))K context."
    echo "Pick a shorter CTX or a smaller QUANT (run without them for the menu)."
    exit 1
  fi
  IFS='|' read -r QUANT_SEL BYTES SHA CTX_SEL KV_SEL <<< "$PICK"
  ask_port
  echo "Plan: $QUANT_SEL weights ($((BYTES/1073741824)) GB) + $KV_SEL K/V at $((CTX_SEL/1024))K context = ~$(( ($(nb_need_mb "$BYTES" "$CTX_SEL" "$(kv_bytes "$KV_SEL")") + LC_OVERHEAD_MB) / 1024 )) GB of $((FREE_MB/1024)) GB"
  if (( DRY_RUN )); then
    echo
    echo "Would do (dry run, nothing done):"
    echo "  1. Install what's missing: curl, Docker, NVIDIA container toolkit"
    echo "  2. Download $LC_FILE_PREFIX-$QUANT_SEL.gguf ($((BYTES/1073741824)) GB), check size + GGUF header (sha256 with --verify)"
    echo "     from ${LC_REPO%/resolve/main}"
    if [ -n "$LC_LORA" ]; then
      echo "     + LoRA adapter ${LC_LORA%%|*} ($(( $(echo "$LC_LORA" | cut -d'|' -f2) / 1048576 )) MB) from $(echo "$LC_LORA" | cut -d'|' -f4 | sed 's#/resolve/main##')"
    fi
    echo "  3. Pull $LC_IMAGE"
    echo "  4. Start llama-server: $LC_TITLE $QUANT_SEL, -c $CTX_SEL, $KV_SEL cache, all layers on the GPU,"
    echo "     temp ${TEMPERATURE} / top_p ${TOP_P}${TOP_K:+ / top_k $TOP_K}, n-gram speculation ${SPEC:-on}, host-RAM prompt cache"
    [ -n "$LC_KWARGS" ] && echo "     chat template: $LC_KWARGS"
    echo "  5. Access: $(access_text)"
    echo "  6. Wait for /health, then test a tool call"
    echo
    echo "Endpoint: http://$HOST_HINT:$PORT/v1   model: $LC_ALIAS   context window: $CTX_SEL"
    echo "Run it for real:       $0 $LC_ID $PORT"
    echo "Same, without menus:   CTX=$CTX_SEL QUANT=$QUANT_SEL $0 $LC_ID $PORT"
    return 0
  fi

  # --- Prerequisites: only touch apt when something is missing ---
  local APT_PKGS=()
  command -v curl   >/dev/null || APT_PKGS+=(curl ca-certificates)
  command -v gpg    >/dev/null || APT_PKGS+=(gnupg)
  command -v docker >/dev/null || APT_PKGS+=(docker.io)
  if (( ${#APT_PKGS[@]} )); then
    apt_get "${APT_PKGS[@]}"
    apt_get btop bmon >/dev/null 2>&1 || echo "Note: optional monitors (btop/bmon) unavailable, continuing."
  fi

  # --- Docker + NVIDIA container toolkit (rented pods often have no systemd: start dockerd directly) ---
  local RESTART_DOCKER=""
  if ! command -v nvidia-ctk >/dev/null; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    APT_UPDATED=0; apt_get nvidia-container-toolkit   # new repo: needs its own apt-get update
    nvidia-ctk runtime configure --runtime=docker >/dev/null
    RESTART_DOCKER=1
  fi
  if [ -d /run/systemd/system ]; then
    if [ -n "$RESTART_DOCKER" ]; then systemctl restart docker; else systemctl start docker; fi
  else
    if [ -n "$RESTART_DOCKER" ] && [ -f /var/run/docker.pid ]; then kill "$(cat /var/run/docker.pid)" 2>/dev/null || true; sleep 3; fi
    if ! docker info >/dev/null 2>&1; then
      nohup dockerd >/var/log/dockerd.log 2>&1 &
      for i in $(seq 30); do docker info >/dev/null 2>&1 && break; sleep 1; done
    fi
  fi
  docker info >/dev/null 2>&1 || { echo "Docker cannot run on this machine. Rent a VM or a Docker-in-Docker pod, or use ENGINE=ollama."; tail -5 /var/log/dockerd.log 2>/dev/null; exit 1; }
  local DDIR="" d
  for d in "${LC_DIRS[@]}"; do if mount_ok "$d"; then DDIR=$d; break; fi; done
  [ -n "$DDIR" ] || { echo "Docker can't mount any folder into containers here. Use a VM or a privileged pod, or ENGINE=ollama."; exit 1; }
  echo "Docker OK: $(docker --version). Data folder: $DDIR"

  # --- Model download(s): the weights, plus the LoRA adapter if the model has one ---
  local FILE="$LC_FILE_PREFIX-$QUANT_SEL.gguf" LORA_FILE="" LORA_BYTES LORA_SHA LORA_REPO
  mkdir -p "$DDIR/models"
  docker pull -q "$LC_IMAGE" >/dev/null 2>&1 & local PULL_PID=$!   # pull the ~3 GB image during the download
  fetch_gguf "$LC_REPO" "$FILE" "$BYTES" "$SHA"
  if [ -n "$LC_LORA" ]; then
    IFS='|' read -r LORA_FILE LORA_BYTES LORA_SHA LORA_REPO <<< "$LC_LORA"
    fetch_gguf "$LORA_REPO" "$LORA_FILE" "$LORA_BYTES" "$LORA_SHA"
  fi
  local OTHERS
  OTHERS=$(ls "$DDIR/models" | grep -vxF -e "$FILE" -e "${LORA_FILE:-//}" || true)
  [ -n "$OTHERS" ] && echo "Other model files in $DDIR/models (kept, switching back is instant): $OTHERS"

  # --- Start (or replace) the server ---
  wait "$PULL_PID" || docker pull -q "$LC_IMAGE" >/dev/null   # retry in the foreground so a pull error shows
  stop_ollama; stop_llamacpp; wait_port_free
  apply_access docker   # before the port opens
  # Prompt cache in host RAM: with one slot, a side request (chat title, subagent) overwrites the main
  # conversation's K/V. llama-server parks the old state in RAM first, but its default 8 GiB cap holds only ~75K
  # tokens of this model. Give it half the available host RAM.
  local AVAIL_MB CACHE_RAM_MB SERVER_ARGS
  AVAIL_MB=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  CACHE_RAM_MB=$(( AVAIL_MB / 2 )); (( CACHE_RAM_MB < 8192 )) && CACHE_RAM_MB=8192
  SERVER_ARGS=(--cache-ram "$CACHE_RAM_MB")
  if [ -n "${THINK_BUDGET:-}" ]; then SERVER_ARGS+=(--reasoning-budget "$THINK_BUDGET"); fi
  if [ -n "$TOP_K" ]; then SERVER_ARGS+=(--top-k "$TOP_K"); fi
  if [ -n "$LC_KWARGS" ]; then SERVER_ARGS+=(--chat-template-kwargs "$LC_KWARGS"); fi
  if [ -n "$LORA_FILE" ]; then SERVER_ARGS+=(--lora "/models/$LORA_FILE"); fi
  # n-gram self-speculative decoding: drafts from text already in the context, verified by the model, so the
  # output is identical. Pays off when the agent re-types code it has seen. No draft model, no extra VRAM.
  if [ "${SPEC:-ngram-mod}" != off ]; then
    SERVER_ARGS+=(--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 16 --spec-ngram-mod-n-max 48)
  fi
  # -ngl 999 + --fit off: every layer on the GPU or fail loudly (no silent CPU offload, no silently smaller -c).
  # -ctv q8_0 needs Flash Attention (-fa on). -np 1: one slot, the whole context for one conversation.
  # --cache-reuse: reuse cached K/V chunks when an agent edits earlier messages. --jinja: native XML tool calls.
  docker run -d --name "$LC_NAME" --restart unless-stopped --gpus all \
    -v "$DDIR/models:/models:ro" -p "$BIND:$PORT:8080" "$LC_IMAGE" \
    -m "/models/$FILE" --alias "$LC_ALIAS" --host 0.0.0.0 --port 8080 \
    --jinja \
    -c "$CTX_SEL" -np 1 -ngl 999 --fit off -fa on -ctk "$KV_SEL" -ctv "$KV_SEL" \
    --cache-reuse 256 --temp "$TEMPERATURE" --top-p "$TOP_P" "${SERVER_ARGS[@]}" >/dev/null

  echo "Loading the model (1-3 min)..."
  for i in $(seq 300); do
    # a crash at load (e.g. out of VRAM) shows as a stopped container or, with the restart policy, a restart count
    if ! docker ps --format '{{.Names}}' | grep -qx "$LC_NAME" || [ "$(docker inspect -f '{{.RestartCount}}' "$LC_NAME" 2>/dev/null)" != 0 ]; then
      docker update --restart=no "$LC_NAME" >/dev/null 2>&1 || true
      echo "The server stopped. Last log lines:"; docker logs --tail 25 "$LC_NAME" 2>&1
      echo "If it ran out of memory, rerun with a smaller context, e.g.: CTX=$(( CTX_SEL / 2 )) $0 $PORT"; exit 1
    fi
    curl -fs --max-time 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
    sleep 2
  done
  curl -fs --max-time 5 "http://127.0.0.1:$PORT/health" | grep -q '"ok"' || { echo "Server not healthy after 10 min:"; docker logs --tail 25 "$LC_NAME" 2>&1; exit 1; }

  # --- Smoke test: a tool call must come back as an OpenAI tool_calls entry ---
  echo "Testing tool calling..."
  local R TOOLS VRAM
  R=$(curl -fs --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
    "model": "'"$LC_ALIAS"'", "max_tokens": 4096,
    "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}' || true)
  if echo "$R" | grep -q '"tool_calls"' && echo "$R" | grep -q 'get_weather'; then TOOLS="OK"; else TOOLS="NOT DETECTED - check: docker logs $LC_NAME"; fi
  VRAM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tr '\n' ' ')

  echo
  echo "Done. $LC_TITLE $QUANT_SEL is serving on $BIND:$PORT (OpenAI API, llama.cpp)"
  echo "  Context:     $CTX_SEL tokens, $KV_SEL K/V cache, all layers on the GPU (VRAM used: $VRAM)"
  echo "  Sampling:    temp $TEMPERATURE / top_p $TOP_P${TOP_K:+ / top_k $TOP_K}${THINK_BUDGET:+, thinking capped at $THINK_BUDGET tokens}"
  echo "  Speculation: ${SPEC:-ngram-mod} (n-gram drafts from the context; SPEC=off to disable)"
  echo "  Tool calls:  $TOOLS"
  echo "  Base URL:    http://$HOST_HINT:$PORT/v1     model: $LC_ALIAS     api key: any string"
  echo "  Access:      $(access_text)"
  echo "  Clients:     set the context window to $CTX_SEL; one request at a time (others queue)"
  echo "  Prompt cache: up to $((CACHE_RAM_MB/1024)) GB of host RAM keeps conversations across side requests"
  echo "  Logs:        docker logs -f $LC_NAME          Monitor: btop / nvidia-smi"
}

# ======================================================================================================
# Engine: Ollama
# ======================================================================================================
install_ollama() {
  # --- Prerequisites (not in a dry run) ---
  local APT_PKGS=()
  if (( ! DRY_RUN )); then
    command -v curl   >/dev/null || APT_PKGS+=(curl ca-certificates)
    command -v screen >/dev/null || APT_PKGS+=(screen)
    command -v zstd   >/dev/null || APT_PKGS+=(zstd)
    if (( ${#APT_PKGS[@]} )); then
      apt_get "${APT_PKGS[@]}"
      apt_get btop bmon >/dev/null 2>&1 || echo "Note: optional monitors (btop/bmon) unavailable, continuing."
    fi
    command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
  fi

  # What to pull: the model picked in the menu, or the rows in ONLY=1,3 (several at once, no menu)
  local N=${#RANKS[@]} t ONLY_PICKS=""
  SEL=()
  if [ -n "${ONLY:-}" ]; then
    for t in ${ONLY//,/ }; do
      if [[ "$t" =~ ^[0-9]+$ ]] && (( t >= 1 && t <= N )); then SEL+=($((t-1))); ONLY_PICKS+=" $t"
      else echo "Ignoring invalid row in ONLY: '$t' (rows 1..$N: $0 --help)"; fi
    done
    (( ${#SEL[@]} )) || { echo "ONLY=${ONLY} has no valid Ollama rows (1..$N)."; exit 1; }
  elif [ -n "$OL_PICK" ]; then
    SEL=("$OL_PICK")
  elif (( REC >= 0 )); then
    SEL=("$REC")
  fi
  echo "Ollama: $QUANT_ORDER, $KV_TYPE KV cache. Pulling: $(for i in "${SEL[@]}"; do printf '%s ' "${MNAME[i]}"; done)"

  # Warn if anything selected won't fit even at Q4 @64K
  local TOO_BIG="" KEEP
  for i in "${SEL[@]}"; do [ "${MCTX[i]}" = 0 ] && TOO_BIG+="${MNAME[i]} (~${GB[i]}GB @64K)\n"; done
  if [ -n "$TOO_BIG" ] && ! confirm_yesno \
      "Only $((FREE_MB/1024))GB VRAM. These will spill to CPU (slow):\n\n${TOO_BIG}\nPull them anyway? (No = skip them)"; then
    KEEP=(); for i in "${SEL[@]}"; do [ "${MCTX[i]}" = 0 ] || KEEP+=($i); done; SEL=("${KEEP[@]}")
  fi
  ask_port

  if (( DRY_RUN )); then
    echo
    echo "Would do (dry run, nothing done):"
    echo "  1. Install what's missing: curl, screen, zstd, Ollama"
    echo "  2. Start Ollama on $BIND:$PORT ($KV_TYPE KV cache, 128K default context, models stay loaded)"
    echo "  3. Access: $(access_text)"
    echo "  4. Pull:"
    for i in "${SEL[@]}"; do
      if (( MCTX[i] )); then echo "       ${REF[i]}  ->  $((MCTX[i]/1024))K context, ~${GB[i]} GB VRAM"
      else echo "       ${REF[i]}  ->  TOO BIG for this card (~${GB[i]} GB at 64K): would spill to the CPU (slow)"; fi
    done
    (( ${#SEL[@]} )) || echo "       (nothing selected)"
    echo
    echo "Endpoint: http://$HOST_HINT:$PORT/v1"
    return 0
  fi

  # --- Start the server detached in screen, default context 128K ---
  stop_llamacpp; stop_ollama; wait_port_free
  apply_access host   # before the port opens
  OLLAMA_HOST=$BIND:$PORT OLLAMA_CONTEXT_LENGTH=$OL_CTX_MAX OLLAMA_KV_CACHE_TYPE="$KV_TYPE" \
    screen -dmS ollama ollama serve
  echo "Waiting for ollama on port $PORT..."
  for i in $(seq 60); do curl -fs --max-time 2 "http://127.0.0.1:$PORT" >/dev/null 2>&1 && break; sleep 1; done
  curl -fs --max-time 2 "http://127.0.0.1:$PORT" >/dev/null 2>&1 || {
    echo "Ollama didn't come up on $PORT after 60 s. Check: screen -r ollama"; exit 1; }

  # Pull selected models. Ones that don't fit at 128K get num_ctx 64K baked in (same name).
  for i in "${SEL[@]}"; do
    OLLAMA_HOST=127.0.0.1:$PORT ollama pull "${REF[i]}"
    if [ "${MCTX[i]}" != "$OL_CTX_MAX" ]; then
      if [[ ${REF[i]} == hf.co/* ]]; then
        echo "${REF[i]}: fits only at 64K here - set num_ctx 65536 in the client instead"
      else
        printf 'FROM %s\nPARAMETER num_ctx %s\n' "${REF[i]}" $OL_CTX_MIN > /tmp/Modelfile.ctx
        OLLAMA_HOST=127.0.0.1:$PORT ollama create "${REF[i]}" -f /tmp/Modelfile.ctx
        echo "${REF[i]}: context set to 64K"
      fi
    fi
  done
  # Load each model once to confirm it really sits in VRAM at its context
  local hdr=1 m
  if (( ${#SEL[@]} )); then
    echo; echo "Checking real VRAM fit (PROCESSOR should be 100% GPU):"
    for i in "${SEL[@]}"; do
      m=${REF[i]}
      curl -s "http://127.0.0.1:$PORT/api/generate" -d "{\"model\":\"$m\",\"keep_alive\":\"1m\"}" >/dev/null || true
      OLLAMA_HOST=127.0.0.1:$PORT ollama ps | awk -v m="$m" -v hdr=$hdr '(NR==1 && hdr) || $1==m'; hdr=0
      curl -s "http://127.0.0.1:$PORT/api/generate" -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null || true
    done
  fi
  echo
  echo "Done. Ollama is serving on $BIND:$PORT ($KV_TYPE KV cache, $((OL_CTX_MAX/1024))K default context, $QUANT_ORDER)"
  for i in "${SEL[@]}"; do echo "  ${REF[i]}  ->  $((MCTX[i] ? MCTX[i]/1024 : 64))K context"; done
  echo "  Base URL:          http://$HOST_HINT:$PORT/v1     api key: any string"
  echo "  Access:            $(access_text)"
  echo "  Attach to server:  screen -r ollama   (detach: Ctrl+A then D)"
  echo "  Run a model:       OLLAMA_HOST=127.0.0.1:$PORT ollama run <model>"
  echo "  Monitor:           btop / bmon"
}

# ======================================================================================================
# Run the chosen engine (each asks for the port last, after its model choices)
# ======================================================================================================
if [ "$ENGINE" = ollama ]; then install_ollama; else install_llamacpp "$ENGINE"; fi
