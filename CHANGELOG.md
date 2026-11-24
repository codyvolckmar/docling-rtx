# Changelog

All notable changes to Docling RTX. Install time on a fresh pod: **about 12 minutes**, mostly downloading CUDA PyTorch and Docling's models. Updating reruns the installer, which reuses everything already downloaded: `cd ~/docling-rtx && git pull && ./docling-setup.sh install`.

## 1.4.2 (2026-11-24): status filter

- **Filter the document table by status:** All, Converting, Waiting, Done, Failed and Cancelled chips with live counts. Empty ones are greyed out, and the choice is kept across reloads.

## 1.4.1 (2026-11-24): folder uploads

- **Upload whole folders.** Drag a folder onto the page, or use **choose a folder**. Every file inside is uploaded with its subfolders, and the outbox mirrors the layout. Before, a dropped folder failed with "Upload failed: <folder> (0)".
- Files Docling can't read (for example `.mobi`) are skipped before upload, and listed in one summary instead of an error per file. Hidden files (`.DS_Store`) are ignored.
- One progress line for the batch ("Uploading 3 of 12: ..."), then a summary of uploaded, skipped and failed files.
- The server checks every part of an uploaded path, rejects anything that would land outside the inbox, and accepts only formats Docling reads.
- Pages and time are shown per file, even when two folders contain files with the same name.

## 1.4.0 (2026-11-24): public link

- **Cloudflare quick tunnel, on by default.** When the web page launches, `cloudflared` gives it a public `https://<random>.trycloudflare.com` address. You need no Cloudflare account and no open port on the pod. Anyone with the link and its key can use the page. Without the key it answers 403, and the key is now required even from localhost. The page shows the current link with a **copy** button, since the address changes when the tunnel restarts. `docling-rtx web url` prints it too.
- `docling-rtx web tunnel on|off|logs`. The Settings menu has three access modes: public link (tunnel), SSH tunnel only, or a direct `http://IP:PORT` link with a key.
- Uploads over 100 MB are stopped with an explanation when you use the tunnel, because that is Cloudflare's per-request limit. Copy big files into the inbox instead.
- The tunnel runs as the `docling-tunnel` service, or as a background process on pods without systemd. It restarts if it crashes and comes back at login, like the other services.

## 1.3.0 (2026-11-24): code review fixes, faster setup

Setup
- **vLLM installs on first use**, not by default: `docling-rtx start`, choosing the `vlm` engine in menu 3, or `WITH_VLM=1` at install. This saves ~12 GB and several minutes when you only use the standard engine.
- NVIDIA drivers 525–559, common on rented pods (535/550), use the CUDA 12.6 build instead of aborting the install.
- apt is skipped when its packages are already present.
- A warm-up conversion at the end of the install downloads Docling's models, so the first document starts right away.
- vLLM's GPU share is 0.85 when the drop folder uses the vlm engine, and 0.5 when it shares the card with the standard pipeline.

Robustness on rented pods
- **Without systemd:** a service that can't start is reported instead of shown as UP, for example a web page whose port Jupyter already holds. A service you stopped stays stopped, and autostart at login takes a lock, so two terminals opening together start one converter, not two.
- Changing the vLLM port or memory in Settings restarts the running vLLM, never a second one.
- RunPod: the public port comes from `RUNPOD_TCP_PORT_<port>`, and the HTTPS proxy link is offered.
- Local traffic never goes through an `http_proxy`.

Web page and converter
- Uploads and retries start at once: the page tells the converter the file is ready. Upload to JSON takes 0.1–0.9 s instead of 5–10 s, and the inbox scan runs every 2 s instead of 5 s.
- File names with `#`, `?`, `&`, `'`, `+` or accents work for download, cancel, retry and remove.
- A cancel that lands as a document finishes can no longer leave the next document empty.
- Downloads and "Download all (.zip)" are streamed instead of held in memory.

## 1.2.0 (2026-11-24): easy mode

- **One-line installer:** `curl -fsSL https://raw.githubusercontent.com/codyvolckmar/docling-rtx/main/install.sh | bash`.
- **Web page:** drag & drop upload, live page-by-page progress with time left, downloads, a zip of all results, and plain-English errors with **retry**.
- **Cancel** a running document (it stops within seconds and the models stay loaded) or a waiting one.
- **Clear finished** and per-row remove: the originals move to `~/docling/archive/`, and the JSON stays in the outbox.
- A live **CPU / RAM / GPU / VRAM** widget, measured against the container's own limits on pods, plus a "converter on GPU / CPU" badge.
- The **`docling-rtx`** command (menu, `status`, `howto`, `web`, `watch`, ...), also linked into `/usr/local/bin` when installed as root.
- **Web port** at install (`WEB_PORT=... WEB_PUBLIC=1`) or later (`docling-rtx web port <port> [public port]`). On Vast.ai the public port mapping is detected.
- Services restart if they crash and come back at login on pods without systemd.
- Logo, in the page header and as the tab icon.
- Fixes: the menu installs `whiptail` itself on minimal images. vLLM uses uv's own Python, which includes `Python.h`. vLLM is sized to the card (the 6 GB GTX 1660 Ti failed to start). The PyTorch CUDA build is pinned. The VLM badge follows a changed vLLM port.

## 1.1.0 (2026-11-24): drop folder

- `~/docling/inbox` → `~/docling/outbox`: every file put in the inbox becomes structured JSON (a DoclingDocument), written atomically, or `name.error.json` with the reason. Changed files are converted again, and failures aren't retried in a loop.
- Every format Docling reads: PDF, Word, Excel, PowerPoint, CSV, HTML, Markdown, images, EPUB, email, LaTeX and more.
- The `docling-watch` service, and a drop-folder menu (folders, output formats, engine).

## 1.0.0 (2026-11-24): first version

- Forked from [Summon](https://github.com/codyvolckmar/summon): a whiptail menu that installs Docling on CUDA PyTorch, plus granite-docling-258M on vLLM, following Docling's RTX guide.
- `convert.py` (`docling-convert`): batch conversion of files, folders and URLs with three engines (standard, vlm, local).
- `GUIDE.md`: how to process documents.
