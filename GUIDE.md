# How to process documents

You ran `./docling-setup.sh` and the install finished. This guide covers the daily workflow: the drop folder, converting by hand, picking an engine and using the output.

## 0. The web page and the drop folder (the main workflow)

**The web page** is the easiest way in. Run `docling-rtx howto` on the server to get the address. Normally that's http://localhost:8080 through `ssh -L 8080:127.0.0.1:8080 ...`, or a direct link with a key after `docling-rtx web public`. Drag documents in, watch the progress bar, then click **json** or **Download all results (.zip)**. The page simply puts files into the inbox below and reads the outbox, so both ways work together. **Cancel** stops the running document within a second or two, or skips a waiting one. It gets `name.error.json` with `"status": "cancelled"` and stays put until you click **convert**. The header shows whether the converter runs on the **GPU** or, in red, the **CPU**. **Clear finished** (or the × on a row) takes done, failed and cancelled documents off the list. Their originals move to `~/docling/archive/`, nothing is deleted, and their JSON stays in the outbox.


Put documents in **`~/docling/inbox`**. A Markdown file for each one appears in **`~/docling/outbox`**, plus a structured JSON file if you tick `json` in the drop-folder menu. That's all there is to it.

```bash
cp contracts/*.pdf prices.csv memo.docx ~/docling/inbox/
ls ~/docling/outbox        # contract-a.md  prices.md  memo.md ...
```

The `docling-watch` service runs this. It starts on boot and scans the inbox every 2 seconds (uploads from the web page start at once), including subfolders, whose layout is mirrored in the outbox. Control it with `./docling-setup.sh watch start|stop|restart|logs`, or from menu 3, where you can also change the folders, output formats and engine.

**What gets converted:** everything Docling reads. That includes PDF, DOCX/DOC, PPTX/PPT, XLSX/XLS, CSV, ODT/ODS/ODP, HTML, Markdown, TXT, EPUB, AsciiDoc, LaTeX, email (`.eml`, `.msg`), calendars (`.ics`), XML (JATS/USPTO/XBRL) and images (PNG, JPG, TIFF, BMP, WEBP, GIF). The watcher ignores hidden files (`.name`), Office lock files (`~$name`) and audio/video.

**Rules for the agent that reads the outbox:**

1. **A `.md` (or `.json`) file in the outbox is always complete.** Files are written under a hidden temp name and then renamed, so the agent never sees a half-written file. Ignore files that start with `.`.
2. **`name.md` = success** (Markdown, the default). With JSON enabled, `name.json` is a [DoclingDocument](https://docling-project.github.io/docling/concepts/docling_document/): text in reading order, headings, lists, tables as cells, page numbers and bounding boxes for PDFs.
3. **`name.error.json` = failure.** It holds `{"source", "status", "errors", "time"}`. A failed file is not retried until it changes. Drop a fixed version under the same name and the error file is removed once it converts.
4. **Re-conversion:** if a source file changes (newer mtime), its JSON is rewritten. Deleting a source does not delete its JSON.
5. **Copying big files in:** a file is converted once it stops changing (unchanged between scans, or untouched for 3 s). If a copy stalls for more than ~3 s, the file may be converted early and then again once complete. For big network copies, the safest way is to copy under a hidden name and then rename: `cp big.pdf inbox/.big.pdf && mv inbox/.big.pdf inbox/big.pdf`.

Reading the JSON, in Python:

```python
from pathlib import Path
from docling_core.types.doc import DoclingDocument

for f in sorted(Path("~/docling/outbox").expanduser().rglob("*.json")):
    if f.name.startswith(".") or f.name.endswith(".error.json"):
        continue
    doc = DoclingDocument.load_from_json(f)
    text = doc.export_to_markdown()                                   # whole document as Markdown
    tables = [t.export_to_dataframe(doc=doc) for t in doc.tables]     # every table as a pandas DataFrame
```

Without docling-core, it's plain JSON. `texts[]` items have `label` (title, section_header, text, list_item, caption...), `text` and `prov[].page_no`. `tables[].data.table_cells[]` have `text`, `start_row_offset_idx` and `start_col_offset_idx`. `body.children` gives the reading order.

**Size:** the JSON is complete, not compact. An 8-page paper is ~290 KB of JSON against ~37 KB of Markdown. If the agent only needs to read the text, add Markdown in menu 3 (output `json` + `md`). The agent can then read `name.md` and look up `name.json` for exact tables and page positions.

## 1. Check it's live

```bash
./docling-setup.sh status          # GPU, Docling/torch versions, CUDA true/false, vLLM up/down
curl -s http://127.0.0.1:8000/v1/models   # vLLM answers with ibm-granite/granite-docling-258M
```

`CUDA True` means Docling's own models run on the GPU. `Server: UP` means the granite-docling VLM is being served by vLLM. If the server is down, run `./docling-setup.sh start`. If it won't start, run `./docling-setup.sh logs`.

Smoke test: `./docling-setup.sh test` converts the Docling paper (arXiv 2501.17887) with both engines and writes the results to `~/.docling-rtx/test-output/`.

## 2. Convert documents

`docling-convert` takes files, folders (searched recursively) and URLs:

```bash
docling-convert report.pdf                       # -> ./out/report.md
docling-convert ~/inbox -o ~/converted           # every supported file, folder layout kept
docling-convert https://arxiv.org/pdf/2501.17887 # URLs work too
docling-convert ~/inbox --to md --to json        # several formats at once
docling-convert scans/ --engine vlm              # through granite-docling on vLLM
```

Formats (`--to`, repeatable): `md` (default), `json` (lossless DoclingDocument, reloadable), `html`, `txt`, `doctags`.

Supported inputs: PDF, DOCX, PPTX, XLSX, HTML, Markdown, CSV, AsciiDoc, ODF, EPUB and images (PNG, JPG, TIFF, BMP, WEBP).

Reruns skip documents whose outputs already exist. That makes large batches resumable: if a run dies, start it again. Use `--overwrite` to redo everything.

Each document prints `ok`, `PART` (some pages failed) or `FAIL` with the reason. The batch keeps going, and the exit code is 1 if anything failed.

## 3. Pick an engine

| `--engine` | What runs | Use it for |
|---|---|---|
| `standard` (default) | Docling's layout, TableFormer and OCR models on the GPU | Born-digital PDFs, Office files, HTML. Most accurate tables, and fastest when the PDF already has a text layer. |
| `vlm` | granite-docling-258M on the vLLM server (installed on first use: `docling-rtx start`, or pick `vlm` in menu 3); each page image becomes DocTags | Scans, photos of pages, equations, code blocks, unusual layouts. vLLM is ~4x faster than llama-server. |
| `local` | granite-docling loaded in-process (transformers) | No server running, or a small one-off job. |

If you're unsure, run both on a few typical files and compare the Markdown:

```bash
docling-convert sample/ -o cmp/standard
docling-convert sample/ -o cmp/vlm --engine vlm
diff -r cmp/standard cmp/vlm | less
```

Office, HTML and Markdown files skip the page models entirely, whatever the engine.

## 4. Speed and GPU memory

- `--batch N` sets how many pages are in flight. It defaults to 32 (`DOCLING_BATCH` env). Docling's guidance: RTX 5090 (32 GB) 64-128, RTX 4090 (24 GB) 32-64, 12 GB cards 16-32. For `--engine vlm`, a higher batch means more parallel requests to vLLM, which is where the throughput comes from.
- vLLM's settings follow the card's memory (`GPU_UTIL=auto`). On 20 GB and up it uses the RTX guide's 512 parallel pages and takes 0.5 of the card, leaving room for the drop folder's `standard` engine. 10-20 GB cards get 128 pages. Under 10 GB (e.g. a 6 GB GTX 1660 Ti) it uses 32 pages, 4096-token batches and 0.7 of the card, and float16 on cards older than RTX 30. On a small card both engines barely fit together, so pick one: stop vLLM (`./docling-setup.sh stop`) if the drop folder uses `standard`, or switch the drop folder to `vlm`. If you only use `--engine vlm`, set `GPU_UTIL` to 0.9 under menu, Settings.
- `--no-ocr` skips OCR for clean digital PDFs. That's faster, but scanned pages come out empty.
- Watch the GPU with `nvidia-smi -l 1`. Low utilization during a big batch means you should raise `--batch`.

## 5. Use the output

**Markdown** is what you feed to an LLM or a wiki. **JSON** is the full DoclingDocument with pages, bounding boxes, tables and reading order. Reload it with:

```python
from docling_core.types.doc import DoclingDocument
doc = DoclingDocument.load_from_json("out/report.json")
for table in doc.tables:
    print(table.export_to_dataframe(doc=doc))   # pandas DataFrame per table
```

**Chunks for RAG** come from the Docling CLI. Each chunk carries its headings and source items:

```bash
docling report.pdf --to chunks --output chunks/   # -> chunks/report.chunks.jsonl, one JSON chunk per line
```

**Pictures**: `--images` embeds page pictures in the `md`/`html`/`json` output.

## 6. From Python

Use the venv's Python: `~/.docling-rtx/venv/bin/python`.

Standard pipeline on the GPU:

```python
from docling.document_converter import DocumentConverter
doc = DocumentConverter().convert("report.pdf").document   # CUDA is picked automatically
print(doc.export_to_markdown())
```

VLM pipeline through the vLLM server (the same code `docling-convert --engine vlm` runs):

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.datamodel.settings import settings
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.inference_engines.vlm.base import VlmEngineType
from docling.pipeline.vlm_pipeline import VlmPipeline

BATCH = 64
engine = ApiVlmEngineOptions(engine_type=VlmEngineType.API, url="http://127.0.0.1:8000/v1/chat/completions",
                             params={"model": "ibm-granite/granite-docling-258M"}, concurrency=BATCH, timeout=300)
opts = VlmPipelineOptions(vlm_options=VlmConvertOptions.from_preset("granite_docling", engine_options=engine),
                          enable_remote_services=True)
settings.perf.page_batch_size = BATCH
converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_cls=VlmPipeline, pipeline_options=opts)})
print(converter.convert("scan.pdf").document.export_to_markdown())
```

The plain Docling CLI is installed as `docling` too, for example `docling --pipeline vlm --vlm-model granite_docling file.pdf` for the in-process VLM.

## 7. Use it from another machine

**The public link (default).** The install starts a Cloudflare quick tunnel, and `docling-rtx web url` prints `https://<random>.trycloudflare.com/?token=...`. Open it anywhere. The key in the link is the lock: anyone who has the whole link can use the page, and without the key the page answers 403. The address changes when the tunnel restarts (a reboot, or `docling-rtx web tunnel on`), and the page shows the current one with a copy button. Uploads through the tunnel are limited to 100 MB per file, so copy bigger files into the inbox with scp. `docling-rtx web tunnel off` removes the public link.

**SSH tunnel (private).** With the public link off, the page listens on the server only:

```bash
ssh -N -L 8080:127.0.0.1:8080 user@gpu-box     # then open http://localhost:8080
```

The VLM server (vLLM) never leaves 127.0.0.1. It has no key, so reach it through SSH: `ssh -N -L 8000:127.0.0.1:8000 user@gpu-box`, then `docling-convert scans/ --engine vlm` on your laptop.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA False` in status | Check `nvidia-smi`. If the driver is fine, the CPU torch wheel got installed: under menu, Settings, set CUDA to `cu128`, then reinstall Docling (menu 2, `docling` only). |
| `No vLLM server answering` | `./docling-setup.sh start`. The first start takes 1-2 minutes. Then check `./docling-setup.sh logs`. |
| vLLM: CUDA out of memory at start, or `No available memory for the cache blocks` | Another process holds the GPU (check `nvidia-smi`), or `GPU_UTIL` is too low for the card. Free the GPU, or raise `GPU_UTIL` (e.g. 0.8) under Settings. |
| Web page says **converter on CPU (slow)**, or a big PDF sits at "page 0" with the GPU at 0% | PyTorch in the Docling venv can't use CUDA, usually because the build is newer than the driver. Run `docling-rtx status`: if it shows `CUDA False`, run `nvidia-smi` for the driver version, set CUDA to match under menu, Settings (cu128 for driver 570+, cu126 for 560+), then reinstall with `docling-rtx install`. |
| Nothing appears in the outbox | `./docling-setup.sh watch logs`. Is the file type supported? Is it a hidden file? Is the watcher up (`status`)? |
| Out of memory during `standard` runs | Lower `--batch`, or stop vLLM / lower `GPU_UTIL`. |
| `Address already in use` | Something else is on port 8000. Change `PORT` in Settings and pass `--server http://127.0.0.1:PORT/v1/chat/completions`. |
| Scanned PDF comes out empty | Don't use `--no-ocr`, or use `--engine vlm`. |
| Pages time out with `--engine vlm` | Raise `--timeout` (default 300 s per page) or lower `--batch`. |
| Driver too old | Use driver 570+ (580+ for CUDA 13). On Ubuntu, menu 2, `driver`, then reboot. |
