# How to process documents

You ran `./docling-setup.sh` and the install finished. This guide covers the daily workflow: check the server, convert documents, pick an engine and use the output.

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
| `vlm` | granite-docling-258M on the vLLM server; each page image becomes DocTags | Scans, photos of pages, equations, code blocks, unusual layouts. vLLM is ~4x faster than llama-server. |
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
- vLLM reserves `GPU_UTIL` of the card (0.9 by default), which leaves too little room for the `standard` engine on the same GPU. If you use both, lower `GPU_UTIL` to about 0.4 under menu, Settings, or stop the server (`./docling-setup.sh stop`) during standard runs.
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

The server listens on `127.0.0.1` only, on purpose: it has no API key. To reach it from your laptop, tunnel over SSH:

```bash
ssh -N -L 8000:127.0.0.1:8000 user@gpu-box
docling-convert scans/ --engine vlm      # on the laptop, with docling installed there
```

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA False` in status | Check `nvidia-smi`. If the driver is fine, the CPU torch wheel got installed: under menu, Settings, set CUDA to `cu128`, then reinstall Docling (menu 2, `docling` only). |
| `No vLLM server answering` | `./docling-setup.sh start`. The first start takes 1-2 minutes. Then check `./docling-setup.sh logs`. |
| vLLM: CUDA out of memory at start | Another process holds the GPU (check `nvidia-smi`). Stop it or lower `GPU_UTIL`. |
| Out of memory during `standard` runs | Lower `--batch`, or stop vLLM / lower `GPU_UTIL`. |
| `Address already in use` | Something else is on port 8000. Change `PORT` in Settings and pass `--server http://127.0.0.1:PORT/v1/chat/completions`. |
| Scanned PDF comes out empty | Don't use `--no-ocr`, or use `--engine vlm`. |
| Pages time out with `--engine vlm` | Raise `--timeout` (default 300 s per page) or lower `--batch`. |
| Driver too old | Use driver 570+ (580+ for CUDA 13). On Ubuntu, menu 2, `driver`, then reboot. |
