#!/usr/bin/env python3
"""Batch-convert documents with Docling on an NVIDIA GPU.

Inputs: files, folders (searched recursively) or URLs. Outputs one file per format per document,
mirroring the input folder layout under --out.

Engines (--engine):
  standard  Docling's layout + table + OCR models on the GPU. Best for born-digital PDFs, Office files, HTML.
  vlm       granite-docling-258M served by the local vLLM server (docling-setup.sh starts it). Fastest VLM option,
            best for scans, formulas, code blocks and odd layouts.
  local     granite-docling-258M loaded inside this process (transformers). No server needed, slower.

Examples:
  python convert.py ~/inbox                       # every supported file in ~/inbox -> ./out as Markdown
  python convert.py --engine vlm --to md --to json scans/ -o out/
  python convert.py https://arxiv.org/pdf/2501.17887 --engine vlm
"""

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".md", ".csv", ".png", ".jpg", ".jpeg",
            ".tif", ".tiff", ".bmp", ".webp", ".asciidoc", ".adoc", ".odt", ".ods", ".odp", ".epub"}
WRITERS = {
    "md": ("md", lambda d, p: d.save_as_markdown(p)),
    "json": ("json", lambda d, p: d.save_as_json(p)),
    "html": ("html", lambda d, p: d.save_as_html(p)),
    "txt": ("txt", lambda d, p: p.write_text(d.export_to_text(), encoding="utf-8")),
    "doctags": ("doctags", lambda d, p: p.write_text(d.export_to_doctags(), encoding="utf-8")),
}


def collect(inputs):
    """Return (source, relative output stem) pairs."""
    items = []
    for raw in inputs:
        if raw.startswith(("http://", "https://")):
            name = Path(raw.split("?")[0].rstrip("/")).name or "download"
            items.append((raw, Path(name[:-4] if name.lower().endswith(".pdf") else name)))
            continue
        p = Path(raw).expanduser()
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUFFIXES:
                    rel = f.relative_to(p)
                    items.append((f, rel.parent / rel.stem))
        elif p.is_file():
            items.append((p, Path(p.stem)))
        else:
            sys.exit(f"Not found: {raw}")
    return items


def target(out, rel, ext):
    return out / rel.parent / f"{rel.name}.{ext}"   # not with_suffix: "report.v2" must stay "report.v2.md"


def server_ok(url):
    base = url.split("/v1/")[0]
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def build_converter(a):
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.settings import settings
    from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

    device = AcceleratorDevice.AUTO if a.device == "auto" else AcceleratorDevice(a.device)
    accel = AcceleratorOptions(device=device)

    if a.engine == "standard":
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        opts = PdfPipelineOptions(accelerator_options=accel, do_ocr=not a.no_ocr, do_table_structure=True)
        opts.ocr_batch_size = opts.layout_batch_size = a.batch
        opts.table_batch_size = max(1, a.batch // 16)
        if a.images:
            opts.generate_picture_images = True
            opts.images_scale = 2.0
        fmt = {InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
               InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts)}
        return DocumentConverter(format_options=fmt)

    from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
    from docling.pipeline.vlm_pipeline import VlmPipeline

    if a.engine == "vlm":
        from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
        from docling.models.inference_engines.vlm.base import VlmEngineType
        if not server_ok(a.server):
            sys.exit(f"No vLLM server answering at {a.server}. Start it: ./docling-setup.sh start "
                     f"(or use --engine standard / --engine local).")
        engine = ApiVlmEngineOptions(
            engine_type=VlmEngineType.API, url=a.server, timeout=a.timeout, concurrency=a.batch,
            params={"model": a.model})
        vlm = VlmConvertOptions.from_preset("granite_docling", engine_options=engine)
        settings.perf.page_batch_size = a.batch   # keep enough pages in flight to fill the server
        opts = VlmPipelineOptions(vlm_options=vlm, enable_remote_services=True, accelerator_options=accel)
    else:  # local
        vlm = VlmConvertOptions.from_preset("granite_docling")
        opts = VlmPipelineOptions(vlm_options=vlm, accelerator_options=accel)
    if a.images:
        opts.generate_picture_images = True
    fmt = {InputFormat.PDF: PdfFormatOption(pipeline_cls=VlmPipeline, pipeline_options=opts),
           InputFormat.IMAGE: ImageFormatOption(pipeline_cls=VlmPipeline, pipeline_options=opts)}
    return DocumentConverter(format_options=fmt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="files, folders or URLs")
    ap.add_argument("-o", "--out", default="out", help="output folder (default: ./out)")
    ap.add_argument("--to", action="append", choices=sorted(WRITERS), help="output format, repeatable (default: md)")
    ap.add_argument("--engine", choices=["standard", "vlm", "local"], default=os.environ.get("DOCLING_ENGINE", "standard"))
    ap.add_argument("--server", default=os.environ.get("DOCLING_VLM_URL", "http://127.0.0.1:8000/v1/chat/completions"))
    ap.add_argument("--model", default=os.environ.get("DOCLING_VLM_MODEL", "ibm-granite/granite-docling-258M"),
                    help="model name the server answers to (vLLM: the default)")
    ap.add_argument("--batch", type=int, default=int(os.environ.get("DOCLING_BATCH", "32")),
                    help="pages in flight / model batch size (RTX 5090: 64-128, 4090: 32-64, 12 GB: 16-32)")
    ap.add_argument("--timeout", type=float, default=300, help="seconds per page request (vlm engine)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--no-ocr", action="store_true", help="standard engine: skip OCR (born-digital PDFs only)")
    ap.add_argument("--images", action="store_true", help="keep picture images (embedded in md/html/json)")
    ap.add_argument("--overwrite", action="store_true", help="redo documents whose outputs already exist")
    a = ap.parse_args()
    formats = a.to or ["md"]

    items = collect(a.inputs)
    out = Path(a.out).expanduser()
    todo = [(s, r) for s, r in items
            if a.overwrite or not all(target(out, r, WRITERS[f][0]).exists() for f in formats)]
    if not todo:
        print(f"Nothing to do: {len(items)} document(s) already converted in {out} (--overwrite to redo).")
        return
    print(f"{len(todo)} document(s) -> {out}  [engine: {a.engine}, formats: {', '.join(formats)}]"
          + (f", {len(items) - len(todo)} already done" if len(items) > len(todo) else ""))

    from docling.datamodel.base_models import ConversionStatus
    from docling_core.types.doc import ImageRefMode

    conv = build_converter(a)
    ok = failed = pages = 0
    t0 = time.time()
    results = conv.convert_all([s for s, _ in todo], raises_on_error=False)
    for (src, rel), res in zip(todo, results):
        name = getattr(src, "name", src)
        if res.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
            failed += 1
            errs = "; ".join(e.error_message for e in res.errors) or res.status.value
            print(f"  FAIL  {name}: {errs}")
            continue
        doc = res.document
        for f in formats:
            ext, write = WRITERS[f]
            dst = target(out, rel, ext)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if a.images and f in ("md", "html", "json"):
                {"md": doc.save_as_markdown, "html": doc.save_as_html, "json": doc.save_as_json}[f](
                    dst, image_mode=ImageRefMode.EMBEDDED)
            else:
                write(doc, dst)
        n = len(res.pages) if res.pages else doc.num_pages()
        pages += n
        ok += 1
        tag = "PART " if res.status == ConversionStatus.PARTIAL_SUCCESS else "ok   "
        print(f"  {tag} {name}  ({n} page{'s' if n != 1 else ''})")
    dt = time.time() - t0
    print(f"Done: {ok} converted, {failed} failed, {pages} pages in {dt:.1f}s"
          + (f" ({pages / dt:.2f} pages/s)" if dt > 0 and pages else "") + f". Output: {out}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
