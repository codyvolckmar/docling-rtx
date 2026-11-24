#!/usr/bin/env python3
"""Batch-convert documents with Docling on an NVIDIA GPU.

Inputs: files, folders (searched recursively) or URLs. Outputs one file per format per document,
mirroring the input folder layout under --out. Outputs are written atomically (temp file, then rename), so a
reader never sees a half-written file. A document that fails gets <name>.error.json instead.

Drop-folder mode (--watch): keep running, convert every new or changed file that lands in the folder, e.g.
  docling-convert --watch ~/docling/inbox -o ~/docling/outbox --to json

Engines (--engine):
  standard  Docling's layout + table + OCR models on the GPU. Best for born-digital PDFs, Office files, HTML.
  vlm       granite-docling-258M served by the local vLLM server (docling-setup.sh starts it). Fastest VLM option,
            best for scans, formulas, code blocks and odd layouts.
  local     granite-docling-258M loaded inside this process (transformers). No server needed, slower.

Examples:
  python convert.py ~/inbox                       # every supported file in ~/inbox -> ./out as Markdown
  python convert.py --engine vlm --to md --to json scans/ -o out/
  python convert.py https://arxiv.org/pdf/2501.17887 --engine vlm
  python convert.py --watch ~/docling/inbox -o ~/docling/outbox --to json
"""

import argparse
import datetime
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

# Every input Docling reads, from Docling's own list, minus: its own JSON (outputs would loop back in), audio/video
# (need the ASR pipeline) and METS tarballs.
SKIP_FORMATS = {"json_docling", "audio", "video", "mets_gbs"}
IGNORE_PREFIXES = (".", "~$")   # hidden files, temp files, Office lock files


def suffixes():
    from docling.datamodel.base_models import FormatToExtensions
    return {"." + e.lower() for fmt, exts in FormatToExtensions.items() if fmt.value not in SKIP_FORMATS for e in exts}


WRITERS = {
    "md": ("md", lambda d, p: d.save_as_markdown(p)),
    "json": ("json", lambda d, p: d.save_as_json(p)),
    "html": ("html", lambda d, p: d.save_as_html(p)),
    "txt": ("txt", lambda d, p: p.write_text(d.export_to_text(), encoding="utf-8")),
    "doctags": ("doctags", lambda d, p: p.write_text(d.export_to_doctags(), encoding="utf-8")),
}


def collect(inputs, out):
    """Return (source, relative output stem) pairs. Skips hidden/temp files and anything inside the output folder."""
    sfx, out_abs, items = suffixes(), out.resolve(), []
    for raw in inputs:
        if raw.startswith(("http://", "https://")):
            name = Path(raw.split("?")[0].rstrip("/")).name or "download"
            items.append((raw, Path(name[:-4] if name.lower().endswith(".pdf") else name)))
            continue
        p = Path(raw).expanduser()
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if (f.is_file() and f.suffix.lower() in sfx and not f.name.startswith(IGNORE_PREFIXES)
                        and out_abs not in f.resolve().parents):
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
    ap.add_argument("--watch", action="store_true",
                    help="drop-folder mode: keep running and convert new or changed files as they arrive")
    ap.add_argument("--interval", type=float, default=float(os.environ.get("DOCLING_WATCH_INTERVAL", "5")),
                    help="--watch: seconds between folder scans; a file must be unchanged for one scan (default 5)")
    a = ap.parse_args()
    formats = a.to or ["md"]
    out = Path(a.out).expanduser()
    if a.watch:
        watch(a, out, formats)
        return

    items = collect(a.inputs, out)
    todo = [(s, r) for s, r in items if a.overwrite or not is_done(s, r, out, formats)]
    if not todo:
        print(f"Nothing to do: {len(items)} document(s) already converted in {out} (--overwrite to redo).")
        return
    print(f"{len(todo)} document(s) -> {out}  [engine: {a.engine}, formats: {', '.join(formats)}]"
          + (f", {len(items) - len(todo)} already done" if len(items) > len(todo) else ""))
    t0 = time.time()
    ok, failed, pages = convert_items(build_converter(a), todo, out, a, formats)
    dt = time.time() - t0
    print(f"Done: {ok} converted, {failed} failed, {pages} pages in {dt:.1f}s"
          + (f" ({pages / dt:.2f} pages/s)" if dt > 0 and pages else "") + f". Output: {out}")
    sys.exit(1 if failed else 0)


def is_done(src, rel, out, formats, skip_failed=False):
    """True if every output exists and is newer than the source (so an edited file is converted again).
    skip_failed: also True if the last attempt on this exact version failed (watch mode doesn't retry in a loop)."""
    newest = 0 if isinstance(src, str) else src.stat().st_mtime
    outs = [target(out, rel, WRITERS[f][0]) for f in formats]
    if all(o.exists() and o.stat().st_mtime >= newest for o in outs):
        return True
    err = target(out, rel, "error.json")
    return skip_failed and err.exists() and err.stat().st_mtime >= newest


def write_atomic(dst, write):
    """Write via a hidden temp file in the same folder, then rename: readers see the whole file or nothing."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.stem}.tmp{dst.suffix}")
    try:
        write(tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def convert_items(conv, todo, out, a, formats):
    from docling.datamodel.base_models import ConversionStatus
    from docling_core.types.doc import ImageRefMode

    ok = failed = pages = 0
    for src, rel in todo:
        name = src if isinstance(src, str) else src.name
        err_path = target(out, rel, "error.json")
        try:
            res = conv.convert(src, raises_on_error=False)
            errors = [e.error_message for e in res.errors]
            good = res.status in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS)
            status = res.status.value
        except Exception as e:   # a crash on one file must not stop the batch (or the watcher)
            res, good, status, errors = None, False, "failure", [f"{type(e).__name__}: {e}"]
        if not good:
            failed += 1
            print(f"  FAIL  {name}: {'; '.join(errors) or status}", flush=True)
            report = {"source": str(src), "status": status, "errors": errors,
                      "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
            write_atomic(err_path, lambda p: p.write_text(json.dumps(report, indent=2), encoding="utf-8"))
            continue
        doc = res.document
        for f in formats:
            ext, write = WRITERS[f]
            if a.images and f in ("md", "html", "json"):
                save = {"md": doc.save_as_markdown, "html": doc.save_as_html, "json": doc.save_as_json}[f]
                write_atomic(target(out, rel, ext), lambda p: save(p, image_mode=ImageRefMode.EMBEDDED))
            else:
                write_atomic(target(out, rel, ext), lambda p: write(doc, p))
        err_path.unlink(missing_ok=True)
        n = len(res.pages) if res.pages else doc.num_pages()
        pages += n
        ok += 1
        tag = "PART " if res.status == ConversionStatus.PARTIAL_SUCCESS else "ok   "
        print(f"  {tag} {name}" + (f"  ({n} page{'s' if n != 1 else ''})" if n else ""), flush=True)
    return ok, failed, pages


def watch(a, out, formats):
    """Poll the input folders; convert a file once its size and mtime stayed the same for one scan (so a file
    that is still being copied in isn't read half-way). Models load once, on the first file."""
    for raw in a.inputs:   # created if missing, so a fresh setup can start watching an empty inbox
        if raw.startswith(("http://", "https://")) or Path(raw).expanduser().is_file():
            sys.exit(f"--watch needs folders: {raw} is not one")
        Path(raw).expanduser().mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Watching {', '.join(a.inputs)} -> {out}  [engine: {a.engine}, formats: {', '.join(formats)}, "
          f"every {a.interval:g}s]. Ctrl+C to stop.", flush=True)
    conv, seen = None, {}
    while True:
        ready = []
        for src, rel in collect(a.inputs, out):
            try:
                if is_done(src, rel, out, formats, skip_failed=True):
                    seen.pop(src, None)
                    continue
                st = src.stat()
            except FileNotFoundError:   # removed while we looked
                continue
            key = (st.st_size, st.st_mtime_ns)
            if seen.get(src) == key:
                ready.append((src, rel))
            else:
                seen[src] = key
        if ready:
            conv = conv or build_converter(a)
            print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {len(ready)} new document(s)", flush=True)
            convert_items(conv, ready, out, a, formats)
            for src, _ in ready:
                seen.pop(src, None)
        time.sleep(a.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
