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
import threading
import time
import urllib.parse
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


class Progress:
    """Live progress for long documents: a log line every 10% (rate, time left) and, with --status-file, a small
    JSON file (current document, pages done/total, queue, recent results) that web.py shows. Also the callback
    Docling calls after every page."""

    def __init__(self, path=None):
        self.path = Path(path).expanduser() if path else None
        # Cancel requests (from web.py): one empty file per document, named after its inbox path, in <status dir>/cancel
        self.cancel_dir = self.path.parent / "cancel" if self.path else None
        self.opts = None            # the pipeline options of the running converter (cancel lowers their timeout)
        self.key = None             # inbox path of the document being converted
        self.cancelled = False
        self.on_hard_cancel = None
        self.lock = threading.Lock()
        self.s = {"state": "idle", "file": None, "pages_done": 0, "pages_total": 0, "started": None,
                  "queue": [], "recent": []}
        self.logged_tenth, self.first = 0, None
        self.written = 0.0
        if self.path and self.path.exists():   # keep the recent results across restarts
            try:
                self.s["recent"] = json.loads(self.path.read_text())["recent"]
            except Exception:
                pass
        if self.cancel_dir:
            threading.Thread(target=self._cancel_loop, daemon=True).start()

    def _cancel_file(self, key):
        return self.cancel_dir / urllib.parse.quote(key, safe="") if self.cancel_dir else None

    def cancel_requested(self, key):
        f = self._cancel_file(key)
        return bool(f and f.exists())

    def clear_cancel(self, key):
        f = self._cancel_file(key)
        if f:
            f.unlink(missing_ok=True)

    def _cancel_loop(self):
        """Stop the running document when its cancel file appears. Soft first: Docling's own per-document timeout,
        lowered to ~0, ends the conversion within a second or so and keeps the models loaded. If it hasn't stopped
        after 30 s (a model call that never returns), exit: the service (or the restart loop) starts a fresh one."""
        asked = None
        while True:
            time.sleep(0.5)
            key = self.key
            if not key or not self.cancel_requested(key):
                asked = None
                continue
            if not self.cancelled:
                self.cancelled, asked = True, time.time()
                print(f"  ...   cancelling {key}", flush=True)
                if self.opts is not None:
                    self.opts.document_timeout = 0.001
            elif asked and time.time() - asked > 30 and self.key == key:
                print(f"  ...   {key} didn't stop within 30 s: restarting the converter", flush=True)
                if self.on_hard_cancel:
                    self.on_hard_cancel()
                os._exit(75)

    def write(self, force=True):
        if not self.path or (not force and time.time() - self.written < 1):
            return
        self.written = self.s["updated"] = time.time()
        data = json.dumps(self.s)
        write_atomic(self.path, lambda p: p.write_text(data, encoding="utf-8"))

    def queue(self, names):
        with self.lock:
            self.s["queue"] = names
            self.write()

    def start(self, name, queue, key=None, pages_total=0):
        with self.lock:
            self.key, self.cancelled = key, False
            self.s.update(state="converting", file=name, key=key, pages_done=0, pages_total=pages_total,
                          started=time.time(), queue=queue)
            self.logged_tenth, self.first = 0, None
            self.write()

    def __call__(self, ev):   # Docling's progress_callback
        if type(ev).__name__ != "PageCompletedProgress" or self.cancelled:
            return   # after a cancel Docling reports the abandoned pages as completed: not progress
        with self.lock:
            s = self.s
            s["pages_done"], s["pages_total"] = ev.completed_pages, ev.total_pages
            self.write(force=False)
            tenth = ev.completed_pages * 10 // max(ev.total_pages, 1)
            if self.first is None:   # rate from the first finished page on: model loading would skew it
                self.first = (time.time(), ev.completed_pages)
            if ev.total_pages >= 20 and tenth > self.logged_tenth and ev.completed_pages < ev.total_pages:
                self.logged_tenth = tenth
                dt, dp = time.time() - self.first[0], ev.completed_pages - self.first[1]
                rate = dp / dt if dt > 0 and dp > 0 else ev.completed_pages / max(time.time() - s["started"], 1e-6)
                left = (ev.total_pages - ev.completed_pages) / rate if rate else 0
                print(f"        {s['file']}: {ev.completed_pages}/{ev.total_pages} pages, {rate:.2f} pages/s, "
                      f"~{fmt_secs(left)} left", flush=True)

    def done(self, name, status, pages, secs, error=None):
        with self.lock:
            self.key = None
            item = {"file": name, "status": status, "pages": pages, "seconds": round(secs, 1),
                    "finished": time.time(), "error": error}
            self.s["recent"] = [item] + [r for r in self.s["recent"] if r["file"] != name][:49]
            self.s.update(state="idle", file=None, key=None, pages_done=0, pages_total=0, started=None)
            self.write()


def fmt_secs(t):
    t = int(t)
    return f"{t // 3600}h{t % 3600 // 60:02d}m" if t >= 3600 else f"{t // 60}m{t % 60:02d}s" if t >= 60 else f"{t}s"


def page_count(src):
    """Pages of a PDF, read in milliseconds (so the progress shows "page 0 of 412" right away); 0 if unknown."""
    if isinstance(src, str) or src.suffix.lower() != ".pdf":
        return 0
    try:
        import pypdfium2
        doc = pypdfium2.PdfDocument(str(src))
        try:
            return len(doc)
        finally:
            doc.close()
    except Exception:
        return 0


def build_converter(a, progress=None):
    conv, opts = _build_converter(a, progress)
    if progress is not None:
        progress.opts = opts
        try:
            import torch
            progress.s["device"] = ("cuda" if a.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
            if progress.s["device"] == "cuda":
                progress.s["device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            progress.s["device"] = "cpu"
        print(f"Converter runs on: {progress.s['device'].upper()}" + (
            "  (no CUDA: much slower; check 'docling-rtx status')" if progress.s["device"] == "cpu" else ""), flush=True)
    return conv


def _build_converter(a, progress=None):
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
        return DocumentConverter(format_options=fmt, progress_callback=progress), opts

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
    return DocumentConverter(format_options=fmt, progress_callback=progress), opts


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
    ap.add_argument("--status-file", default=os.environ.get("DOCLING_STATUS_FILE"),
                    help="write live progress (current document, pages, recent results) to this JSON file")
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
    progress = Progress(a.status_file)
    ok, failed, pages = convert_items(build_converter(a, progress), todo, out, a, formats, progress)
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


def convert_items(conv, todo, out, a, formats, progress):
    from docling.datamodel.base_models import ConversionStatus
    from docling_core.types.doc import ImageRefMode

    ok = failed = pages = 0
    def cancelled(src, name, key, err_path, t0):
        report = {"source": str(src), "status": "cancelled", "errors": ["Cancelled (web page)"],
                  "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
        write_atomic(err_path, lambda p: p.write_text(json.dumps(report, indent=2), encoding="utf-8"))
        progress.clear_cancel(key)
        progress.done(name, "cancelled", 0, time.time() - t0, "Cancelled")
        print(f"  STOP  {name}: cancelled", flush=True)

    for i, (src, rel) in enumerate(todo):
        name = src if isinstance(src, str) else src.name
        key = name if isinstance(src, str) else str(rel.parent / src.name)   # its path inside the inbox
        err_path = target(out, rel, "error.json")
        t0 = time.time()
        if progress.cancel_requested(key):   # cancelled while it was waiting
            cancelled(src, name, key, err_path, t0)
            continue
        progress.start(name, [s if isinstance(s, str) else s.name for s, _ in todo[i + 1:]], key, page_count(src))
        progress.on_hard_cancel = lambda: cancelled(src, name, key, err_path, t0)
        if len(todo) > 1 or a.watch:
            print(f"  ...   {name}" + (f"  ({progress.s['pages_total']} pages)" if progress.s["pages_total"] else ""),
                  flush=True)
        try:
            res = conv.convert(src, raises_on_error=False)
            errors = [e.error_message for e in res.errors]
            good = res.status in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS)
            status = res.status.value
        except Exception as e:   # a crash on one file must not stop the batch (or the watcher)
            res, good, status, errors = None, False, "failure", [f"{type(e).__name__}: {e}"]
        finally:
            if progress.opts is not None:
                progress.opts.document_timeout = None   # undo a cancel
        if progress.cancelled:
            cancelled(src, name, key, err_path, t0)
            continue
        if not good:
            failed += 1
            print(f"  FAIL  {name}: {'; '.join(errors) or status}", flush=True)
            report = {"source": str(src), "status": status, "errors": errors,
                      "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
            write_atomic(err_path, lambda p: p.write_text(json.dumps(report, indent=2), encoding="utf-8"))
            progress.done(name, "failed", 0, time.time() - t0, "; ".join(errors) or status)
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
        secs = time.time() - t0
        print(f"  {tag} {name}  (" + (f"{n} page{'s' if n != 1 else ''}, " if n else "") + f"{fmt_secs(secs)})", flush=True)
        progress.done(name, "partial" if tag.startswith("PART") else "ok", n, secs)
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
    conv, seen, progress = None, {}, Progress(a.status_file)
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
            conv = conv or build_converter(a, progress)
            print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {len(ready)} new document(s)", flush=True)
            convert_items(conv, ready, out, a, formats, progress)
            for src, _ in ready:
                seen.pop(src, None)
        else:
            progress.queue([p.name for p in seen])   # arriving files (still being copied) + heartbeat
        time.sleep(a.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
