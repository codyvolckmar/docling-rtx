#!/usr/bin/env python3
"""Docling RTX web page: drag documents in, watch the conversion, download the JSON.

A thin page over the drop folder: uploads land in the inbox (the docling-watch service converts them), downloads
come from the outbox, live progress comes from the watcher's status file. Python standard library only.

  python web.py --inbox ~/docling/inbox --outbox ~/docling/outbox --status-file ~/.docling-rtx/watch-status.json

Listens on 127.0.0.1:8080 by default: reach it through SSH (ssh -L 8080:127.0.0.1:8080 ...). With --host 0.0.0.0
it requires --token (or DOCLING_WEB_TOKEN): open http://HOST:8080/?token=... once, a cookie remembers it.
"""

import argparse
import hmac
import json
import os
import re
import shutil
import tempfile
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from urllib.parse import quote
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert import collect, suffixes, target  # noqa: E402  (same folder: file discovery and naming stay identical)

A = None   # parsed arguments
_gpu = {"t": 0, "v": None}


def gpu():
    if time.time() - _gpu["t"] > 1.5:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
            name, used, total, util = [x.strip() for x in out.splitlines()[0].split(",")]
            _gpu["v"] = {"name": name, "used_mb": int(used), "total_mb": int(total), "util": int(util)}
        except Exception:
            _gpu["v"] = None
        _gpu["t"] = time.time()
    return _gpu["v"]


CGROUP = Path("/sys/fs/cgroup")
_cpu = {"prev": None}


def _read(path):
    try:
        return Path(path).read_text().strip()
    except Exception:
        return None


def cpu_percent():
    """CPU use since the previous call, as a share of the CPUs this machine (or container) may use.
    Containers: cgroup v2 cpu.stat / cpu.max (a rented pod's /proc/stat shows the whole host)."""
    now = time.monotonic()
    stat, limit = _read(CGROUP / "cpu.stat"), _read(CGROUP / "cpu.max")
    if stat and limit:   # cgroup v2 with its own CPU accounting
        used = int(next(line.split()[1] for line in stat.splitlines() if line.startswith("usage_usec"))) / 1e6
        quota, period = (limit.split() + ["100000"])[:2]
        cores = int(quota) / int(period) if quota != "max" else (os.cpu_count() or 1)
        sample = ("cg", now, used, cores)
    else:                # /proc/stat: busy and total jiffies of the whole machine
        f = [int(x) for x in (_read("/proc/stat") or "cpu 0 0 0 0").splitlines()[0].split()[1:9]]
        sample = ("proc", sum(f) - f[3] - f[4], sum(f), os.cpu_count() or 1)
    prev, _cpu["prev"] = _cpu["prev"], sample
    if not prev or prev[0] != sample[0]:
        return None, sample[3]
    if sample[0] == "cg":
        dt = sample[1] - prev[1]
        pct = 100 * (sample[2] - prev[2]) / (dt * sample[3]) if dt > 0 else None
    else:
        dt = sample[2] - prev[2]
        pct = 100 * (sample[1] - prev[1]) / dt if dt > 0 else None
    return (None if pct is None else max(0.0, min(100.0, pct))), sample[3]


def memory():
    """(used MB, total MB). Containers: cgroup v2 memory.current minus reclaimable file cache, against memory.max
    (as `docker stats` does); otherwise /proc/meminfo MemTotal - MemAvailable."""
    cur, lim = _read(CGROUP / "memory.current"), _read(CGROUP / "memory.max")
    info = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        k, _, v = line.partition(":")
        info[k] = int(v.split()[0]) * 1024 if v.split() else 0
    if cur and lim and lim != "max":
        inactive = next((int(line.split()[1]) for line in (_read(CGROUP / "memory.stat") or "").splitlines()
                         if line.startswith("inactive_file ")), 0)
        return (int(cur) - inactive) / 2**20, min(int(lim), info.get("MemTotal", int(lim))) / 2**20
    total = info.get("MemTotal", 0)
    return (total - info.get("MemAvailable", total)) / 2**20, total / 2**20


def system():
    cpu, cores = cpu_percent()
    used, total = memory()
    return {"cpu": None if cpu is None else round(cpu), "cores": round(cores, 1), "mem_used_mb": round(used),
            "mem_total_mb": round(total)}


_direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # localhost: never through http_proxy


def vllm_port():
    """The vLLM port from the saved settings, read on every check: changing it in docling-rtx Settings must not
    leave this page looking at the old port. Falls back to --vllm-port."""
    for line in (_read(Path(A.config).expanduser()) or "").splitlines():
        if line.startswith("PORT="):
            v = line.split("=", 1)[1].strip().strip('"')
            if v.isdigit():
                return int(v)
    return A.vllm_port


def vllm_up(port):
    try:
        with _direct.open(f"http://127.0.0.1:{port}/v1/models", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


FINISHED = ("done", "failed", "cancelled")


def archive(rel):
    """Move a finished document's source out of the inbox into the archive folder (same subfolders), so it leaves
    the list. Its results stay in the outbox. Never overwrites: a name that's taken gets a timestamp."""
    inbox, arch = Path(A.inbox).expanduser().resolve(), Path(A.archive).expanduser()
    src = (inbox / rel).resolve()
    if inbox not in src.parents or not src.is_file():
        return False
    dst = arch / src.relative_to(inbox)
    if dst.exists():
        dst = dst.with_name(f"{dst.stem}.{time.strftime('%Y%m%d-%H%M%S')}{dst.suffix}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return True


def poke(rel):
    """Wake the watcher now and tell it this inbox file is complete, so it converts it without waiting a scan."""
    try:
        with open(Path(A.status_file).expanduser().with_name("poke"), "a") as f:
            f.write(str(rel) + "\n")
    except OSError:
        pass


def public_url():
    """The Cloudflare quick tunnel's current address, from cloudflared's log (it changes on every restart)."""
    if not A.tunnel_log:
        return None
    found = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", _read(Path(A.tunnel_log).expanduser()) or "")
    return found[-1] if found else None


def state():
    inbox, outbox = Path(A.inbox).expanduser(), Path(A.outbox).expanduser()
    try:
        st = json.loads(Path(A.status_file).expanduser().read_text())
    except Exception:
        st = {"state": "idle", "recent": [], "queue": []}
    age = time.time() - st.get("updated", 0)
    watcher = "converting" if st.get("state") == "converting" and age < 600 else "watching" if age < 30 else "stopped"
    recent = {r.get("key", r["file"]): r for r in st.get("recent", [])}
    cancel_dir = Path(A.status_file).expanduser().parent / "cancel"
    docs = []
    for src, rel in collect([str(inbox)], outbox) if inbox.is_dir() else []:
        outs = {f: target(outbox, rel, f) for f in ("json", "md", "html", "txt", "error.json")}
        mtime = src.stat().st_mtime
        fresh = {f: p for f, p in outs.items() if p.exists() and p.stat().st_mtime >= mtime}
        name, key = src.name, str(src.relative_to(inbox))
        if watcher == "converting" and st.get("key", st.get("file")) in (key, name):
            status = "converting"
        elif any(f != "error.json" for f in fresh):
            status = "done"
        elif "error.json" in fresh:
            try:
                status = "cancelled" if json.loads(fresh["error.json"].read_text())["status"] == "cancelled" else "failed"
            except Exception:
                status = "failed"
        else:
            status = "waiting"
        cancelling = (cancel_dir / quote(key, safe="")).exists() and status in ("converting", "waiting")
        r = recent.get(key) or recent.get(name, {})
        docs.append({"name": key, "size": src.stat().st_size, "status": status, "cancelling": cancelling,
                     "pages": r.get("pages"), "seconds": r.get("seconds"), "error": r.get("error"),
                     "files": {f: str(p.relative_to(outbox)) for f, p in fresh.items()}, "mtime": mtime})
    docs.sort(key=lambda d: (d["status"] != "converting", d["status"] != "waiting", -d["mtime"]))
    cur = None
    if watcher == "converting":
        cur = {k: st.get(k) for k in ("file", "key", "pages_done", "pages_total", "started")}
        cur["elapsed"] = time.time() - (st.get("started") or time.time())
    port = vllm_port()
    return {"watcher": watcher, "current": cur, "vllm": vllm_up(port), "vllm_port": port, "gpu": gpu(), "docs": docs,
            "archive": str(Path(A.archive).expanduser()),
            "device": st.get("device"), "batch": A.batch, "sys": system(), "suffixes": sorted(suffixes()),
            "public_url": (lambda u: u and u + "/?token=" + quote(A.token))(public_url()) if A.token else None,
            "inbox": str(inbox), "outbox": str(outbox)}


class H(BaseHTTPRequestHandler):
    server_version = "docling-rtx"

    def log_message(self, fmt, *args):   # quiet: no access log per poll
        pass

    def authed(self):
        if not A.token:
            return True
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get("token", [""])[0]
        c = dict(x.strip().split("=", 1) for x in self.headers.get("Cookie", "").split(";") if "=" in x).get("drtx", "")
        return hmac.compare_digest(q, A.token) or hmac.compare_digest(c, A.token)

    def send_file(self, path, ctype, extra=None):
        """Stream a file (no full copy in memory: a big outbox zip or JSON must not eat the pod's RAM)."""
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(path.stat().st_size))
        for k, v in (extra or {"Cache-Control": "no-store"}).items():
            self.send_header(k, v)
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile, 1 << 20)

    def send(self, code, body=b"", ctype="text/plain; charset=utf-8", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if "Cache-Control" not in (extra or {}):
            self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def safe(self, base, rel):
        base = Path(base).expanduser().resolve()
        p = (base / urllib.parse.unquote(rel)).resolve()
        return p if base in p.parents and not any(part.startswith(".") for part in p.relative_to(base).parts) else None

    def do_GET(self):
        if not self.authed():
            return self.send(403, b"Add ?token=... to the URL (docling-rtx web prints it).")
        path = urllib.parse.urlsplit(self.path).path
        if "token=" in self.path:   # remember the token, then drop it from the address bar
            return self.send(302, extra={"Location": "/", "Set-Cookie": f"drtx={A.token}; HttpOnly; SameSite=Strict; Path=/"})
        if path == "/":
            return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
        if path in ("/logo.png", "/favicon.ico"):   # next to web.py (the installer copies it)
            logo = Path(__file__).resolve().parent / "logo.png"
            if not logo.is_file():
                return self.send(404, b"no logo")
            return self.send(200, logo.read_bytes(), "image/png", {"Cache-Control": "max-age=86400"})
        if path == "/api/state":
            return self.send(200, json.dumps(state()).encode(), "application/json")
        if path.startswith("/files/"):
            p = self.safe(A.outbox, path[len("/files/"):])
            if not p or not p.is_file():
                return self.send(404, b"not found")
            disp = f"attachment; filename*=UTF-8''{quote(p.name)}"
            ctype = "application/json" if p.suffix == ".json" else "text/plain; charset=utf-8"
            return self.send_file(p, ctype, {"Content-Disposition": disp, "Cache-Control": "no-store"})
        if path == "/download/outbox.zip":   # built in a temp file on disk, then streamed
            out = Path(A.outbox).expanduser()
            with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
                with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
                    for f in sorted(out.rglob("*")):
                        rel = f.relative_to(out)
                        if f.is_file() and not any(x.startswith(".") for x in rel.parts):
                            z.write(f, rel)
                tmp.flush()
                return self.send_file(Path(tmp.name), "application/zip",
                                      {"Content-Disposition": 'attachment; filename="docling-outbox.zip"',
                                       "Cache-Control": "no-store"})
        self.send(404, b"not found")

    def do_PUT(self):   # upload: PUT /upload/<file name>, raw body
        if not self.authed():
            return self.send(403, b"forbidden")
        path = urllib.parse.urlsplit(self.path).path
        # PUT /upload/<name> or /upload/<folder>/<sub>/<name> (folder uploads keep their layout in the inbox)
        parts = [urllib.parse.unquote(x) for x in path[len("/upload/"):].split("/")] if path.startswith("/upload/") else []
        if not parts or any(x in ("", ".", "..") or x.startswith((".", "~$")) or "/" in x or "\\" in x for x in parts):
            return self.send(400, b"bad file name")   # (a decoded "/" inside a part would be a path escape)
        rel = Path(*parts)
        if rel.suffix.lower() not in suffixes():
            return self.send(415, f"not a format Docling reads ({rel.suffix or 'no extension'})".encode())
        inbox = Path(A.inbox).expanduser()
        dst = inbox / rel
        if inbox.resolve() not in dst.resolve().parents:   # backstop: never write outside the inbox
            return self.send(400, b"bad file name")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp, left = dst.parent / f".{dst.name}.uploading", int(self.headers.get("Content-Length", 0))
        with open(tmp, "wb") as f:   # hidden name while it arrives: the watcher ignores it until the rename
            while left > 0:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        if left:
            tmp.unlink(missing_ok=True)
            return self.send(400, b"upload interrupted")
        os.replace(tmp, dst)
        poke(rel)
        self.send(200, b"ok")

    def do_POST(self):   # /retry/<inbox path>: convert again; /cancel/<inbox path>: stop or skip it
        if not self.authed():
            return self.send(403, b"forbidden")
        path = urllib.parse.urlsplit(self.path).path
        if path == "/clear":   # move every finished document's source to the archive
            n = sum(archive(d["name"]) for d in state()["docs"] if d["status"] in FINISHED)
            return self.send(200, json.dumps({"cleared": n}).encode(), "application/json")
        if path.startswith("/remove/"):
            rel = urllib.parse.unquote(path[len("/remove/"):])
            d = next((d for d in state()["docs"] if d["name"] == rel), None)
            if d and d["status"] in FINISHED and archive(rel):
                return self.send(200, b"ok")
            return self.send(409, b"only finished documents can be removed (cancel it first)")
        if path.startswith("/cancel/"):
            p = self.safe(A.inbox, path[len("/cancel/"):])
            rel = urllib.parse.unquote(path[len("/cancel/"):])
            d = next((d for d in state()["docs"] if d["name"] == rel), None)
            if not d or d["status"] not in ("converting", "waiting"):
                return self.send(409, b"only waiting or converting documents can be cancelled")
            if p and p.is_file():
                d = Path(A.status_file).expanduser().parent / "cancel"
                d.mkdir(parents=True, exist_ok=True)
                key = str(p.relative_to(Path(A.inbox).expanduser().resolve()))
                (d / quote(key, safe="")).touch()
                return self.send(200, b"ok")
        if path.startswith("/retry/"):
            p = self.safe(A.inbox, path[len("/retry/"):])
            if p and p.is_file():
                os.utime(p)
                poke(p.relative_to(Path(A.inbox).expanduser().resolve()))
                return self.send(200, b"ok")
        self.send(404, b"not found")


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Docling RTX</title><link rel="icon" type="image/png" href="/logo.png"><style>
:root{--bg:#0b1020;--card:#141b2e;--line:#26304a;--text:#e6e9f2;--dim:#8b93a7;--acc:#5b9dff;--ok:#3ecf8e;--bad:#ff6b6b;--warn:#f5b942}
*{box-sizing:border-box}body{margin:0;font:15px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--text)}
main{max-width:1000px;margin:0 auto;padding:24px}h1{font-size:22px;margin:0;display:flex;align-items:center;gap:12px}
h1 img{width:52px;height:52px;filter:drop-shadow(0 2px 8px #0008)}header{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap}
.pills{display:flex;gap:8px;flex-wrap:wrap}.pill{background:var(--card);border:1px solid var(--line);border-radius:99px;padding:4px 12px;font-size:13px;color:var(--dim)}
.pill b{color:var(--text);font-weight:600}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;background:var(--dim)}
.on{background:var(--ok)}.busy{background:var(--acc)}.off{background:var(--bad)}
#drop{margin:20px 0;border:2px dashed var(--line);border-radius:14px;padding:34px;text-align:center;color:var(--dim);cursor:pointer;transition:.15s}
#drop.over{border-color:var(--acc);background:#5b9dff14;color:var(--text)}#drop b{color:var(--text)}
.cur{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:16px}
.bar{height:10px;background:var(--line);border-radius:99px;overflow:hidden;margin:10px 0 6px}.bar i{display:block;height:100%;background:var(--acc);transition:width .5s}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden}
th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line)}th{color:var(--dim);font-weight:500;font-size:13px}tr:last-child td{border:0}
.st{font-size:13px;font-weight:600}.done{color:var(--ok)}.failed{color:var(--bad)}.waiting{color:var(--warn)}.converting{color:var(--acc)}.cancelled{color:var(--dim)}
.cpu{border-color:var(--bad);color:var(--bad)}.stop{border-color:#ff6b6b88;color:#ffb3b3}.stop:hover{background:#ff6b6b22}
.curhead{display:flex;justify-content:space-between;align-items:center;gap:12px}
.share{display:flex;align-items:center;gap:10px;margin-top:12px;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:8px 12px;font-size:13px}
.share code{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--text)}.share span{color:var(--dim)}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:16px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:9px 12px}
.stat .k{display:flex;justify-content:space-between;font-size:12px;color:var(--dim)}.stat .k b{color:var(--text);font-weight:600;font-variant-numeric:tabular-nums}
.meter{height:6px;background:var(--line);border-radius:99px;overflow:hidden;margin-top:7px}.meter i{display:block;height:100%;border-radius:99px;transition:width .6s,background .6s}
@media(max-width:640px){.stats{grid-template-columns:repeat(2,1fr)}}
a{color:var(--acc);text-decoration:none;margin-right:10px}a:hover{text-decoration:underline}.dim{color:var(--dim);font-size:13px}
button{background:none;border:1px solid var(--line);color:var(--text);border-radius:8px;padding:3px 10px;cursor:pointer}
button:disabled{opacity:.45;cursor:default}
#docs td:nth-child(3),#docs td:nth-child(4),#docs td:nth-child(5){white-space:nowrap}#docs td:first-child{word-break:break-word}
.filters{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 10px}.chip{border-radius:99px;font-size:13px;padding:3px 12px;color:var(--dim)}
.chip b{font-weight:600;margin-left:4px;color:var(--text)}.chip.on{border-color:var(--acc);background:#5b9dff22;color:var(--text)}
.row{display:flex;justify-content:space-between;align-items:center;margin:22px 0 10px}.err{color:var(--bad);font-size:12px;max-width:420px}
</style></head><body><main>
<header><h1><img src="/logo.png" alt="" onerror="this.remove()">Docling RTX</h1><div class="pills" id="pills"></div></header>
<div class="stats" id="stats"></div>
<div id="share"></div>
<div id="drop"><b>Drop files or folders here</b> or <a href="#" id="pf">choose files</a> &middot; <a href="#" id="pd">choose a folder</a><br><span class="dim">PDF, Word, Excel, PowerPoint, CSV, HTML, images, EPUB, email&hellip; Each one becomes Markdown for your agent. Folders keep their layout.</span>
<input type="file" id="pick" multiple hidden><input type="file" id="pickdir" webkitdirectory multiple hidden></div>
<div id="up"></div><div id="cur"></div>
<div class="row"><b>Documents</b><span><button id="clear" onclick="clearDone()" title="Move finished documents to the archive folder; their JSON stays in the outbox">Clear finished</button>
<a href="/download/outbox.zip" style="margin-left:14px">Download all results (.zip)</a></span></div>
<div class="filters" id="filters"></div>
<table><thead><tr><th>Document</th><th>Status</th><th>Pages</th><th>Time</th><th>Results</th></tr></thead><tbody id="docs"></tbody></table>
<p class="dim" id="paths"></p>
</main><script>
const $=s=>document.querySelector(s), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const enc=p=>String(p).split('/').map(encodeURIComponent).join('/');   // safe for '#', '?', '&', "'", '+' in names
document.addEventListener('click',e=>{if(e.target.closest('button[data-close]')){$('#up').innerHTML='';return}
  const fb=e.target.closest('button[data-filter]');
  if(fb){FILTER=fb.dataset.filter;localStorage.setItem('drtx-filter',FILTER);refresh();return}
  const c=e.target.closest('button[data-copy]');
  if(c){navigator.clipboard.writeText($('#pub').textContent).then(()=>{c.textContent='copied';setTimeout(()=>c.textContent='copy',1500)});return}
  const b=e.target.closest('button[data-act]');if(!b||b.disabled)return;
  ({cancel:cancelDoc,retry:retry,remove:removeDoc})[b.dataset.act](b.dataset.n)});
const secs=t=>t==null?'':t<60?Math.round(t)+'s':t<3600?Math.floor(t/60)+'m '+String(Math.round(t%60)).padStart(2,'0')+'s':Math.floor(t/3600)+'h '+String(Math.floor(t%3600/60)).padStart(2,'0')+'m';
const size=b=>b<1e6?(b/1e3).toFixed(0)+' KB':(b/1e6).toFixed(1)+' MB';
async function refresh(){
  let s; try{s=await (await fetch('/api/state')).json()}catch(e){return}
  SUPPORTED=new Set(s.suffixes||[]);
  const w={converting:['busy','converting'],watching:['on','watching the inbox'],stopped:['off','watcher stopped: run docling-rtx watch start']}[s.watcher];
  $('#pills').innerHTML=`<span class="pill"><span class="dot ${w[0]}"></span>${w[1]}</span>`+
    (s.device=='cpu'?`<span class="pill cpu" title="PyTorch has no CUDA in the converter: run docling-rtx status">&#9888; converter on CPU (slow)</span>`:s.device=='cuda'?`<span class="pill"><span class="dot on"></span>converter on GPU</span>`:'')+
    `<span class="pill" title="${s.vllm?`vLLM answers on 127.0.0.1:${s.vllm_port}`:`Nothing answers on 127.0.0.1:${s.vllm_port}/v1/models. Start it: docling-rtx start. Status and log: docling-rtx status / docling-rtx logs`}"><span class="dot ${s.vllm?'on':''}"></span>VLM server ${s.vllm?'up':'off'} &middot; :${s.vllm_port}</span>`+
    (s.gpu?`<span class="pill"><b>${esc(s.gpu.name)}</b></span>`:'<span class="pill">no GPU</span>');
  const gb=mb=>(mb/1024).toFixed(1), y=s.sys||{};
  const tile=(k,pct,txt,title)=>{const c=pct==null?'var(--line)':pct>=90?'var(--bad)':pct>=70?'var(--warn)':'var(--acc)';
    return `<div class="stat" title="${title}"><div class="k">${k}<b>${txt}</b></div><div class="meter"><i style="width:${pct??0}%;background:${c}"></i></div></div>`};
  $('#stats').innerHTML=
    tile('CPU',y.cpu,y.cpu==null?'&hellip;':y.cpu+'%',`${y.cores} CPUs available to this machine/container`)+
    tile('RAM',y.mem_total_mb?100*y.mem_used_mb/y.mem_total_mb:null,y.mem_total_mb?`${gb(y.mem_used_mb)} / ${gb(y.mem_total_mb)} GB`:'n/a','System memory in use')+
    tile('GPU',s.gpu?s.gpu.util:null,s.gpu?s.gpu.util+'%':'n/a','GPU utilization (nvidia-smi)')+
    tile('VRAM',s.gpu?100*s.gpu.used_mb/s.gpu.total_mb:null,s.gpu?`${gb(s.gpu.used_mb)} / ${gb(s.gpu.total_mb)} GB`:'n/a','GPU memory in use (vLLM reserves its share up front)');
  const c=s.current;
  if(c){const pct=c.pages_total?100*c.pages_done/c.pages_total:0, rate=c.pages_done/Math.max(c.elapsed,1);
    const left=c.pages_total&&c.pages_done?(c.pages_total-c.pages_done)/rate:null;
    const d=s.docs.find(x=>x.status=='converting'), stopping=d&&d.cancelling;
    const first=c.pages_total&&!c.pages_done?` &middot; the first ${Math.min(s.batch||8,c.pages_total)} pages go through all models together, so page 1 can take a while`:'';
    $('#cur').innerHTML=`<div class="cur"><div class="curhead"><b>Converting ${esc(c.file)}</b>${d?`<button class="stop" ${stopping?'disabled':''} data-act="cancel" data-n="${esc(d.name)}">${stopping?'Cancelling&hellip;':'Cancel'}</button>`:''}</div><div class="bar"><i style="width:${pct}%"></i></div>
    <span class="dim">${c.pages_total?`page ${c.pages_done} of ${c.pages_total} &middot; `:'loading models / reading the file&hellip; &middot; '}${secs(c.elapsed)} so far${left!=null?` &middot; about ${secs(left)} left`:''}${first}</span></div>`;
  } else $('#cur').innerHTML='';
  const counts={all:s.docs.length};for(const d of s.docs)counts[d.status]=(counts[d.status]||0)+1;
  if(FILTER!='all'&&!counts[FILTER])FILTER='all';   // a filter that became empty: show everything again
  $('#filters').innerHTML=s.docs.length?['all','converting','waiting','done','failed','cancelled'].map(k=>
    `<button class="chip ${FILTER==k?'on':''}" data-filter="${k}" ${k!='all'&&!counts[k]?'disabled':''}>${k[0].toUpperCase()+k.slice(1)}<b>${counts[k]||0}</b></button>`).join(''):'';
  const shown=s.docs.filter(d=>FILTER=='all'||d.status==FILTER);
  $('#docs').innerHTML=shown.length?shown.map(d=>`<tr><td>${esc(d.name)}<br><span class="dim">${size(d.size)}</span></td>
    <td><span class="st ${d.status}">${d.status}</span>${d.status=='failed'&&d.error?`<div class="err" title="${esc(d.error)}">${esc(why(d.error))}</div>`:''}</td>
    <td>${d.pages||''}</td><td>${d.seconds==null?'':d.seconds<1?'&lt;1s':secs(d.seconds)}</td>
    <td>${Object.entries(d.files).filter(([f])=>f!='error.json').map(([f,p])=>`<a href="/files/${enc(p)}">${f}</a>`).join('')}
    ${d.status=='failed'?`<a href="/files/${enc(d.files['error.json'])}">error</a>`:''}
    ${d.status=='failed'||d.status=='cancelled'?`<button data-act="retry" data-n="${esc(d.name)}">${d.status=='cancelled'?'convert':'retry'}</button>`:''}
    ${['done','failed','cancelled'].includes(d.status)?`<button title="Remove from the list (the source moves to the archive, the JSON stays)" data-act="remove" data-n="${esc(d.name)}">&times;</button>`:''}
    ${d.status=='converting'||d.status=='waiting'?`<button class="stop" ${d.cancelling?'disabled':''} data-act="cancel" data-n="${esc(d.name)}">${d.cancelling?'cancelling&hellip;':'cancel'}</button>`:''}</td></tr>`).join('')
    :`<tr><td colspan="5" class="dim">${s.docs.length?'Nothing with this status.':'No documents yet. Drop some above.'}</td></tr>`;
  $('#share').innerHTML=s.public_url?`<div class="share"><span>Public link</span><code id="pub">${esc(s.public_url)}</code>
    <button data-copy="1" title="Anyone with this link (its key included) can use this page. It changes when the tunnel restarts.">copy</button></div>`:'';
  $('#paths').textContent=`Inbox: ${s.inbox}   Outbox: ${s.outbox}   Cleared documents: ${s.archive}`;
  const fin=s.docs.filter(d=>['done','failed','cancelled'].includes(d.status)).length;
  $('#clear').disabled=!fin; $('#clear').textContent=fin?`Clear finished (${fin})`:'Clear finished';
}
function why(e){  // plain-English reason for the common failures; the full text is in the tooltip and the error file
  if(/could not load document|trailer dictionary|not valid/i.test(e))return 'Not a readable PDF (damaged, or not really a PDF).';
  if(/password|encrypt/i.test(e))return 'The PDF is password-protected.';
  if(/out of memory/i.test(e))return 'The GPU ran out of memory. Try again, or lower the batch size in Settings.';
  if(/timed? ?out/i.test(e))return 'Timed out.';
  if(/No vLLM server|Connection/i.test(e))return 'The VLM server is not running (docling-rtx start).';
  return e.length>160?e.slice(0,160)+'…':e;
}
async function clearDone(){
  if(!confirm('Remove all finished documents from the list?\nTheir JSON stays in the outbox; the original files move to the archive folder.'))return;
  await fetch('/clear',{method:'POST'});refresh()}
async function removeDoc(n){const r=await fetch('/remove/'+enc(n),{method:'POST'});if(r.status!=200)alert(await r.text());refresh()}
async function retry(n){await fetch('/retry/'+enc(n),{method:'POST'});refresh()}
async function cancelDoc(n){if(!confirm('Stop converting '+n+'?\nIts pages done so far are discarded. You can convert it again later.'))return;
  await fetch('/cancel/'+enc(n),{method:'POST'});refresh()}
let FILTER=localStorage.getItem('drtx-filter')||'all';   // status filter for the table, kept across reloads
let SUPPORTED=null;   // file extensions Docling reads (from /api/state)
const ext=n=>{const i=n.lastIndexOf('.');return i>0?n.slice(i).toLowerCase():''};
async function filesFrom(dt){   // dropped files and folders -> [{file, path}] (folders walked recursively)
  const entries=[...dt.items].filter(i=>i.kind=='file').map(i=>i.webkitGetAsEntry&&i.webkitGetAsEntry()).filter(Boolean);
  if(!entries.length)return [...dt.files].map(f=>({file:f,path:f.name}));   // must be read before the first await
  const out=[];
  const walk=async(en,prefix)=>{
    if(en.isFile){out.push({file:await new Promise((ok,ko)=>en.file(ok,ko)),path:prefix+en.name});return}
    const r=en.createReader();let batch;
    do{batch=await new Promise((ok,ko)=>r.readEntries(ok,ko));for(const c of batch)await walk(c,prefix+en.name+'/')}while(batch.length);
  };
  for(const en of entries)await walk(en,'');
  return out;
}
function sendOne(f,path,i,n){   // one PUT with a progress bar; resolves to '' (ok) or the reason it failed
  $('#up').innerHTML=`<div class="cur">Uploading ${n>1?`${i} of ${n}: `:''}<b>${esc(path)}</b> (${size(f.size)})&hellip;<div class="bar"><i id="ub" style="width:0"></i></div></div>`;
  return new Promise(ok=>{const x=new XMLHttpRequest();x.open('PUT','/upload/'+enc(path));
    x.upload.onprogress=e=>{const b=$('#ub');if(b&&e.lengthComputable)b.style.width=(100*e.loaded/e.total)+'%'};
    x.onloadend=()=>ok(x.status==200?'':x.status?`${x.status} ${x.responseText}`:'connection lost (unreadable file, or the server went away)');
    x.send(f)});
}
async function upload(items){
  if(!SUPPORTED)await refresh();
  const viaTunnel=location.hostname.endsWith('.trycloudflare.com'), skipped={}, failed=[], todo=[];
  for(const it of items){
    const name=it.path.split('/').pop();
    if(it.path.split('/').some(p=>p.startsWith('.')||p.startsWith('~$')))continue;          // hidden / lock files
    if(SUPPORTED&&!SUPPORTED.has(ext(name))){const k=ext(name)||'no extension';skipped[k]=(skipped[k]||0)+1;continue}
    if(viaTunnel&&it.file.size>100e6){failed.push(`${it.path}: ${size(it.file.size)}, over the tunnel's 100 MB limit (copy it into the inbox with scp)`);continue}
    todo.push(it);
  }
  let done=0;
  for(const [i,it] of todo.entries()){const why=await sendOne(it.file,it.path,i+1,todo.length);if(why)failed.push(`${it.path}: ${why}`);else done++}
  const sk=Object.entries(skipped).map(([k,v])=>`${k} &times;${v}`).join(', ');
  $('#up').innerHTML=(done||sk||failed.length)?`<div class="cur"><div class="curhead"><span>Uploaded <b>${done}</b> file${done==1?'':'s'}${sk?` &middot; skipped (not a format Docling reads): ${sk}`:''}${failed.length?` &middot; <span class="failed">${failed.length} failed</span>`:''}</span><button data-close="1">&times;</button></div>${failed.map(f=>`<div class="err">${esc(f)}</div>`).join('')}</div>`:'';
  refresh();
}
const drop=$('#drop');
drop.onclick=e=>{if(e.target.id=='pd'){e.preventDefault();$('#pickdir').click()}else{e.preventDefault();$('#pick').click()}};
$('#pick').onchange=e=>{upload([...e.target.files].map(f=>({file:f,path:f.name})));e.target.value=''};
$('#pickdir').onchange=e=>{upload([...e.target.files].map(f=>({file:f,path:f.webkitRelativePath||f.name})));e.target.value=''};
['dragenter','dragover'].forEach(t=>drop.addEventListener(t,e=>{e.preventDefault();drop.classList.add('over')}));
['dragleave','drop'].forEach(t=>drop.addEventListener(t,e=>{e.preventDefault();drop.classList.remove('over')}));
drop.addEventListener('drop',async e=>upload(await filesFrom(e.dataTransfer)));
refresh();setInterval(refresh,2000);
</script></body></html>"""


def main():
    global A
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inbox", default="~/docling/inbox")
    ap.add_argument("--outbox", default="~/docling/outbox")
    ap.add_argument("--archive", default=None, help="where cleared documents go (default: archive/ next to the inbox)")
    ap.add_argument("--status-file", default="~/.docling-rtx/watch-status.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--vllm-port", type=int, default=8000, help="used when --config has no PORT")
    ap.add_argument("--config", default="~/.docling-rtx/config.env", help="docling-rtx settings (for the vLLM port)")
    ap.add_argument("--token", default=os.environ.get("DOCLING_WEB_TOKEN", ""))
    ap.add_argument("--tunnel-log", default=None, help="cloudflared's log: the page shows its public https link")
    ap.add_argument("--batch", type=int, default=0, help="the converter's batch size (shown while page 1 is pending)")
    A = ap.parse_args()
    A.archive = A.archive or str(Path(A.inbox).expanduser().parent / "archive")
    if A.host not in ("127.0.0.1", "localhost", "::1") and not A.token:
        sys.exit("Listening beyond localhost needs a --token (or DOCLING_WEB_TOKEN).")
    os.makedirs(Path(A.inbox).expanduser(), exist_ok=True)
    srv = ThreadingHTTPServer((A.host, A.port), H)
    print(f"Docling RTX web page on http://{A.host}:{A.port}/" + ("?token=..." if A.token else ""), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
