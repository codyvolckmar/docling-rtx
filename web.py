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
import io
import json
import os
import shutil
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
from convert import collect, target  # noqa: E402  (same folder: file discovery and naming stay identical)

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


def vllm_up():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{A.vllm_port}/v1/models", timeout=1) as r:
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


def state():
    inbox, outbox = Path(A.inbox).expanduser(), Path(A.outbox).expanduser()
    try:
        st = json.loads(Path(A.status_file).expanduser().read_text())
    except Exception:
        st = {"state": "idle", "recent": [], "queue": []}
    age = time.time() - st.get("updated", 0)
    watcher = "converting" if st.get("state") == "converting" and age < 600 else "watching" if age < 30 else "stopped"
    recent = {r["file"]: r for r in st.get("recent", [])}
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
        r = recent.get(name, {})
        docs.append({"name": key, "size": src.stat().st_size, "status": status, "cancelling": cancelling,
                     "pages": r.get("pages"), "seconds": r.get("seconds"), "error": r.get("error"),
                     "files": {f: str(p.relative_to(outbox)) for f, p in fresh.items()}, "mtime": mtime})
    docs.sort(key=lambda d: (d["status"] != "converting", d["status"] != "waiting", -d["mtime"]))
    cur = None
    if watcher == "converting":
        cur = {k: st.get(k) for k in ("file", "key", "pages_done", "pages_total", "started")}
        cur["elapsed"] = time.time() - (st.get("started") or time.time())
    return {"watcher": watcher, "current": cur, "vllm": vllm_up(), "gpu": gpu(), "docs": docs,
            "archive": str(Path(A.archive).expanduser()),
            "device": st.get("device"), "batch": A.batch, "sys": system(),
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
            data = p.read_bytes()
            ctype = "application/json" if p.suffix == ".json" else "text/plain; charset=utf-8"
            return self.send(200, data, ctype, {"Content-Disposition": f'attachment; filename="{p.name}"'})
        if path == "/download/outbox.zip":
            buf, out = io.BytesIO(), Path(A.outbox).expanduser()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for f in sorted(out.rglob("*")):
                    rel = f.relative_to(out)
                    if f.is_file() and not any(x.startswith(".") for x in rel.parts):
                        z.write(f, rel)
            return self.send(200, buf.getvalue(), "application/zip",
                             {"Content-Disposition": 'attachment; filename="docling-outbox.zip"'})
        self.send(404, b"not found")

    def do_PUT(self):   # upload: PUT /upload/<file name>, raw body
        if not self.authed():
            return self.send(403, b"forbidden")
        path = urllib.parse.urlsplit(self.path).path
        name = Path(urllib.parse.unquote(path[len("/upload/"):])).name if path.startswith("/upload/") else ""
        if not name or name.startswith((".", "~$")):
            return self.send(400, b"bad file name")
        inbox = Path(A.inbox).expanduser()
        inbox.mkdir(parents=True, exist_ok=True)
        tmp, left = inbox / f".{name}.uploading", int(self.headers.get("Content-Length", 0))
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
        os.replace(tmp, inbox / name)
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
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:16px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:9px 12px}
.stat .k{display:flex;justify-content:space-between;font-size:12px;color:var(--dim)}.stat .k b{color:var(--text);font-weight:600;font-variant-numeric:tabular-nums}
.meter{height:6px;background:var(--line);border-radius:99px;overflow:hidden;margin-top:7px}.meter i{display:block;height:100%;border-radius:99px;transition:width .6s,background .6s}
@media(max-width:640px){.stats{grid-template-columns:repeat(2,1fr)}}
a{color:var(--acc);text-decoration:none;margin-right:10px}a:hover{text-decoration:underline}.dim{color:var(--dim);font-size:13px}
button{background:none;border:1px solid var(--line);color:var(--text);border-radius:8px;padding:3px 10px;cursor:pointer}
button:disabled{opacity:.45;cursor:default}
.row{display:flex;justify-content:space-between;align-items:center;margin:22px 0 10px}.err{color:var(--bad);font-size:12px;max-width:420px}
</style></head><body><main>
<header><h1><img src="/logo.png" alt="" onerror="this.remove()">Docling RTX</h1><div class="pills" id="pills"></div></header>
<div class="stats" id="stats"></div>
<div id="drop"><b>Drop documents here</b> or click to choose<br><span class="dim">PDF, Word, Excel, PowerPoint, CSV, HTML, images, email&hellip; Each one becomes structured JSON.</span>
<input type="file" id="pick" multiple hidden></div>
<div id="up"></div><div id="cur"></div>
<div class="row"><b>Documents</b><span><button id="clear" onclick="clearDone()" title="Move finished documents to the archive folder; their JSON stays in the outbox">Clear finished</button>
<a href="/download/outbox.zip" style="margin-left:14px">Download all results (.zip)</a></span></div>
<table><thead><tr><th>Document</th><th>Status</th><th>Pages</th><th>Time</th><th>Results</th></tr></thead><tbody id="docs"></tbody></table>
<p class="dim" id="paths"></p>
</main><script>
const $=s=>document.querySelector(s), esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const secs=t=>t==null?'':t<60?Math.round(t)+'s':t<3600?Math.floor(t/60)+'m '+String(Math.round(t%60)).padStart(2,'0')+'s':Math.floor(t/3600)+'h '+String(Math.floor(t%3600/60)).padStart(2,'0')+'m';
const size=b=>b<1e6?(b/1e3).toFixed(0)+' KB':(b/1e6).toFixed(1)+' MB';
async function refresh(){
  let s; try{s=await (await fetch('/api/state')).json()}catch(e){return}
  const w={converting:['busy','converting'],watching:['on','watching the inbox'],stopped:['off','watcher stopped: run docling-rtx watch start']}[s.watcher];
  $('#pills').innerHTML=`<span class="pill"><span class="dot ${w[0]}"></span>${w[1]}</span>`+
    (s.device=='cpu'?`<span class="pill cpu" title="PyTorch has no CUDA in the converter: run docling-rtx status">&#9888; converter on CPU (slow)</span>`:s.device=='cuda'?`<span class="pill"><span class="dot on"></span>converter on GPU</span>`:'')+
    `<span class="pill"><span class="dot ${s.vllm?'on':''}"></span>VLM server ${s.vllm?'up':'off'}</span>`+
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
    $('#cur').innerHTML=`<div class="cur"><div class="curhead"><b>Converting ${esc(c.file)}</b>${d?`<button class="stop" ${stopping?'disabled':''} onclick="cancelDoc('${encodeURI(d.name)}')">${stopping?'Cancelling&hellip;':'Cancel'}</button>`:''}</div><div class="bar"><i style="width:${pct}%"></i></div>
    <span class="dim">${c.pages_total?`page ${c.pages_done} of ${c.pages_total} &middot; `:'loading models / reading the file&hellip; &middot; '}${secs(c.elapsed)} so far${left!=null?` &middot; about ${secs(left)} left`:''}${first}</span></div>`;
  } else $('#cur').innerHTML='';
  $('#docs').innerHTML=s.docs.length?s.docs.map(d=>`<tr><td>${esc(d.name)}<br><span class="dim">${size(d.size)}</span></td>
    <td><span class="st ${d.status}">${d.status}</span>${d.status=='failed'&&d.error?`<div class="err" title="${esc(d.error)}">${esc(why(d.error))}</div>`:''}</td>
    <td>${d.pages||''}</td><td>${d.seconds==null?'':d.seconds<1?'&lt;1s':secs(d.seconds)}</td>
    <td>${Object.entries(d.files).filter(([f])=>f!='error.json').map(([f,p])=>`<a href="/files/${encodeURI(p)}">${f}</a>`).join('')}
    ${d.status=='failed'?`<a href="/files/${encodeURI(d.files['error.json'])}">error</a>`:''}
    ${d.status=='failed'||d.status=='cancelled'?`<button onclick="retry('${encodeURI(d.name)}')">${d.status=='cancelled'?'convert':'retry'}</button>`:''}
    ${['done','failed','cancelled'].includes(d.status)?`<button title="Remove from the list (the source moves to the archive, the JSON stays)" onclick="removeDoc('${encodeURI(d.name)}')">&times;</button>`:''}
    ${d.status=='converting'||d.status=='waiting'?`<button class="stop" ${d.cancelling?'disabled':''} onclick="cancelDoc('${encodeURI(d.name)}')">${d.cancelling?'cancelling&hellip;':'cancel'}</button>`:''}</td></tr>`).join('')
    :'<tr><td colspan="5" class="dim">No documents yet. Drop some above.</td></tr>';
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
async function removeDoc(n){const r=await fetch('/remove/'+n,{method:'POST'});if(r.status!=200)alert(await r.text());refresh()}
async function retry(n){await fetch('/retry/'+n,{method:'POST'});refresh()}
async function cancelDoc(n){if(!confirm('Stop converting '+decodeURI(n)+'?\nIts pages done so far are discarded. You can convert it again later.'))return;
  await fetch('/cancel/'+n,{method:'POST'});refresh()}
async function upload(files){
  for(const f of files){
    $('#up').innerHTML=`<div class="cur">Uploading <b>${esc(f.name)}</b> (${size(f.size)})&hellip;<div class="bar"><i id="ub" style="width:0"></i></div></div>`;
    await new Promise(ok=>{const x=new XMLHttpRequest();x.open('PUT','/upload/'+encodeURIComponent(f.name));
      x.upload.onprogress=e=>{const b=$('#ub');if(b&&e.lengthComputable)b.style.width=(100*e.loaded/e.total)+'%'};
      x.onloadend=()=>{if(x.status!=200)alert('Upload failed: '+f.name+' ('+x.status+' '+x.responseText+')');ok()};x.send(f)});
  }
  $('#up').innerHTML='';refresh();
}
const drop=$('#drop');drop.onclick=()=>$('#pick').click();$('#pick').onchange=e=>upload(e.target.files);
['dragenter','dragover'].forEach(t=>drop.addEventListener(t,e=>{e.preventDefault();drop.classList.add('over')}));
['dragleave','drop'].forEach(t=>drop.addEventListener(t,e=>{e.preventDefault();drop.classList.remove('over')}));
drop.addEventListener('drop',e=>upload(e.dataTransfer.files));
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
    ap.add_argument("--vllm-port", type=int, default=8000)
    ap.add_argument("--token", default=os.environ.get("DOCLING_WEB_TOKEN", ""))
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
