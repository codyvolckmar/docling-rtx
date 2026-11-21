#!/usr/bin/env python3
"""Benchmark a Summon server (any OpenAI-compatible llama.cpp endpoint): speed and coherence.

  python3 tools/bench.py --url http://HOST:PORT/v1 --key-file api-key [--quick] [--out results/]

Speed: decode tok/s per workload (prose, code, edit-a-file), prefill tok/s and time-to-first-token from 1K to
the largest context you ask for, and draft acceptance when speculative decoding is on (llama.cpp `timings`).
Coherence: graded tasks with checkable answers (math, logic, code run against tests, strict JSON, tool calls,
multi-turn recall), needle-in-a-haystack at every context size, and a long free-form text checked for loops.
Standard library only. Model-written code runs in a temp folder with `python3 -I` and a 10 s timeout.
"""
import argparse, json, os, random, re, statistics, subprocess, sys, tempfile, time, urllib.request

A = None  # parsed args


def chat(messages, max_tokens=1024, think=True, tools=None, temperature=None, **extra):
    """Streamed chat call. Returns content, reasoning, tool_calls, ttft_s, wall_s, timings, usage."""
    body = {"model": A.model, "messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}}
    if not think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if tools:
        body["tools"] = tools
    if temperature is not None:
        body["temperature"] = temperature
    body.update(extra)
    req = urllib.request.Request(A.url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + A.key})
    out = {"content": "", "reasoning": "", "tool_calls": {}, "ttft_s": None, "timings": {}, "usage": {},
           "finish": None}
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=A.timeout) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            ev = json.loads(data)
            out["timings"] = ev.get("timings") or out["timings"]
            out["usage"] = ev.get("usage") or out["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if ch.get("finish_reason"):
                    out["finish"] = ch["finish_reason"]
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if (piece or d.get("tool_calls")) and out["ttft_s"] is None:
                    out["ttft_s"] = time.time() - t0
                out["content"] += d.get("content") or ""
                out["reasoning"] += d.get("reasoning_content") or ""
                for tc in d.get("tool_calls") or []:
                    slot = out["tool_calls"].setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                    f = tc.get("function") or {}
                    slot["name"] += f.get("name") or ""
                    slot["arguments"] += f.get("arguments") or ""
    out["wall_s"] = time.time() - t0
    out["tool_calls"] = list(out["tool_calls"].values())
    return out


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- filler text for long prompts
WORDS = ("river stone market winter lantern harbor copper meadow signal orchard ledger canyon velvet engine "
         "quarry thistle beacon saddle glacier pepper compass ribbon tundra marble falcon kettle prairie "
         "anchor violet timber cobalt ferry walnut summit oyster cinder planet hollow maple ember cable").split()
VERBS = "carried watched counted painted followed measured repaired described traded buried".split()


def filler(n_words, rng):
    out = []
    while len(out) < n_words:
        s = [rng.choice(WORDS).capitalize(), rng.choice(VERBS), "the", rng.choice(WORDS), "near", "the",
             rng.choice(WORDS), rng.choice(["while", "because", "after", "before"]), "a", rng.choice(WORDS),
             rng.choice(VERBS), "every", rng.choice(WORDS) + "."]
        out += s
    return " ".join(out)


WORDS_PER_TOKEN = 0.87  # measured on this filler with Nanbeige's tokenizer (1.14 tok/word)


# ---------------------------------------------------------------- speed
def bench_speed():
    code_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "termshot.py")).read()[:6000]
    loads = {
        "prose (new text)": ([{"role": "user", "content":
                               "Write a 600-word short story about a lighthouse keeper who finds a map. Plain prose."}], 900),
        "code (new code)": ([{"role": "user", "content":
                              "Write a Python module implementing an LRU cache class with get/put/delete, type hints, "
                              "docstrings and a small unittest suite. Code only."}], 900),
        "edit (rewrite a file)": ([{"role": "user", "content":
                                    "Return this file unchanged except rename every variable called `lines` to `rows`. "
                                    "Output the whole file, code only.\n\n```python\n" + code_src + "\n```"}], 1500),
    }
    rows = []
    for name, (msgs, mt) in loads.items():
        runs = []
        for i in range(A.reps):
            nonce = f"[run {random.random():.8f}] "  # defeat prompt cache so each run is cold-ish and comparable
            m = [{"role": "user", "content": nonce + msgs[0]["content"]}]
            r = chat(m, max_tokens=mt, think=False, temperature=0.6)
            t = r["timings"]
            runs.append({"decode_tps": t.get("predicted_per_second"), "tokens": t.get("predicted_n"),
                         "ttft_s": r["ttft_s"], "draft_n": t.get("draft_n", 0),
                         "draft_acc": t.get("draft_n_accepted", 0),
                         "e2e_tps": (t.get("predicted_n") or 0) / r["wall_s"]})
            log(f"  {name:22s} run {i+1}: {runs[-1]['decode_tps']:.1f} tok/s, {runs[-1]['tokens']} tok, "
                f"draft {runs[-1]['draft_acc']}/{runs[-1]['draft_n']}")
        dn = sum(x["draft_n"] for x in runs)
        rows.append({"workload": name,
                     "decode_tps_median": statistics.median(x["decode_tps"] for x in runs),
                     "decode_tps_min": min(x["decode_tps"] for x in runs),
                     "decode_tps_max": max(x["decode_tps"] for x in runs),
                     "e2e_tps_median": statistics.median(x["e2e_tps"] for x in runs),
                     "ttft_s_median": statistics.median(x["ttft_s"] for x in runs),
                     "tokens_median": statistics.median(x["tokens"] for x in runs),
                     "draft_accept": (sum(x["draft_acc"] for x in runs) / dn) if dn else None})
    return rows


# ---------------------------------------------------------------- long context: prefill speed + needle recall
def bench_context(rows, save):
    for ctx in A.ctx:
        rng = random.Random(ctx)
        code = f"{rng.choice(WORDS)}-{rng.randint(1000, 9999)}"
        n_words = int((ctx - 300) * WORDS_PER_TOKEN)
        for depth in A.depths:
            text = filler(n_words, rng).split(". ")
            pos = int(len(text) * depth)
            text.insert(pos, f"The secret passphrase is {code}")
            doc = f"[doc {random.random():.8f}]\n" + ". ".join(text)
            q = doc + "\n\nWhat is the secret passphrase mentioned in the text above? Reply with the passphrase only."
            r = chat([{"role": "user", "content": q}], max_tokens=4096, think=True, temperature=0.6)
            t = r["timings"]
            ok = code in r["content"]
            row = {"ctx_target": ctx, "depth": depth, "prompt_tokens": t.get("prompt_n"),
                   "prefill_tps": t.get("prompt_per_second"), "prefill_s": (t.get("prompt_ms") or 0) / 1000,
                   "ttft_s": r["ttft_s"], "decode_tps": t.get("predicted_per_second"), "found": ok,
                   "answer": r["content"].strip()[:80]}
            rows.append(row)
            save()
            log(f"  ctx {ctx:>6} depth {depth:.0%}: {row['prompt_tokens']} tok prefill "
                f"{row['prefill_tps']:.0f} tok/s ({row['prefill_s']:.1f}s), decode {row['decode_tps']:.1f} tok/s, "
                f"needle " + ("FOUND" if ok else "MISSED: " + repr(row["answer"])))


# ---------------------------------------------------------------- coherence tasks
def num_in(text, want):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    return any(abs(float(n.replace(",", "").rstrip(".")) - want) < 1e-6 for n in nums if n.replace(",", "").rstrip("."))


def last_line(t):
    t = t.strip().splitlines()
    return t[-1] if t else ""


def run_code(code, tests):
    m = re.findall(r"```(?:python)?\n(.*?)```", code, re.S)
    src = (m[0] if m else code) + "\n\n" + tests + "\nprint('ALL_PASS')\n"
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.py")
        open(p, "w").write(src)
        try:
            r = subprocess.run([sys.executable, "-I", p], cwd=d, capture_output=True, text=True, timeout=10)
            return "ALL_PASS" in r.stdout, (r.stderr or r.stdout)[-300:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


WEATHER_TOOL = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                    "unit": {"type": "string", "enum": ["c", "f"]}},
                   "required": ["city"]}}}]


def t_math1():
    r = chat([{"role": "user", "content": "A shop sells pens at 3 for $4. Maria buys 27 pens and pays with a $50 bill. "
               "How much change does she get, in dollars? End with the number on its own line."}], 6000)
    return num_in(last_line(r["content"]), 14), r


def t_math2():
    r = chat([{"role": "user", "content": "What is the sum of all positive integers n < 100 such that n^2 + 1 is "
               "divisible by 5? End with the number on its own line."}], 8000)
    # n ≡ 2 or 3 mod 5: sum = 2+7+..+97 (20 terms, 990) + 3+8+..+98 (20 terms, 1010)
    return num_in(last_line(r["content"]), 2000), r


def t_logic():
    r = chat([{"role": "user", "content": "Alice, Bob and Cara each own exactly one pet: a cat, a dog or a fish. "
               "Alice doesn't own the dog. Bob owns neither the dog nor the cat. Who owns the cat? "
               "Answer with just the name."}], 4000)
    return "alice" in r["content"].lower() and "cara" not in r["content"].lower(), r


def t_trick():
    r = chat([{"role": "user", "content": "How many times does the letter r appear in the word 'strawberry'? "
               "Answer with the number only."}], 3000)
    return num_in(r["content"], 3), r


def t_code():
    r = chat([{"role": "user", "content": "Write a Python function `merge_intervals(iv: list[tuple[int,int]]) -> "
               "list[tuple[int,int]]` that merges overlapping or touching closed intervals and returns them sorted. "
               "Only the code in one ```python block."}], 6000)
    tests = ("assert merge_intervals([]) == []\n"
             "assert merge_intervals([(1,3),(2,6),(8,10),(15,18)]) == [(1,6),(8,10),(15,18)]\n"
             "assert merge_intervals([(5,7),(1,2),(2,4)]) == [(1,4),(5,7)]\n"
             "assert merge_intervals([(1,10),(2,3)]) == [(1,10)]\n")
    ok, err = run_code(r["content"], tests)
    r["note"] = err if not ok else ""
    return ok, r


def t_code2():
    r = chat([{"role": "user", "content": "Write a Python function `roman(n: int) -> str` converting 1..3999 to Roman "
               "numerals. Only the code in one ```python block."}], 6000)
    tests = "assert roman(1994)=='MCMXCIV'\nassert roman(3999)=='MMMCMXCIX'\nassert roman(4)=='IV'\nassert roman(58)=='LVIII'\n"
    ok, err = run_code(r["content"], tests)
    r["note"] = err if not ok else ""
    return ok, r


def t_json():
    r = chat([{"role": "user", "content": "Extract from this sentence a JSON object with keys name (string), age "
               "(integer) and cities (array of strings), nothing else, no code fence: "
               "'Jonas Berg, 41, has lived in Oslo, Lisbon and Kyoto.'"}], 3000)
    try:
        j = json.loads(r["content"].strip())
        return (j == {"name": "Jonas Berg", "age": 41, "cities": ["Oslo", "Lisbon", "Kyoto"]}), r
    except Exception:
        return False, r


def t_format():
    r = chat([{"role": "user", "content": "Give exactly 3 bullet points (lines starting with '- ') on why sleep matters. "
               "No intro, no outro, nothing else."}], 3000)
    lines = [x for x in r["content"].strip().splitlines() if x.strip()]
    return len(lines) == 3 and all(x.startswith("- ") for x in lines), r


def t_tool():
    r = chat([{"role": "user", "content": "What's the weather in Reykjavik right now, in celsius?"}], 3000,
             tools=WEATHER_TOOL)
    tc = r["tool_calls"]
    try:
        args = json.loads(tc[0]["arguments"]) if tc else {}
    except Exception:
        args = {}
    return bool(tc) and tc[0]["name"] == "get_weather" and "reykjav" in str(args.get("city", "")).lower(), r


def t_tool_followup():
    msgs = [{"role": "user", "content": "What's the weather in Reykjavik right now?"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "get_weather", "arguments": json.dumps({"city": "Reykjavik", "unit": "c"})}}]},
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"temp_c": -3, "sky": "snow", "wind_kmh": 41})}]
    r = chat(msgs, 3000, tools=WEATHER_TOOL)
    c = r["content"].lower()
    return ("-3" in c or "minus 3" in c or "−3" in c) and "snow" in c, r


def t_multiturn():
    msgs = [{"role": "user", "content": "My dog is called Pixel and my sister lives in Ghent. Just say OK."},
            {"role": "assistant", "content": "OK."},
            {"role": "user", "content": filler(400, random.Random(7)) + "\n\nIgnore the text above. Just say OK."},
            {"role": "assistant", "content": "OK."},
            {"role": "user", "content": "What is my dog's name and where does my sister live? One short sentence."}]
    r = chat(msgs, 3000)
    c = r["content"].lower()
    return "pixel" in c and "ghent" in c, r


def t_refuse_hallucinate():
    r = chat([{"role": "user", "content": "What was the exact closing price of the fictional company Zentravox Corp "
               "on 3 March 2019? If you can't know, say so in one sentence."}], 3000)
    c = r["content"].lower()
    return any(w in c for w in ("not", "no ", "can't", "cannot", "unable", "fictional", "don't")), r


TASKS = [("math: word problem", t_math1), ("math: number theory", t_math2), ("logic puzzle", t_logic),
         ("letter counting", t_trick), ("code: merge_intervals (tested)", t_code), ("code: roman (tested)", t_code2),
         ("strict JSON extraction", t_json), ("format: exactly 3 bullets", t_format), ("tool call", t_tool),
         ("tool result -> answer", t_tool_followup), ("multi-turn recall", t_multiturn),
         ("admits unknowable fact", t_refuse_hallucinate)]


def bench_coherence():
    rows = []
    for name, fn in TASKS:
        for i in range(A.coh_reps):
            try:
                ok, r = fn()
            except Exception as e:
                ok, r = False, {"content": f"ERROR {e}", "timings": {}, "reasoning": ""}
            t = r.get("timings", {})
            rows.append({"task": name, "pass": ok, "think_chars": len(r.get("reasoning", "")),
                         "tokens": t.get("predicted_n"), "decode_tps": t.get("predicted_per_second"),
                         "answer": (r.get("content") or (json.dumps(r["tool_calls"]) if r.get("tool_calls") else
                                    f"(no answer; finish={r.get('finish')}; thinking ends: {r.get('reasoning', '')[-100:]})"))[-160:].strip(),
                         "note": r.get("note", "")})
            log(f"  {'PASS' if ok else 'FAIL'}  {name:32s} {t.get('predicted_n') or 0:>5} tok  "
                f"{(t.get('predicted_per_second') or 0):5.1f} tok/s" + ("" if ok else f"  -> {rows[-1]['answer'][-90:]!r}"))
    return rows


def bench_longform():
    """One long free-form generation: does it stay on topic and avoid loops?"""
    r = chat([{"role": "user", "content": "Write a detailed 2500-word essay on the history of the printing press and "
               "its effects on Europe, with section headings. Do not repeat yourself."}], 5000, think=False)
    words = re.findall(r"\w+", r["content"].lower())
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    distinct4 = len(set(grams)) / max(1, len(grams))
    # longest repeated 12-word span = loop detector
    seen, loop = set(), 0
    for i in range(len(words) - 11):
        g = tuple(words[i:i + 12])
        if g in seen:
            loop += 1
        seen.add(g)
    t = r["timings"]
    out = {"words": len(words), "tokens": t.get("predicted_n"), "decode_tps": t.get("predicted_per_second"),
           "distinct_4gram": distinct4, "repeated_12gram_spans": loop, "finish": r["finish"],
           "on_topic": sum(w in words for w in ("gutenberg", "press", "printing", "europe", "reformation")) >= 4,
           "headings": len(re.findall(r"^#+ |\n#+ ", r["content"]))}
    log(f"  {out['words']} words, {out['decode_tps']:.1f} tok/s, distinct 4-grams {distinct4:.1%}, "
        f"repeated 12-gram spans {loop}, on topic {out['on_topic']}, headings {out['headings']}")
    return out, r["content"]


# ---------------------------------------------------------------- report
def report(res):
    L = [f"# Benchmark: {res['model']} @ {res['url_host']}", "", f"{res['date']} · {res['gpu']}", ""]
    if res.get("speed"):
        L += [f"## Decode speed (thinking off, median of {A.reps} runs)", "",
              "| Workload | Decode tok/s (min–max) | End-to-end tok/s | TTFT | Draft accepted |", "|---|---|---|---|---|"]
        for r in res["speed"]:
            acc = f"{r['draft_accept']:.0%}" if r["draft_accept"] is not None else "n/a"
            L.append(f"| {r['workload']} | **{r['decode_tps_median']:.1f}** ({r['decode_tps_min']:.1f}–"
                     f"{r['decode_tps_max']:.1f}) | {r['e2e_tps_median']:.1f} | {r['ttft_s_median']*1000:.0f} ms | {acc} |")
        L.append("")
    if res.get("context"):
        L += ["## Long context: prefill speed and needle recall", "",
              "| Prompt tokens | Needle depth | Prefill tok/s | Prefill time | TTFT | Decode tok/s | Needle |",
              "|---|---|---|---|---|---|---|"]
        for r in res["context"]:
            L.append(f"| {r['prompt_tokens']:,} | {r['depth']:.0%} | {r['prefill_tps']:.0f} | {r['prefill_s']:.1f} s | "
                     f"{r['ttft_s']:.1f} s | {r['decode_tps']:.1f} | {'found' if r['found'] else 'MISSED'} |")
        L.append("")
    if res.get("coherence"):
        tasks = {}
        for r in res["coherence"]:
            tasks.setdefault(r["task"], []).append(r)
        tot = sum(r["pass"] for r in res["coherence"])
        L += [f"## Coherence (thinking on): {tot}/{len(res['coherence'])} passed", "",
              "| Task | Passed | Avg tokens (incl. thinking) | Last answer |", "|---|---|---|---|"]
        for k, v in tasks.items():
            ans = v[-1]["answer"].replace("\n", " ").replace("|", "\\|")[-70:]
            L.append(f"| {k} | {sum(x['pass'] for x in v)}/{len(v)} | "
                     f"{statistics.mean(x['tokens'] or 0 for x in v):.0f} | {ans} |")
        L.append("")
    if res.get("longform"):
        f = res["longform"]
        L += ["## Long-form generation (thinking off)", "",
              f"{f['words']} words / {f['tokens']} tokens at {f['decode_tps']:.1f} tok/s · distinct 4-grams "
              f"{f['distinct_4gram']:.1%} · repeated 12-word spans: {f['repeated_12gram_spans']} · on topic: "
              f"{f['on_topic']} · headings: {f['headings']} · finish: {f['finish']}", ""]
    return "\n".join(L)


def main():
    global A
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True, help="base URL ending in /v1")
    p.add_argument("--key-file", help="file holding the API key (or set SUMMON_API_KEY)")
    p.add_argument("--model", default=None, help="model name (default: first from /v1/models)")
    p.add_argument("--ctx", type=lambda s: [int(x) for x in s.split(",")],
                   default=[1024, 8192, 32768, 131072, 245760], help="context sizes for prefill/needle tests")
    p.add_argument("--depths", type=lambda s: [float(x) for x in s.split(",")], default=[0.1, 0.5, 0.9])
    p.add_argument("--reps", type=int, default=3, help="runs per speed workload")
    p.add_argument("--coh-reps", type=int, default=2, help="runs per coherence task")
    p.add_argument("--only", default="speed,context,coherence,longform")
    p.add_argument("--quick", action="store_true", help="1 rep, contexts up to 32K, middle depth only")
    p.add_argument("--gpu", default="", help="label for the report, e.g. 'A100 80GB PCIe'")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--out", default="bench-results")
    A = p.parse_args()
    A.key = os.environ.get("SUMMON_API_KEY") or (open(A.key_file).read().strip() if A.key_file else "none")
    if A.quick:
        A.reps, A.coh_reps, A.ctx, A.depths = 1, 1, [1024, 8192, 32768], [0.5]
    if not A.model:
        req = urllib.request.Request(A.url.rstrip("/") + "/models", headers={"Authorization": "Bearer " + A.key})
        A.model = json.load(urllib.request.urlopen(req, timeout=30))["data"][0]["id"]
    only = A.only.split(",")
    res = {"model": A.model, "url_host": A.url,
           "gpu": A.gpu, "date": time.strftime("%Y-%m-%d %H:%M")}
    t0 = time.time()
    os.makedirs(A.out, exist_ok=True)
    stem = os.path.join(A.out, time.strftime("bench-%Y%m%d-%H%M%S"))

    def save():  # after every section and every long-context point, so Ctrl-C keeps what finished
        res["total_s"] = time.time() - t0
        json.dump(res, open(stem + ".json", "w"), indent=1)
        open(stem + ".md", "w").write(report(res))

    try:
        if "speed" in only:
            log("== decode speed"); res["speed"] = bench_speed(); save()
        if "coherence" in only:
            log("== coherence"); res["coherence"] = bench_coherence(); save()
        if "longform" in only:
            log("== long-form"); res["longform"], essay = bench_longform(); save()
            open(stem + "-essay.md", "w").write(essay)
        if "context" in only:
            log("== long context"); res["context"] = []; bench_context(res["context"], save)
    except KeyboardInterrupt:
        log("\nInterrupted: keeping the results so far.")
    save()
    log("\n" + report(res) + f"\nTotal {res['total_s']/60:.1f} min. Saved {stem}.md / .json")


if __name__ == "__main__":
    main()
