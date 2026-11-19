#!/usr/bin/env python3
"""Run a command in a pseudo-terminal, drive it with keystrokes and save PNG screenshots of the screen.

Used to make the README / website screenshots of Summon's dry runs (real whiptail, real script):
    python3 -I tools/termshot.py            # writes docs/img/*.png (summon-hero.png is composed separately)
A tiny VT100/xterm emulator (just what whiptail/newt and plain colored output use) feeds a PIL renderer.
"""
import os, pty, re, select, signal, struct, fcntl, termios, time, sys
from PIL import Image, ImageDraw, ImageFont

ROWS, COLS = 34, 124
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
BG = (17, 12, 26)
PALETTE = [(17, 12, 26), (224, 88, 104), (96, 200, 140), (230, 192, 110), (100, 148, 240), (178, 110, 232), (86, 196, 214), (214, 214, 222),
           (96, 88, 112), (255, 120, 136), (130, 230, 170), (255, 220, 140), (140, 180, 255), (226, 150, 255), (130, 226, 240), (250, 250, 255)]
ACS = {'j': '┘', 'k': '┐', 'l': '┌', 'm': '└', 'n': '┼', 'q': '─', 't': '├', 'u': '┤', 'v': '┴', 'w': '┬', 'x': '│',
       'a': '▒', '`': '◆', '~': '·', '0': '█', 'h': '▒', 'f': '°', 'g': '±', '+': '→', ',': '←', '-': '↑', '.': '↓', 'o': '⎺', 's': '⎽'}


def xterm256(n):
    if n < 16:
        return PALETTE[n]
    if n < 232:
        n -= 16
        lv = [0, 95, 135, 175, 215, 255]
        return (lv[n // 36], lv[(n // 6) % 6], lv[n % 6])
    g = 8 + (n - 232) * 10
    return (g, g, g)


class Screen:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cells = [[(' ', None, None, False) for _ in range(COLS)] for _ in range(ROWS)]
        self.r = self.c = 0
        self.fg = self.bg = None
        self.bold = self.rev = False
        self.acs = False
        self.g1acs = False
        self.so = False
        self.top, self.bot = 0, ROWS - 1
        self.buf = ''
        self.last = ' '

    def put(self, ch):
        if self.c >= COLS:
            self.c = 0
            self.lf()
        if (self.acs and not self.so) or (self.so and self.g1acs):
            ch = ACS.get(ch, ch)
        fg, bg = self.fg, self.bg
        if self.rev:
            fg, bg = (bg if bg is not None else 0), (fg if fg is not None else 7)
        self.cells[self.r][self.c] = (ch, fg, bg, self.bold)
        self.last = ch
        self.c += 1

    def lf(self):
        if self.r == self.bot:
            del self.cells[self.top]
            self.cells.insert(self.bot, [(' ', None, None, False) for _ in range(COLS)])
        elif self.r < ROWS - 1:
            self.r += 1

    def erase(self, r, c0, c1):
        for c in range(max(0, c0), min(COLS, c1)):
            self.cells[r][c] = (' ', self.fg if False else None, self.bg, False)

    def sgr(self, ps):
        ps = ps or [0]
        i = 0
        while i < len(ps):
            p = ps[i]
            if p == 0:
                self.fg = self.bg = None; self.bold = self.rev = False
            elif p == 1: self.bold = True
            elif p == 22: self.bold = False
            elif p == 7: self.rev = True
            elif p == 27: self.rev = False
            elif 30 <= p <= 37: self.fg = p - 30
            elif p == 39: self.fg = None
            elif 40 <= p <= 47: self.bg = p - 40
            elif p == 49: self.bg = None
            elif 90 <= p <= 97: self.fg = p - 90 + 8
            elif 100 <= p <= 107: self.bg = p - 100 + 8
            elif p in (38, 48) and i + 2 < len(ps) and ps[i + 1] == 5:
                if p == 38: self.fg = ('x', ps[i + 2])
                else: self.bg = ('x', ps[i + 2])
                i += 2
            i += 1

    def feed(self, data):
        s = self.buf + data
        self.buf = ''
        i = 0
        n = len(s)
        while i < n:
            ch = s[i]
            if ch == '\x1b':
                if i + 1 >= n:
                    self.buf = s[i:]; return
                nx = s[i + 1]
                if nx == '[':
                    m = re.match(r'\x1b\[([?>=]?)([0-9;]*)([ -/]*)([@-~])', s[i:])
                    if not m:
                        if len(s) - i < 32:
                            self.buf = s[i:]; return
                        i += 2; continue
                    priv, params, _, fin = m.groups()
                    ps = [int(x) if x else 0 for x in params.split(';')] if params else []
                    self.csi(priv, ps, fin)
                    i += m.end(); continue
                if nx in '()':
                    if i + 2 >= n:
                        self.buf = s[i:]; return
                    if nx == '(':
                        self.acs = s[i + 2] == '0'
                    else:
                        self.g1acs = s[i + 2] == '0'
                    i += 3; continue
                if nx == ']':
                    end = s.find('\x07', i)
                    end2 = s.find('\x1b\\', i)
                    ends = [e for e in (end, end2) if e != -1]
                    if not ends:
                        self.buf = s[i:]; return
                    e = min(ends)
                    i = e + (1 if e == end else 2); continue
                if nx == 'M':
                    if self.r == self.top:
                        del self.cells[self.bot]
                        self.cells.insert(self.top, [(' ', None, None, False) for _ in range(COLS)])
                    else:
                        self.r = max(0, self.r - 1)
                    i += 2; continue
                if nx == 'c':
                    self.reset(); i += 2; continue
                i += 2; continue
            if ch == '\r': self.c = 0
            elif ch == '\n': self.lf()
            elif ch == '\b': self.c = max(0, self.c - 1)
            elif ch == '\t': self.c = min(COLS - 1, (self.c // 8 + 1) * 8)
            elif ch == '\x0e': self.so = True
            elif ch == '\x0f': self.so = False
            elif ch == '\x07': pass
            elif ord(ch) >= 32: self.put(ch)
            i += 1

    def csi(self, priv, ps, f):
        p1 = ps[0] if ps else 0
        if priv:
            if f in 'hl' and any(p in (47, 1047, 1049) for p in ps):
                if f == 'h':   # switch to the alternate screen (whiptail), remember the normal one
                    self.saved = ([row[:] for row in self.cells], self.r, self.c)
                    self.cells = [[(' ', None, None, False) for _ in range(COLS)] for _ in range(ROWS)]
                elif getattr(self, 'saved', None):
                    self.cells, self.r, self.c = self.saved[0], self.saved[1], self.saved[2]
                    self.saved = None
            return
        if f in 'Hf':
            self.r = min(ROWS - 1, max(1, ps[0] if ps else 1) - 1)
            self.c = min(COLS - 1, max(1, ps[1] if len(ps) > 1 else 1) - 1)
        elif f == 'A': self.r = max(0, self.r - max(1, p1))
        elif f == 'B': self.r = min(ROWS - 1, self.r + max(1, p1))
        elif f == 'C': self.c = min(COLS - 1, self.c + max(1, p1))
        elif f == 'D': self.c = max(0, self.c - max(1, p1))
        elif f == 'G': self.c = min(COLS - 1, max(1, p1) - 1)
        elif f == 'd': self.r = min(ROWS - 1, max(1, p1) - 1)
        elif f == 'J':
            if p1 == 2 or p1 == 3:
                for r in range(ROWS): self.erase(r, 0, COLS)
            elif p1 == 0:
                self.erase(self.r, self.c, COLS)
                for r in range(self.r + 1, ROWS): self.erase(r, 0, COLS)
            elif p1 == 1:
                for r in range(self.r): self.erase(r, 0, COLS)
                self.erase(self.r, 0, self.c + 1)
        elif f == 'K':
            if p1 == 0: self.erase(self.r, self.c, COLS)
            elif p1 == 1: self.erase(self.r, 0, self.c + 1)
            else: self.erase(self.r, 0, COLS)
        elif f == 'X': self.erase(self.r, self.c, self.c + max(1, p1))
        elif f == 'b':
            for _ in range(max(1, p1)): self.put(self.last)
        elif f == 'm': self.sgr(ps)
        elif f == 'r':
            self.top = (ps[0] - 1) if ps else 0
            self.bot = (ps[1] - 1) if len(ps) > 1 else ROWS - 1
            self.r, self.c = 0, 0
        elif f == '@':
            row = self.cells[self.r]
            for _ in range(max(1, p1)):
                row.insert(self.c, (' ', None, self.bg, False)); row.pop()
        elif f == 'P':
            row = self.cells[self.r]
            for _ in range(max(1, p1)):
                row.pop(self.c); row.append((' ', None, self.bg, False))

    def color(self, v, default):
        if v is None: return default
        if isinstance(v, tuple): return xterm256(v[1])
        return PALETTE[v]

    def render(self, path, title):
        font = ImageFont.truetype(FONT, 16)
        fontb = ImageFont.truetype(FONT_BOLD, 16)
        cw = round(font.getlength('M')); ch = 21
        pad, bar = 18, 34
        W, H = COLS * cw + pad * 2, ROWS * ch + pad * 2 + bar
        img = Image.new('RGB', (W, H), (40, 30, 56))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, W - 1, H - 1], radius=12, fill=BG, outline=(70, 52, 98), width=2)
        d.rectangle([2, 2, W - 3, bar], fill=(34, 25, 48))
        for k, col in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
            d.ellipse([16 + k * 22, 11, 28 + k * 22, 23], fill=col)
        d.text((W // 2, bar // 2 + 1), title, font=font, fill=(200, 180, 230), anchor='mm')
        y0 = bar + pad
        for r in range(ROWS):
            for c in range(COLS):
                chr_, fg, bg, bold = self.cells[r][c]
                x, y = pad + c * cw, y0 + r * ch
                if bg is not None:
                    d.rectangle([x, y, x + cw - 1, y + ch - 1], fill=self.color(bg, BG))
                if chr_ != ' ':
                    fgc = fg
                    if bold and isinstance(fg, int) and fg < 8: fgc = fg + 8
                    col = self.color(fgc, PALETTE[7])
                    if chr_ == '█':
                        d.rectangle([x, y, x + cw - 1, y + ch - 1], fill=col)
                    elif chr_ in '─│┌┐└┘├┤┬┴┼':
                        self.boxdraw(d, chr_, x, y, cw, ch, col)
                    elif chr_ in '═║╔╗╚╝':
                        self.dbldraw(d, chr_, x, y, cw, ch, col)
                    else:
                        d.text((x, y + 2), chr_, font=fontb if bold else font, fill=col)
        img.save(path)

    @staticmethod
    def dbldraw(d, chr_, x, y, cw, ch, col):
        # double lines, drawn edge to edge like a terminal does (two strokes 4 px apart)
        cx, cy, o = x + cw // 2, y + ch // 2, 2
        L = chr_ in '═╗╝'; R = chr_ in '═╔╚'; U = chr_ in '║╚╝'; D = chr_ in '║╔╗'
        for k in (-o, o):
            if L or R:
                xa = x if L else cx + (k if chr_ in '╔╚' and False else 0)
                xa = x if L else (cx - k if chr_ == '╔' else cx + k if chr_ == '╚' else cx)
                xb = x + cw if R else (cx - k if chr_ == '╝' else cx + k if chr_ == '╗' else cx)
                if chr_ == '═': xa, xb = x, x + cw
                d.line([xa, cy + k, xb, cy + k], fill=col, width=1)
            if U or D:
                ya = y if U else (cy - k if chr_ == '╔' else cy + k if chr_ == '╗' else cy)
                yb = y + ch if D else (cy - k if chr_ == '╝' else cy + k if chr_ == '╚' else cy)
                if chr_ == '║': ya, yb = y, y + ch
                d.line([cx + k, ya, cx + k, yb], fill=col, width=1)

    @staticmethod
    def boxdraw(d, chr_, x, y, cw, ch, col):
        cx, cy = x + cw // 2, y + ch // 2
        L = chr_ in '─┐┘┤┬┴┼'; R = chr_ in '─┌└├┬┴┼'
        U = chr_ in '│└┘├┤┴┼'; D = chr_ in '│┌┐├┤┬┼'
        if L: d.line([x, cy, cx, cy], fill=col, width=2)
        if R: d.line([cx, cy, x + cw, cy], fill=col, width=2)
        if U: d.line([cx, y, cx, cy], fill=col, width=2)
        if D: d.line([cx, cy, cx, y + ch], fill=col, width=2)


def run(cmd, env, steps, outdir):
    import codecs
    dec = codecs.getincrementaldecoder('utf-8')('replace')   # multi-byte characters can be split across reads
    pid, fd = pty.fork()
    if pid == 0:
        os.execvpe('bash', ['bash', '-c', cmd], env)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack('HHHH', ROWS, COLS, 0, 0))
    scr = Screen()
    text = ''

    def pump(sec):
        nonlocal text
        end = time.time() + sec
        while time.time() < end:
            rl, _, _ = select.select([fd], [], [], 0.05)
            if fd in rl:
                try:
                    raw = os.read(fd, 65536)
                    data = dec.decode(raw)
                except OSError:
                    return False
                if not raw:
                    return False
                text += data
                scr.feed(data)
        return True

    for st in steps:
        kind = st[0]
        if kind == 'wait':
            pump(st[1])
        elif kind == 'until':  # wait until a regex shows up in the raw output (or timeout)
            pat, timeout = st[1], st[2]
            end = time.time() + timeout
            while time.time() < end and not re.search(pat, text):
                if not pump(0.2): break
            pump(0.6)
        elif kind == 'key':
            os.write(fd, st[1].encode())
        elif kind == 'shot':
            pump(0.3)
            scr.render(os.path.join(outdir, st[1]), st[2])
            print('saved', st[1])
    pump(1.0)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass
    return text


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    out = os.path.join(root, 'docs', 'img')
    os.makedirs(out, exist_ok=True)
    S = './summon.sh'   # run from the project folder, so no local paths end up in the screenshots
    base = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'USER')}
    base.update(TERM='xterm-256color', LANG='C.UTF-8', LC_ALL='C.UTF-8', COLUMNS=str(COLS), LINES=str(ROWS), PS1='$ ')
    ENTER, DOWN, SPACE = '\r', '\x1b[B', ' '

    # 0) Banner + plan in one screen: CTX/QUANT/port given, so no menus
    env = dict(base, SUMMON_GPU='NVIDIA RTX A6000:49140', CTX='131072', QUANT='Q6_K')
    run(f'cd {root}; clear; echo "$ CTX=131072 QUANT=Q6_K ./summon.sh --dry-run 40000"; "{S}" --dry-run 40000; sleep 30', env, [
        ('until', r'Same, without menus', 15), ('shot', '00-banner-dry-run.png', 'summon.sh --dry-run  (RTX A6000, 48 GB)'),
    ], out)
    # 1) A6000: model menu, port, context, weights, final plan
    env = dict(base, SUMMON_GPU='NVIDIA RTX A6000:49140')
    run(f'cd {root}; clear; echo "$ ./summon.sh --dry-run"; "{S}" --dry-run; sleep 30', env, [
        ('until', r'Which AI', 15), ('shot', '01-model-menu-a6000.png', 'summon.sh --dry-run  (RTX A6000, 48 GB)'),
        ('key', ENTER), ('until', r'Trade precision for speed', 10), ('shot', '11-draft-question.png', 'summon.sh --dry-run  (RTX A6000, 48 GB)'),
        ('key', ENTER), ('until', r'Step 1/2', 10), ('shot', '03-context-menu.png', 'summon.sh --dry-run  (RTX A6000, 48 GB)'),
        ('key', ENTER), ('until', r'Step 2/2', 10), ('shot', '04-weights-menu.png', 'summon.sh --dry-run  (RTX A6000, 48 GB)'),
        ('key', ENTER), ('until', r'Last step: port', 10), ('shot', '02-port.png', 'summon.sh --dry-run  (RTX A6000, 48 GB)'),
        ('key', ENTER), ('until', r'Same, without menus', 10), ('shot', '05-dry-run-plan.png', 'summon.sh --dry-run  (RTX A6000, 48 GB)'),
    ], out)
    # 2) RTX 3090: the model menu pre-selects Ornith 9B
    env = dict(base, SUMMON_GPU='NVIDIA GeForce RTX 3090:24576', SUMMON_RAM='65536')   # a 64 GB RAM box: Ornith 35B fits with experts in RAM
    run(f'cd {root}; clear; "{S}" --dry-run 40000; sleep 30', env, [
        ('until', r'Which AI', 15), ('shot', '06-model-menu-3090.png', 'summon.sh --dry-run 40000  (RTX 3090, 24 GB)'),
        ('key', ENTER), ('until', r'Step 1/2', 10), ('key', ENTER), ('until', r'Step 2/2', 10),
        ('shot', '07-weights-menu-3090.png', 'summon.sh --dry-run 40000  (RTX 3090, 24 GB)'),
        ('key', ENTER), ('until', r'Same, without menus', 10), ('shot', '08-dry-run-plan-3090.png', 'summon.sh --dry-run 40000  (RTX 3090, 24 GB)'),
    ], out)
    # 3) Ollama model checklist
    env = dict(base, SUMMON_GPU='NVIDIA RTX A6000:49140')
    run(f'cd {root}; clear; "{S}" --dry-run ollama 40000; sleep 30', env, [
        ('until', r'Which AI', 25), ('shot', '09-ollama-menu.png', 'summon.sh --dry-run ollama 40000  (RTX A6000, 48 GB)'),
        ('key', ENTER), ('until', r'Endpoint:', 15), ('shot', '10-ollama-plan.png', 'summon.sh --dry-run ollama 40000  (RTX A6000, 48 GB)'),
    ], out)
