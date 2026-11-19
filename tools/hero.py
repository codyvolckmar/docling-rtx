#!/usr/bin/env python3
"""Compose docs/img/summon-hero.png (16:9, for X / link previews) from the termshot screenshots.
Run tools/termshot.py first. Crop boxes are in the 1276x784 screenshot pixels: just the whiptail dialogs."""
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, 'docs', 'img')
W, H = 1600, 900
B = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'
R = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
RB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
PANELS = [
    ('01-model-menu-a6000.png', (70, 222, 1205, 412), 'llama.cpp', ['Nanbeige 4.3 33B', 'Qwen3.8 27B pi', 'Ornith 1.5 35B / 9B'],
     'Pre-selected for your card:\n128K context, Q6_K weights'),
    ('09-ollama-menu.png', (40, 180, 1240, 522), 'Ollama', ['Qwen3.8 27B', 'Qwen3.6, Gemma 4', '+ more'],
     'Ranked with independent\nbenchmarks, sized to your VRAM'),
]

img = Image.new('RGB', (W, H), (14, 10, 22))
glow = Image.new('RGB', (W, H), (14, 10, 22))
ImageDraw.Draw(glow).ellipse([-200, -300, 900, 300], fill=(70, 32, 120))
img = Image.blend(img, glow.filter(ImageFilter.GaussianBlur(140)), 0.9)
d = ImageDraw.Draw(img)
title, tag = ImageFont.truetype(B, 64), ImageFont.truetype(B, 20)
h1, cap, url = ImageFont.truetype(RB, 30), ImageFont.truetype(R, 21), ImageFont.truetype(B, 24)
mask = Image.new('L', (W, H), 0)
ImageDraw.Draw(mask).text((40, 30), 'SUMMON', font=title, fill=255)
grad = Image.new('RGB', (W, H))
gd = ImageDraw.Draw(grad)
for y in range(H):
    k = min(1, max(0, (y - 30) / 70))
    gd.line([0, y, W, y], fill=(int(232 - 60 * k), int(160 - 80 * k), 255))
img.paste(grad, (0, 0), mask)
d.text((330, 54), 'AI · OPEN-SOURCE · PRIVATE · INTELLIGENCE', font=tag, fill=(111, 214, 234))
d.text((330, 82), 'One command turns your NVIDIA GPU into a private, OpenAI-compatible AI.', font=cap, fill=(214, 206, 228))

PW, y = 1060, 160
for name, box, eng, models, note in PANELS:
    im = Image.open(os.path.join(IMG, name)).convert('RGB').crop(box)
    ph = int(im.height * PW / im.width)
    im = im.resize((PW, ph), Image.LANCZOS)
    x = 40
    sh = Image.new('RGBA', (PW + 64, ph + 64), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([20, 20, PW + 44, ph + 44], 16, fill=(0, 0, 0, 170))
    img.paste(sh.filter(ImageFilter.GaussianBlur(12)), (x - 32, y - 26), sh.filter(ImageFilter.GaussianBlur(12)))
    fr = Image.new('RGB', (PW + 24, ph + 24), (17, 12, 26))
    ImageDraw.Draw(fr).rounded_rectangle([0, 0, PW + 23, ph + 23], 14, outline=(90, 62, 130), width=2)
    img.paste(fr, (x - 12, y - 12))
    img.paste(im, (x, y))
    cx, cy = x + PW + 50, y + 2
    d.text((cx, cy), eng, font=h1, fill=(111, 214, 234)); cy += 44
    for m in models:
        d.text((cx, cy), m, font=h1, fill=(226, 155, 255)); cy += 38
    d.multiline_text((cx, cy + 8), note, font=cap, fill=(169, 159, 189), spacing=6)
    y += max(ph, cy + 70 - y) + 40
d.rounded_rectangle([40, H - 74, 620, H - 28], 12, fill=(30, 20, 46), outline=(110, 70, 160), width=2)
d.text((330, H - 51), 'codyvolckmar.github.io/summon', font=url, fill=(123, 227, 166), anchor='mm')
d.text((W - 40, H - 51), 'Linux · Windows (WSL 2) · MIT', font=cap, fill=(169, 159, 189), anchor='rm')
out = os.path.join(IMG, 'summon-hero.png')
img.save(out)
print('saved', out, '- panels end at y =', y - 40)
