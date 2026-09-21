#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把终端日志里的指定片段渲染成 PNG —— 方块搬运卡片的教学配图用。

它渲染的是你**真实跑出来的日志内容**，只是把文字排版成图片，
省掉在终端里掐时间按截图键的麻烦。

用法:
  python log2png.py <日志文件> -o <输出.png> [--slice A B]...

  --slice A B   取"从含 A 的那一行"到"含 B 的那一行"（含两端）；B 省略则取到末尾
  可以给多个 --slice，按给的顺序拼接，块与块之间插一行省略标记

不写 --slice 时，用 --head N / --tail N 取开头或结尾。

示例:
  python log2png.py run_l2r.log -o 03_单独左到右.png \
      --slice "checkpoint=" "WARNING" \
      --slice "Connecting left arm" "All connected" \
      --slice "piper-auto-reset" "自动归位与推理都已成功完成"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# 7 = "Noto Sans Mono CJK SC"（等宽，中文和英文都能显示）
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_INDEX = 7

BG = (30, 30, 30)
FG = (212, 212, 212)
C_WARN = (232, 190, 92)
C_BAD = (242, 118, 108)
C_OK = (140, 210, 140)
C_DIM = (130, 130, 130)

PAD = 18
LINE_SPACING = 1.38
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def color_for(line: str):
    u = line.upper()
    if "WARNING" in u or "⚠" in line:
        return C_WARN
    if "STOPPED" in u or "SAFETY STOP" in u or "ERROR" in u or "FAIL" in u or "Traceback" in line:
        return C_BAD
    if ("ALL CONNECTED" in u or "COMPLETE" in u or "MODEL_READY" in u
            or "WARMUP_OK" in u or "DONE" in u):
        return C_OK
    if line.strip() == "" or set(line.strip()) <= {"…", ".", "-", "="}:
        return C_DIM
    return FG


def clean(raw: str):
    """去掉 ANSI 转义、\r 进度行、行尾空格。"""
    out = []
    for line in raw.split("\n"):
        line = ANSI_RE.sub("", line)
        if "\r" in line:
            parts = [p for p in line.split("\r") if p.strip()]
            line = parts[-1] if parts else ""
        out.append(line.rstrip())
    return out


def take(lines, startswith, ends, cursor):
    """从 cursor 往后找 startswith，再找到 ends。返回 (片段, 新 cursor)。"""
    start = None
    for i in range(cursor, len(lines)):
        if startswith in lines[i]:
            start = i
            break
    if start is None:
        return None, cursor
    if ends is None:
        return lines[start:], len(lines)
    for j in range(start, len(lines)):
        if ends in lines[j]:
            return lines[start:j + 1], j + 1
    return lines[start:], len(lines)


def render(lines, out_path, fontsize):
    font = ImageFont.truetype(FONT_PATH, fontsize, index=FONT_INDEX)
    line_h = int(fontsize * LINE_SPACING)
    widest = 0
    for l in lines:
        widest = max(widest, font.getlength(l))
    w = int(widest) + PAD * 2
    h = line_h * len(lines) + PAD * 2

    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)
    y = PAD
    for l in lines:
        draw.text((PAD, y), l, font=font, fill=color_for(l))
        y += line_h
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--slice", nargs="+", action="append", default=[],
                    metavar=("START", "END"))
    ap.add_argument("--head", type=int)
    ap.add_argument("--tail", type=int)
    ap.add_argument("--fontsize", type=int, default=20)
    ap.add_argument("--max-lines", type=int, default=60)
    args = ap.parse_args()

    lines = clean(Path(args.log).read_text(errors="replace"))

    picked = []
    missing = []

    if args.slice:
        cursor = 0
        for i, spec in enumerate(args.slice):
            start = spec[0]
            end = spec[1] if len(spec) > 1 else None
            chunk, cursor = take(lines, start, end, cursor)
            if chunk is None:
                missing.append(start)
                continue
            if picked:
                picked.append("  …")
            picked.extend(chunk)
    elif args.head:
        picked = lines[:args.head]
    elif args.tail:
        picked = lines[-args.tail:]
    else:
        picked = lines

    while picked and picked[0].strip() == "":
        picked.pop(0)
    while picked and picked[-1].strip() == "":
        picked.pop()

    if not picked:
        print("没有取到任何内容，检查 --slice 的关键词。", file=sys.stderr)
        return 1

    if len(picked) > args.max_lines:
        print(f"注意：取了 {len(picked)} 行，超过 --max-lines={args.max_lines}，"
              f"图会很长。可把 --slice 的范围收窄。", file=sys.stderr)

    w, h = render(picked, args.out, args.fontsize)
    print(f"已生成: {args.out}  ({w}x{h}, {len(picked)} 行, {len(args.slice) or 1} 段)")
    for m in missing:
        print(f"  警告: 日志里没找到 '{m}'", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
