#!/usr/bin/env python3
"""Draw the README hero: a small pixel library, upscaled with nearest neighbour."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "hero-pixel.png"

W, H = 64, 40
SCALE = 12

SKY = (22, 32, 58)
MOON = (240, 230, 196)
STAR = (232, 226, 206)
WALL = (46, 42, 64)
SHELF = (92, 58, 40)
TABLE = (74, 48, 36)
PAGE = (244, 239, 226)
COVER = (90, 52, 40)
INK = (42, 40, 48)
OWL = (236, 236, 230)
EYE = (20, 20, 24)
BEAK = (212, 148, 64)
GLITCH_A = (57, 255, 106)
GLITCH_B = (226, 61, 140)
BOOKS = [
    (176, 72, 64),
    (54, 102, 140),
    (214, 186, 120),
    (42, 110, 84),
    (122, 78, 148),
    (196, 122, 64),
]


def main() -> None:
    img = Image.new("RGB", (W, H), SKY)
    px = img.load()

    def rect(x0, y0, x1, y1, color):
        for y in range(y0, y1):
            for x in range(x0, x1):
                if 0 <= x < W and 0 <= y < H:
                    px[x, y] = color

    def dot(x, y, color):
        if 0 <= x < W and 0 <= y < H:
            px[x, y] = color

    # Night sky and a moon.
    rect(0, 0, W, 16, SKY)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            if dx * dx + dy * dy <= 10:
                dot(52 + dx, 6 + dy, MOON)
    for sx, sy in ((8, 3), (18, 7), (30, 4), (40, 9), (14, 11)):
        dot(sx, sy, STAR)

    # Wall, two shelf boards, a table.
    rect(0, 15, W, 34, WALL)
    rect(1, 21, 63, 23, SHELF)
    rect(1, 30, 63, 32, SHELF)
    rect(0, 34, W, H, TABLE)

    # Upper row of books. One spine is a glitch stripe.
    x = 3
    for i, color in enumerate(BOOKS):
        width = 4 if i != 4 else 5
        rect(x, 15, x + width - 1, 21, color)
        if i == 4:
            for yy in range(16, 21):
                dot(x + (yy % 2), yy, GLITCH_A if yy % 2 == 0 else GLITCH_B)
        x += width + 1

    # Lower row, ordinary spines, leaving the left side clear for the owl.
    x = 22
    for color in reversed(BOOKS):
        rect(x, 24, x + 4, 30, color)
        x += 6

    # Open book on the right. A few pixels on the right page are the wrong colour.
    rect(28, 32, 58, 39, COVER)
    rect(30, 33, 43, 38, PAGE)
    rect(44, 33, 56, 38, PAGE)
    for yy in (34, 36):
        for xx in range(32, 41, 2):
            dot(xx, yy, INK)
        for xx in range(46, 54, 2):
            dot(xx, yy, INK)
    dot(47, 34, GLITCH_B)
    dot(51, 36, GLITCH_A)
    dot(49, 34, GLITCH_A)

    # Snowy owl standing on the table, left of the book. One eye is off-colour.
    owl = [
        "..wwwwww..",
        ".wwwwwwww.",
        ".wwbwwbww.",
        ".wwwwwwww.",
        "..wwyyww..",
        "...wwww...",
        "..wwwwww..",
        ".wwwwwwww.",
        "wwwwwwwwww",
        ".ww....ww.",
    ]
    ox, oy = 8, 29
    for row, line in enumerate(owl):
        for col, ch in enumerate(line):
            if ch == "w":
                dot(ox + col, oy + row, OWL)
            elif ch == "b":
                dot(ox + col, oy + row, EYE)
            elif ch == "y":
                dot(ox + col, oy + row, BEAK)
    dot(ox + 3, oy + 2, GLITCH_B)

    big = img.resize((W * SCALE, H * SCALE), Image.Resampling.NEAREST)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    big.save(OUT, format="PNG")
    print(f"wrote {OUT.relative_to(ROOT)} {big.size}")


if __name__ == "__main__":
    main()
