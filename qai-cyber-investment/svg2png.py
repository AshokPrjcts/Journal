"""
svg2png.py — Minimal, ZERO-DEPENDENCY SVG rasterizer (standard library only).

Renders the specific SVG primitives emitted by this project's figures
(rect, line, polyline, circle, text) to a true-colour PNG. Text is drawn with a
built-in 5x7 bitmap font scaled to the requested size. This exists only because
the offline sandbox has no cairosvg/rsvg/Inkscape and no network to install one.

Not a general SVG engine — it covers exactly what figure1/figure2/graphical
abstract use, which is sufficient to embed raster figures into the DOCX/PDF.

API:
    rasterize(svg_path, png_path, scale=2.0)
"""

from __future__ import annotations

import re
import struct
import xml.dom.minidom as minidom
import zlib

# --------------------------------------------------------------------------- #
# 5x7 bitmap font (uppercase, lowercase, digits, common punctuation)
# Each glyph is 5 columns x 7 rows, encoded as 7 strings of 5 chars.
# --------------------------------------------------------------------------- #
FONT = {
 ' ': ["     "]*7,
 'A': ["01110","10001","10001","11111","10001","10001","10001"],
 'B': ["11110","10001","11110","10001","10001","10001","11110"],
 'C': ["01111","10000","10000","10000","10000","10000","01111"],
 'D': ["11110","10001","10001","10001","10001","10001","11110"],
 'E': ["11111","10000","11110","10000","10000","10000","11111"],
 'F': ["11111","10000","11110","10000","10000","10000","10000"],
 'G': ["01111","10000","10000","10111","10001","10001","01111"],
 'H': ["10001","10001","11111","10001","10001","10001","10001"],
 'I': ["11111","00100","00100","00100","00100","00100","11111"],
 'J': ["00111","00010","00010","00010","00010","10010","01100"],
 'K': ["10001","10010","11100","10010","10001","10001","10001"],
 'L': ["10000","10000","10000","10000","10000","10000","11111"],
 'M': ["10001","11011","10101","10101","10001","10001","10001"],
 'N': ["10001","11001","10101","10011","10001","10001","10001"],
 'O': ["01110","10001","10001","10001","10001","10001","01110"],
 'P': ["11110","10001","10001","11110","10000","10000","10000"],
 'Q': ["01110","10001","10001","10001","10101","10010","01101"],
 'R': ["11110","10001","10001","11110","10100","10010","10001"],
 'S': ["01111","10000","10000","01110","00001","00001","11110"],
 'T': ["11111","00100","00100","00100","00100","00100","00100"],
 'U': ["10001","10001","10001","10001","10001","10001","01110"],
 'V': ["10001","10001","10001","10001","10001","01010","00100"],
 'W': ["10001","10001","10001","10101","10101","11011","10001"],
 'X': ["10001","10001","01010","00100","01010","10001","10001"],
 'Y': ["10001","10001","01010","00100","00100","00100","00100"],
 'Z': ["11111","00001","00010","00100","01000","10000","11111"],
 'a': ["00000","00000","01110","00001","01111","10001","01111"],
 'b': ["10000","10000","11110","10001","10001","10001","11110"],
 'c': ["00000","00000","01111","10000","10000","10000","01111"],
 'd': ["00001","00001","01111","10001","10001","10001","01111"],
 'e': ["00000","00000","01110","10001","11111","10000","01110"],
 'f': ["00110","01000","11110","01000","01000","01000","01000"],
 'g': ["00000","01111","10001","10001","01111","00001","01110"],
 'h': ["10000","10000","11110","10001","10001","10001","10001"],
 'i': ["00100","00000","01100","00100","00100","00100","01110"],
 'j': ["00010","00000","00110","00010","00010","10010","01100"],
 'k': ["10000","10000","10010","10100","11000","10100","10010"],
 'l': ["01100","00100","00100","00100","00100","00100","01110"],
 'm': ["00000","00000","11010","10101","10101","10001","10001"],
 'n': ["00000","00000","11110","10001","10001","10001","10001"],
 'o': ["00000","00000","01110","10001","10001","10001","01110"],
 'p': ["00000","11110","10001","10001","11110","10000","10000"],
 'q': ["00000","01111","10001","10001","01111","00001","00001"],
 'r': ["00000","00000","10110","11000","10000","10000","10000"],
 's': ["00000","00000","01111","10000","01110","00001","11110"],
 't': ["01000","01000","11110","01000","01000","01001","00110"],
 'u': ["00000","00000","10001","10001","10001","10011","01101"],
 'v': ["00000","00000","10001","10001","10001","01010","00100"],
 'w': ["00000","00000","10001","10001","10101","10101","01010"],
 'x': ["00000","00000","10001","01010","00100","01010","10001"],
 'y': ["00000","10001","10001","01111","00001","00001","01110"],
 'z': ["00000","00000","11111","00010","00100","01000","11111"],
 '0': ["01110","10011","10101","10101","11001","10001","01110"],
 '1': ["00100","01100","00100","00100","00100","00100","01110"],
 '2': ["01110","10001","00001","00110","01000","10000","11111"],
 '3': ["11110","00001","00001","01110","00001","00001","11110"],
 '4': ["00010","00110","01010","10010","11111","00010","00010"],
 '5': ["11111","10000","11110","00001","00001","10001","01110"],
 '6': ["00110","01000","10000","11110","10001","10001","01110"],
 '7': ["11111","00001","00010","00100","01000","01000","01000"],
 '8': ["01110","10001","10001","01110","10001","10001","01110"],
 '9': ["01110","10001","10001","01111","00001","00010","01100"],
 '.': ["00000","00000","00000","00000","00000","01100","01100"],
 ',': ["00000","00000","00000","00000","01100","00100","01000"],
 ':': ["00000","01100","01100","00000","01100","01100","00000"],
 ';': ["00000","01100","01100","00000","01100","00100","01000"],
 '-': ["00000","00000","00000","11111","00000","00000","00000"],
 '+': ["00000","00100","00100","11111","00100","00100","00000"],
 '%': ["11001","11010","00010","00100","01000","01011","10011"],
 '(': ["00010","00100","01000","01000","01000","00100","00010"],
 ')': ["01000","00100","00010","00010","00010","00100","01000"],
 '/': ["00001","00010","00010","00100","01000","01000","10000"],
 '&': ["01100","10010","10100","01000","10101","10010","01101"],
 '#': ["01010","01010","11111","01010","11111","01010","01010"],
 '=': ["00000","00000","11111","00000","11111","00000","00000"],
 '[': ["01110","01000","01000","01000","01000","01000","01110"],
 ']': ["01110","00010","00010","00010","00010","00010","01110"],
 "'": ["00100","00100","01000","00000","00000","00000","00000"],
 '\u2013': ["00000","00000","00000","11111","00000","00000","00000"],  # en dash
 '\u2014': ["00000","00000","00000","11111","00000","00000","00000"],  # em dash
 '\u2212': ["00000","00000","00000","11111","00000","00000","00000"],  # minus
 '\u2022': ["00000","00000","01110","01110","01110","00000","00000"],  # bullet
 '\u03b3': ["00000","00000","10001","01010","00100","00100","01000"],  # gamma
 '\u03b8': ["01110","10001","10001","11111","10001","10001","01110"],  # theta
 '\u2208': ["00111","01000","01000","01110","01000","01000","00111"],  # in
 '\u00b2': ["11000","00100","01000","11100","00000","00000","00000"],  # ^2
 '\u00b8': ["     "]*7, '\u00b7': ["00000","00000","00000","00110","00110","00000","00000"],
 '\u00b1': ["00100","00100","11111","00100","11111","00000","00000"],  # plus-minus
}
DEFAULT_GLYPH = ["11111","10001","10001","10001","10001","10001","11111"]


def _hex(c):
    c = (c or "").strip()
    if c in ("none", ""):
        return None
    named = {"white": (255, 255, 255), "black": (0, 0, 0)}
    if c in named:
        return named[c]
    if c.startswith("#"):
        c = c[1:]
        if len(c) == 3:
            c = "".join(ch * 2 for ch in c)
        return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))
    return (80, 80, 80)


class Canvas:
    def __init__(self, w, h, bg=(255, 255, 255)):
        self.w, self.h = w, h
        self.buf = bytearray()
        for _ in range(w * h):
            self.buf += bytes(bg)

    def _set(self, x, y, rgb, a=1.0):
        x, y = int(round(x)), int(round(y))
        if 0 <= x < self.w and 0 <= y < self.h:
            i = (y * self.w + x) * 3
            if a >= 1.0:
                self.buf[i:i + 3] = bytes(rgb)
            else:
                for k in range(3):
                    self.buf[i + k] = int(self.buf[i + k] * (1 - a) + rgb[k] * a)

    def fill_rect(self, x, y, w, h, rgb):
        for yy in range(int(y), int(y + h)):
            for xx in range(int(x), int(x + w)):
                self._set(xx, yy, rgb)

    def stroke_rect(self, x, y, w, h, rgb, sw=1):
        self.line(x, y, x + w, y, rgb, sw)
        self.line(x, y + h, x + w, y + h, rgb, sw)
        self.line(x, y, x, y + h, rgb, sw)
        self.line(x + w, y, x + w, y + h, rgb, sw)

    def line(self, x0, y0, x1, y1, rgb, sw=1, dash=None):
        dx, dy = x1 - x0, y1 - y0
        dist = max(abs(dx), abs(dy), 1)
        steps = int(dist) + 1
        half = max(0, (sw - 1) / 2.0)
        for s in range(steps + 1):
            t = s / steps
            if dash:
                on, off = dash
                if (int(t * dist) % (on + off)) >= on:
                    continue
            cx = x0 + dx * t
            cy = y0 + dy * t
            r = int(half)
            for ox in range(-r, r + 1):
                for oy in range(-r, r + 1):
                    self._set(cx + ox, cy + oy, rgb)

    def circle(self, cx, cy, r, rgb):
        for yy in range(int(cy - r), int(cy + r + 1)):
            for xx in range(int(cx - r), int(cx + r + 1)):
                if (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r:
                    self._set(xx, yy, rgb)

    def text(self, x, y, s, rgb, size=12, anchor="start", bold=False):
        # cell: glyph scaled; baseline roughly at y
        px = max(1, int(round(size / 7.0)))
        gw = 5 * px
        gh = 7 * px
        spacing = px  # inter-char gap
        total = len(s) * (gw + spacing)
        if anchor == "middle":
            x -= total / 2
        elif anchor == "end":
            x -= total
        top = y - gh  # place glyph above baseline
        cx = x
        for ch in s:
            glyph = FONT.get(ch, DEFAULT_GLYPH)
            for ry, row in enumerate(glyph):
                for rx, bit in enumerate(row):
                    if bit == "1":
                        for ox in range(px):
                            for oy in range(px):
                                self._set(cx + rx * px + ox, top + ry * px + oy, rgb)
                                if bold:
                                    self._set(cx + rx * px + ox + 1, top + ry * px + oy, rgb)
            cx += gw + spacing

    def png_bytes(self):
        raw = bytearray()
        for y in range(self.h):
            raw.append(0)  # no filter
            raw += self.buf[y * self.w * 3:(y + 1) * self.w * 3]
        comp = zlib.compress(bytes(raw), 9)

        def chunk(typ, data):
            c = struct.pack(">I", len(data)) + typ + data
            return c + struct.pack(">I", zlib.crc32(typ + data) & 0xffffffff)

        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", self.w, self.h, 8, 2, 0, 0, 0)
        return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", comp) + chunk(b"IEND", b"")


def _attr(node, name, default=None):
    return node.getAttribute(name) or default


def rasterize(svg_path, png_path, scale=2.0):
    doc = minidom.parse(svg_path)
    svg = doc.getElementsByTagName("svg")[0]
    W = int(float(_attr(svg, "width", "800")))
    H = int(float(_attr(svg, "height", "600")))
    cv = Canvas(int(W * scale), int(H * scale))

    def sx(v): return float(v) * scale

    for node in svg.getElementsByTagName("*"):
        tag = node.tagName
        if tag == "rect":
            x, y = sx(_attr(node, "x", "0")), sx(_attr(node, "y", "0"))
            w, h = sx(_attr(node, "width", "0")), sx(_attr(node, "height", "0"))
            fill = _hex(_attr(node, "fill", "none"))
            if fill:
                cv.fill_rect(x, y, w, h, fill)
            stroke = _hex(_attr(node, "stroke"))
            if stroke:
                cv.stroke_rect(x, y, w, h, stroke, max(1, int(sx(_attr(node, "stroke-width", "1")))))
        elif tag == "line":
            stroke = _hex(_attr(node, "stroke", "#000")) or (80, 80, 80)
            sw = max(1, int(sx(_attr(node, "stroke-width", "1"))))
            dash = _attr(node, "stroke-dasharray")
            d = None
            if dash:
                parts = [int(float(p)) for p in re.split(r"[ ,]+", dash.strip()) if p]
                if len(parts) >= 2:
                    d = (int(parts[0] * scale), int(parts[1] * scale))
            cv.line(sx(_attr(node, "x1", "0")), sx(_attr(node, "y1", "0")),
                    sx(_attr(node, "x2", "0")), sx(_attr(node, "y2", "0")), stroke, sw, d)
        elif tag == "polyline":
            stroke = _hex(_attr(node, "stroke", "#000")) or (80, 80, 80)
            sw = max(1, int(sx(_attr(node, "stroke-width", "2"))))
            dash = _attr(node, "stroke-dasharray")
            d = None
            if dash:
                parts = [int(float(p)) for p in re.split(r"[ ,]+", dash.strip()) if p]
                if len(parts) >= 2:
                    d = (int(parts[0] * scale), int(parts[1] * scale))
            pts = [tuple(map(float, p.split(",")))
                   for p in _attr(node, "points", "").split()]
            for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                cv.line(x0 * scale, y0 * scale, x1 * scale, y1 * scale, stroke, sw, d)
        elif tag == "circle":
            fill = _hex(_attr(node, "fill", "#000")) or (80, 80, 80)
            cv.circle(sx(_attr(node, "cx", "0")), sx(_attr(node, "cy", "0")),
                      sx(_attr(node, "r", "3")), fill)
        elif tag == "text":
            fill = _hex(_attr(node, "fill", "#000")) or (30, 30, 30)
            size = sx(_attr(node, "font-size", "12"))
            anchor = _attr(node, "text-anchor", "start")
            bold = _attr(node, "font-weight", "") == "bold"
            txt = "".join(n.data for n in node.childNodes if n.nodeType == n.TEXT_NODE)
            txt = txt.strip()
            if txt:
                cv.text(sx(_attr(node, "x", "0")), sx(_attr(node, "y", "0")),
                        txt, fill, size, anchor, bold)

    with open(png_path, "wb") as f:
        f.write(cv.png_bytes())
    return cv.w, cv.h


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    art = os.path.join(here, "artifacts")
    for name in ("figure1_convergence", "figure2_frontier", "graphical_abstract"):
        w, h = rasterize(os.path.join(art, name + ".svg"),
                         os.path.join(art, name + ".png"), scale=2.0)
        print(f"{name}.png  {w}x{h}")
