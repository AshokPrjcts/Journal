"""
build_docs.py — Convert the Markdown manuscript into Word (.docx) and PDF.

ZERO-DEPENDENCY (Python standard library only). No pandoc / libreoffice /
python-docx / reportlab required. A .docx file is an OOXML ZIP archive, which
we author directly; the PDF is produced by a minimal built-in writer.

Usage:
    python3 build_docs.py            # builds paper.docx and paper.pdf

Supported Markdown subset (sufficient for paper.md):
  # / ## / ### headings, paragraphs, blank-line separation,
  **bold**, *italic*, `code`, bullet lists (- ...), numbered blockquote
  pseudocode (> ...), and GitHub-style pipe tables.
Inline HTML sub/sup tags (<sub>, <sup>) are flattened to plain text.
"""

from __future__ import annotations

import html
import os
import re
import struct
import zipfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "paper.md")
ART = os.path.join(HERE, "artifacts")

# Figures are embedded as PNGs; ensure they exist (rasterized from SVG).
FIG1 = os.path.join(ART, "figure1_convergence.png")
FIG2 = os.path.join(ART, "figure2_frontier.png")


def _ensure_pngs():
    try:
        import svg2png
    except Exception:
        return
    for name in ("figure1_convergence", "figure2_frontier", "graphical_abstract"):
        svg = os.path.join(ART, name + ".svg")
        png = os.path.join(ART, name + ".png")
        if os.path.exists(svg) and not os.path.exists(png):
            svg2png.rasterize(svg, png, scale=2.0)


def png_size(path):
    with open(path, "rb") as f:
        d = f.read(26)
    return struct.unpack(">II", d[16:24])


# --------------------------------------------------------------------------- #
# Markdown parsing into a simple block list
# --------------------------------------------------------------------------- #
def read_blocks(path):
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")

    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped == "" or stripped == "---":
            i += 1
            continue

        # headings
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            blocks.append(("h" + str(len(m.group(1))), m.group(2).strip()))
            i += 1
            continue

        # table (pipe table): current line starts with | and next is a separator
        if stripped.startswith("|") and i + 1 < n and re.match(r"^\s*\|[-:\s|]+\|\s*$", lines[i + 1]):
            tbl = []
            while i < n and lines[i].strip().startswith("|"):
                tbl.append(lines[i].strip())
                i += 1
            rows = []
            for r_idx, row in enumerate(tbl):
                if r_idx == 1:
                    continue  # separator
                cells = [c.strip() for c in row.strip().strip("|").split("|")]
                rows.append(cells)
            blocks.append(("table", rows))
            continue

        # blockquote (pseudocode / algorithm) — gather consecutive > lines
        if stripped.startswith(">"):
            quote = []
            while i < n and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip()[1:].strip())
                i += 1
            blocks.append(("quote", quote))
            continue

        # bullet list
        if stripped.startswith("- "):
            items = []
            while i < n and lines[i].strip().startswith("- "):
                items.append(lines[i].strip()[2:].strip())
                i += 1
            blocks.append(("ul", items))
            continue

        # numbered list
        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while i < n and re.match(r"^\d+\.\s+", lines[i].strip()):
                items.append(re.sub(r"^\d+\.\s+", "", lines[i].strip()))
                i += 1
            blocks.append(("ol", items))
            continue

        # paragraph — gather until blank line
        para = [stripped]
        i += 1
        while i < n and lines[i].strip() != "" and not lines[i].strip().startswith(("#", "|", ">", "- ")) \
                and lines[i].strip() != "---" and not re.match(r"^\d+\.\s+", lines[i].strip()):
            para.append(lines[i].strip())
            i += 1
        blocks.append(("p", " ".join(para)))
    return blocks


def insert_figures(blocks):
    """Insert ('image', png_path) blocks immediately before each figure caption."""
    out = []
    for kind, payload in blocks:
        if kind == "p" and payload.startswith("**Fig. 1.**") and os.path.exists(FIG1):
            out.append(("image", FIG1))
        if kind == "p" and payload.startswith("**Fig. 2.**") and os.path.exists(FIG2):
            out.append(("image", FIG2))
        out.append((kind, payload))
    return out


def flatten_inline(text):
    """Return a list of (text, bold, italic, code) runs from inline markdown."""
    # drop sub/sup HTML tags but keep content
    text = re.sub(r"</?su[bp]>", "", text)
    text = re.sub(r"</?strong>", "**", text)
    text = html.unescape(text)

    runs = []
    # tokenizer for **bold**, *italic*, `code`
    pattern = re.compile(r"(\*\*.+?\*\*|\*.+?\*|`.+?`)")
    pos = 0
    for mo in pattern.finditer(text):
        if mo.start() > pos:
            runs.append((text[pos:mo.start()], False, False, False))
        tok = mo.group(0)
        if tok.startswith("**"):
            runs.append((tok[2:-2], True, False, False))
        elif tok.startswith("`"):
            runs.append((tok[1:-1], False, False, True))
        else:
            runs.append((tok[1:-1], False, True, False))
        pos = mo.end()
    if pos < len(text):
        runs.append((text[pos:], False, False, False))
    return runs or [("", False, False, False)]


# =========================================================================== #
# DOCX WRITER  (OOXML, authored directly, zipped)
# =========================================================================== #
def xml_escape(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def docx_runs(runs):
    out = []
    for txt, b, it, code in runs:
        if txt == "":
            continue
        props = []
        if b:
            props.append("<w:b/>")
        if it:
            props.append("<w:i/>")
        if code:
            props.append('<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>')
        rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
        out.append(f'<w:r>{rpr}<w:t xml:space="preserve">{xml_escape(txt)}</w:t></w:r>')
    return "".join(out)


def docx_paragraph(runs, style=None, size=None, bold=False, align=None, space_after=120):
    ppr = ["<w:spacing w:line=\"360\" w:lineRule=\"auto\" w:after=\"%d\"/>" % space_after]
    if style:
        ppr.insert(0, f'<w:pStyle w:val="{style}"/>')
    if align:
        ppr.append(f'<w:jc w:val="{align}"/>')
    rpr_extra = ""
    if size:
        rpr_extra += f'<w:sz w:val="{size*2}"/><w:szCs w:val="{size*2}"/>'
    if bold:
        rpr_extra += "<w:b/>"
    # inject run-level default size/bold by wrapping: simplest is per-run; here we
    # add a paragraph mark rPr and rely on run props already set.
    body = docx_runs(runs)
    if size or bold:
        # re-wrap each run to include size/bold
        new = []
        for txt, b, it, code in runs:
            if txt == "":
                continue
            props = []
            if bold or b:
                props.append("<w:b/>")
            if it:
                props.append("<w:i/>")
            if code:
                props.append('<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>')
            if size:
                props.append(f'<w:sz w:val="{size*2}"/><w:szCs w:val="{size*2}"/>')
            rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
            new.append(f'<w:r>{rpr}<w:t xml:space="preserve">{xml_escape(txt)}</w:t></w:r>')
        body = "".join(new)
    return f"<w:p><w:pPr>{''.join(ppr)}</w:pPr>{body}</w:p>"


def docx_table(rows):
    header = rows[0]
    ncol = len(header)
    grid = "".join('<w:gridCol w:w="%d"/>' % int(9360 / ncol) for _ in range(ncol))
    out = [f'<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/>'
           f'<w:tblW w:w="9360" w:type="dxa"/>'
           f'<w:tblBorders>'
           f'<w:top w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
           f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
           f'<w:insideH w:val="single" w:sz="2" w:space="0" w:color="CCCCCC"/>'
           f'</w:tblBorders></w:tblPr><w:tblGrid>{grid}</w:tblGrid>']
    for r_idx, row in enumerate(rows):
        cells = []
        for c in row:
            runs = flatten_inline(c)
            run_xml = []
            for txt, b, it, code in runs:
                props = ["<w:sz w:val=\"18\"/><w:szCs w:val=\"18\"/>"]
                if b or r_idx == 0:
                    props.append("<w:b/>")
                if it:
                    props.append("<w:i/>")
                rpr = f"<w:rPr>{''.join(props)}</w:rPr>"
                run_xml.append(f'<w:r>{rpr}<w:t xml:space="preserve">{xml_escape(txt)}</w:t></w:r>')
            shade = '<w:shd w:val="clear" w:fill="F2F2F2"/>' if r_idx == 0 else ""
            cells.append(
                f'<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/>{shade}</w:tcPr>'
                % int(9360 / ncol)
                + f'<w:p><w:pPr><w:spacing w:after="20"/></w:pPr>{"".join(run_xml)}</w:p></w:tc>')
        out.append(f"<w:tr>{''.join(cells)}</w:tr>")
    out.append("</w:tbl><w:p><w:pPr><w:spacing w:after=\"120\"/></w:pPr></w:p>")
    return "".join(out)


def docx_image_paragraph(rel_id, img_w_px, img_h_px, max_w_px=600):
    """Return a centered paragraph containing an inline image drawing."""
    # scale to max width, convert px->EMU (914400 EMU per inch, 96 px per inch)
    scale = min(1.0, max_w_px / img_w_px)
    cx = int(img_w_px * scale * 9525)
    cy = int(img_h_px * scale * 9525)
    drawing = (
        '<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">'
        f'<wp:extent cx="{cx}" cy="{cy}"/>'
        '<wp:docPr id="1" name="Figure"/>'
        '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:nvPicPr><pic:cNvPr id="0" name="Figure"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rel_id}" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>'
        '<a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        '<pic:spPr><a:xfrm><a:off x="0" y="0"/>'
        f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
        '</a:graphicData></a:graphic></wp:inline></w:drawing>')
    return ('<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:after="120"/></w:pPr>'
            f'<w:r>{drawing}</w:r></w:p>')


def build_docx(blocks, out_path):
    body = []
    images = []  # (rel_id, media_name, bytes)
    img_counter = [0]

    def add_image(path):
        img_counter[0] += 1
        rid = f"rIdImg{img_counter[0]}"
        name = f"image{img_counter[0]}.png"
        with open(path, "rb") as f:
            data = f.read()
        images.append((rid, name, data))
        return rid, png_size(path)

    for kind, payload in blocks:
        if kind == "image":
            rid, (iw, ih) = add_image(payload)
            body.append(docx_image_paragraph(rid, iw, ih))
            continue
        if kind == "h1":
            body.append(docx_paragraph(flatten_inline(payload), size=18, bold=True,
                                       align="center", space_after=200))
        elif kind == "h2":
            body.append(docx_paragraph(flatten_inline(payload), size=14, bold=True,
                                       space_after=140))
        elif kind == "h3":
            body.append(docx_paragraph(flatten_inline(payload), size=12, bold=True,
                                       space_after=120))
        elif kind == "p":
            body.append(docx_paragraph(flatten_inline(payload), size=11))
        elif kind == "ul":
            for it in payload:
                runs = flatten_inline("\u2022  " + it)
                body.append(docx_paragraph(runs, size=11, space_after=40))
        elif kind == "ol":
            for idx, it in enumerate(payload, 1):
                runs = flatten_inline(f"{idx}.  " + it)
                body.append(docx_paragraph(runs, size=11, space_after=40))
        elif kind == "quote":
            for q in payload:
                runs = flatten_inline(q)
                # monospace-ish: mark as code run
                runs = [(t, b, i2, True) for (t, b, i2, _c) in runs]
                body.append(docx_paragraph(runs, size=10, space_after=20))
        elif kind == "table":
            body.append(docx_table(payload))

    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f'<w:body>{"".join(body)}'
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>'
        '</w:body></w:document>')

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="png" ContentType="image/png"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '</Types>')
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        '</Relationships>')

    img_rels = "".join(
        f'<Relationship Id="{rid}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
        f'Target="media/{name}"/>' for rid, name, _ in images)
    document_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{img_rels}</Relationships>')

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document)
        z.writestr("word/_rels/document.xml.rels", document_rels)
        for _rid, name, data in images:
            z.writestr(f"word/media/{name}", data)


# =========================================================================== #
# PDF WRITER  (minimal, multi-page, Helvetica core fonts)
# =========================================================================== #
def decode_png_rgb(path):
    """Decode a non-interlaced 8-bit RGB PNG to (w, h, raw_rgb_bytes)."""
    with open(path, "rb") as f:
        d = f.read()
    assert d[:8] == b"\x89PNG\r\n\x1a\n"
    pos = 8
    width = height = 0
    idat = b""
    while pos < len(d):
        ln = struct.unpack(">I", d[pos:pos + 4])[0]
        typ = d[pos + 4:pos + 8]
        data = d[pos + 8:pos + 8 + ln]
        if typ == b"IHDR":
            width, height = struct.unpack(">II", data[:8])
        elif typ == b"IDAT":
            idat += data
        elif typ == b"IEND":
            break
        pos += 12 + ln
    raw = zlib.decompress(idat)
    stride = width * 3
    out = bytearray()
    prev = bytearray(stride)
    p = 0
    for _ in range(height):
        ft = raw[p]; p += 1
        line = bytearray(raw[p:p + stride]); p += stride
        if ft == 1:  # Sub
            for i in range(3, stride):
                line[i] = (line[i] + line[i - 3]) & 0xff
        elif ft == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xff
        elif ft == 3:  # Average
            for i in range(stride):
                a = line[i - 3] if i >= 3 else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xff
        elif ft == 4:  # Paeth
            for i in range(stride):
                a = line[i - 3] if i >= 3 else 0
                b = prev[i]
                c = prev[i - 3] if i >= 3 else 0
                pp = a + b - c
                pa, pb, pc = abs(pp - a), abs(pp - b), abs(pp - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xff
        out += line
        prev = line
    return width, height, bytes(out)


class PDF:
    def __init__(self, width=595, height=842, margin=56):
        self.w, self.h, self.m = width, height, margin
        self.pages = []
        self.cur = []
        self.y = self.h - self.m
        self.images = []  # list of (name, w, h, zlib_rgb)

    def _wrap(self, text, font_size, bold=False, max_width=None):
        max_width = max_width or (self.w - 2 * self.m)
        # approx char width for Helvetica
        cw = font_size * (0.55 if bold else 0.5)
        max_chars = max(10, int(max_width / cw))
        words = text.split()
        lines, line = [], ""
        for wd in words:
            if len(line) + len(wd) + 1 <= max_chars:
                line = (line + " " + wd).strip()
            else:
                if line:
                    lines.append(line)
                line = wd
        if line:
            lines.append(line)
        return lines or [""]

    def _newpage(self):
        if self.cur:
            self.pages.append(self.cur)
        self.cur = []
        self.y = self.h - self.m

    def _ensure(self, needed):
        if self.y - needed < self.m:
            self._newpage()

    def text(self, s, size=11, bold=False, gap=4, align="left", indent=0):
        font = "F2" if bold else "F1"
        for line in self._wrap(s, size, bold, self.w - 2 * self.m - indent):
            self._ensure(size + gap)
            esc = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            x = self.m + indent
            if align == "center":
                cw = size * (0.55 if bold else 0.5)
                x = (self.w - len(line) * cw) / 2
            self.cur.append(f"BT /{font} {size} Tf 1 0 0 1 {x:.1f} {self.y:.1f} Tm ({esc}) Tj ET")
            self.y -= size + gap

    def spacer(self, h=6):
        self.y -= h

    def hline(self):
        self._ensure(8)
        self.cur.append(f"{self.m} {self.y:.1f} m {self.w - self.m} {self.y:.1f} l 0.7 w 0.6 0.6 0.6 RG S 0 0 0 RG")
        self.y -= 8

    def image(self, path, max_w=None):
        iw, ih, rgb = decode_png_rgb(path)
        max_w = max_w or (self.w - 2 * self.m)
        draw_w = max_w
        draw_h = draw_w * ih / iw
        # new page if it doesn't fit
        if self.y - draw_h < self.m:
            self._newpage()
        name = f"Im{len(self.images) + 1}"
        self.images.append((name, iw, ih, zlib.compress(rgb, 9)))
        x = self.m + (max_w - draw_w) / 2
        y = self.y - draw_h
        self.cur.append(f"q {draw_w:.1f} 0 0 {draw_h:.1f} {x:.1f} {y:.1f} cm /{name} Do Q")
        self.y = y - 10

    def table(self, rows, size=8):
        ncol = len(rows[0])
        avail = self.w - 2 * self.m
        colw = avail / ncol
        for r_idx, row in enumerate(rows):
            # compute wrapped cell lines
            cell_lines = []
            maxlines = 1
            for c in row:
                plain = re.sub(r"\*\*|\*|`", "", re.sub(r"</?su[bp]>", "", c))
                plain = html.unescape(plain)
                wl = self._wrap(plain, size, r_idx == 0, colw - 6)
                cell_lines.append(wl)
                maxlines = max(maxlines, len(wl))
            rowh = maxlines * (size + 2) + 4
            self._ensure(rowh)
            top = self.y
            if r_idx == 0:
                self.cur.append(f"{self.m} {top - rowh + 2:.1f} {avail:.1f} {rowh:.1f} re 0.95 0.95 0.95 rg f 0 0 0 rg")
            # cell text
            for c_idx, wl in enumerate(cell_lines):
                cx = self.m + c_idx * colw + 3
                cy = top - size
                for ln in wl:
                    esc = ln.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
                    font = "F2" if r_idx == 0 else "F1"
                    self.cur.append(f"BT /{font} {size} Tf 1 0 0 1 {cx:.1f} {cy:.1f} Tm ({esc}) Tj ET")
                    cy -= size + 2
            # bottom rule
            by = top - rowh + 2
            self.cur.append(f"{self.m} {by:.1f} m {self.w - self.m} {by:.1f} l 0.4 w 0.8 0.8 0.8 RG S 0 0 0 RG")
            self.y = by
        self.y -= 6

    def save(self, path):
        self._newpage()
        # Build as binary from the start (images are binary streams).
        out = bytearray()
        offsets = {}

        def add_obj(num, body_bytes):
            offsets[num] = len(out)
            out.extend(f"{num} 0 obj\n".encode("latin-1"))
            out.extend(body_bytes)
            out.extend(b"\nendobj\n")

        n_pages = len(self.pages)
        font_reg, font_bold = 3, 4
        first_content = 5
        page_obj_start = first_content + n_pages
        img_obj_start = page_obj_start + n_pages

        out.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

        kids = " ".join(f"{page_obj_start + i} 0 R" for i in range(n_pages))
        add_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        add_obj(2, f"<< /Type /Pages /Count {n_pages} /Kids [{kids}] >>".encode("latin-1"))
        add_obj(3, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        add_obj(4, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

        for i, content in enumerate(self.pages):
            stream = ("\n".join(content)).encode("latin-1", "replace")
            body = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
            add_obj(first_content + i, body)

        # image XObjects
        img_names = {}
        for idx, (name, iw, ih, comp) in enumerate(self.images):
            num = img_obj_start + idx
            img_names[name] = num
            hdr = (f"<< /Type /XObject /Subtype /Image /Width {iw} /Height {ih} "
                   f"/ColorSpace /DeviceRGB /BitsPerComponent 8 "
                   f"/Filter /FlateDecode /Length {len(comp)} >>\nstream\n").encode("latin-1")
            add_obj(num, hdr + comp + b"\nendstream")

        # page objects (reference all images in resources for simplicity)
        xobj_res = " ".join(f"/{name} {img_names[name]} 0 R" for name in img_names)
        for i in range(n_pages):
            res = (f"<< /Font << /F1 {font_reg} 0 R /F2 {font_bold} 0 R >> "
                   f"/XObject << {xobj_res} >> >>")
            body = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.w} {self.h}] "
                    f"/Resources {res} /Contents {first_content + i} 0 R >>").encode("latin-1")
            add_obj(page_obj_start + i, body)

        total = 1 + 4 + n_pages + len(self.images) + n_pages
        xref_pos = len(out)
        out.extend(f"xref\n0 {total + 1}\n".encode("latin-1"))
        out.extend(b"0000000000 65535 f \n")
        for num in range(1, total + 1):
            out.extend(f"{offsets.get(num, 0):010d} 00000 n \n".encode("latin-1"))
        out.extend(f"trailer\n<< /Size {total + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode("latin-1"))
        with open(path, "wb") as f:
            f.write(out)


def build_pdf(blocks, out_path):
    pdf = PDF()
    for kind, payload in blocks:
        if kind == "image":
            pdf.image(payload)
            continue
        if kind == "h1":
            plain = html.unescape(re.sub(r"</?su[bp]>|\*\*|\*", "", payload))
            pdf.spacer(4)
            pdf.text(plain, size=15, bold=True, gap=5, align="center")
            pdf.spacer(6)
        elif kind == "h2":
            plain = html.unescape(re.sub(r"</?su[bp]>|\*\*|\*", "", payload))
            pdf.spacer(6)
            pdf.text(plain, size=12.5, bold=True, gap=4)
            pdf.spacer(2)
        elif kind == "h3":
            plain = html.unescape(re.sub(r"</?su[bp]>|\*\*|\*", "", payload))
            pdf.spacer(3)
            pdf.text(plain, size=11, bold=True, gap=3)
        elif kind == "p":
            plain = html.unescape(re.sub(r"</?su[bp]>", "", payload))
            plain = re.sub(r"\*\*(.+?)\*\*", r"\1", plain)
            plain = re.sub(r"\*(.+?)\*", r"\1", plain)
            plain = re.sub(r"`(.+?)`", r"\1", plain)
            pdf.text(plain, size=10, gap=3.5)
            pdf.spacer(3)
        elif kind == "ul":
            for it in payload:
                plain = html.unescape(re.sub(r"</?su[bp]>|\*\*|\*|`", "", it))
                pdf.text("\u2022 " + plain, size=10, gap=3, indent=10)
            pdf.spacer(3)
        elif kind == "ol":
            for idx, it in enumerate(payload, 1):
                plain = html.unescape(re.sub(r"</?su[bp]>|\*\*|\*|`", "", it))
                pdf.text(f"{idx}. " + plain, size=10, gap=3, indent=10)
            pdf.spacer(3)
        elif kind == "quote":
            for q in payload:
                plain = html.unescape(re.sub(r"</?su[bp]>|\*\*|\*|`", "", q))
                pdf.text(plain, size=8.5, gap=2.5, indent=14)
            pdf.spacer(3)
        elif kind == "table":
            pdf.table(payload)
    pdf.save(out_path)


def main():
    _ensure_pngs()
    blocks = insert_figures(read_blocks(SRC))
    docx_path = os.path.join(HERE, "paper.docx")
    pdf_path = os.path.join(HERE, "paper.pdf")
    build_docx(blocks, docx_path)
    build_pdf(blocks, pdf_path)
    n_img = sum(1 for k, _ in blocks if k == "image")
    print("Parsed blocks :", len(blocks))
    print("Figures embedded:", n_img)
    print("Wrote         :", docx_path)
    print("Wrote         :", pdf_path)


if __name__ == "__main__":
    main()
