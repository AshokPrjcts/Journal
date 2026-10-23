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
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "paper.md")


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


def build_docx(blocks, out_path):
    body = []
    for kind, payload in blocks:
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
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '</Types>')
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        '</Relationships>')

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document)


# =========================================================================== #
# PDF WRITER  (minimal, multi-page, Helvetica core fonts)
# =========================================================================== #
class PDF:
    def __init__(self, width=595, height=842, margin=56):
        self.w, self.h, self.m = width, height, margin
        self.pages = []
        self.cur = []
        self.y = self.h - self.m

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
        objs = []
        # 1 catalog, 2 pages, then per page: content + page obj, plus 2 fonts
        n_pages = len(self.pages)
        font_reg = 3
        font_bold = 4
        first_content = 5
        page_obj_start = first_content + n_pages

        kids = " ".join(f"{page_obj_start + i} 0 R" for i in range(n_pages))
        objs.append(("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"))
        objs.append((f"2 0 obj\n<< /Type /Pages /Count {n_pages} /Kids [{kids}] >>\nendobj\n"))
        objs.append(("3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"))
        objs.append(("4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n"))

        for i, content in enumerate(self.pages):
            stream = "\n".join(content)
            objs.append((f"{first_content + i} 0 obj\n<< /Length {len(stream)} >>\nstream\n{stream}\nendstream\nendobj\n"))
        for i in range(n_pages):
            objs.append((f"{page_obj_start + i} 0 obj\n<< /Type /Page /Parent 2 0 R "
                         f"/MediaBox [0 0 {self.w} {self.h}] "
                         f"/Resources << /Font << /F1 {font_reg} 0 R /F2 {font_bold} 0 R >> >> "
                         f"/Contents {first_content + i} 0 R >>\nendobj\n"))

        pdf = "%PDF-1.4\n"
        offsets = []
        for obj in objs:
            offsets.append(len(pdf.encode("latin-1", "replace")))
            pdf += obj
        xref_pos = len(pdf.encode("latin-1", "replace"))
        total = len(objs) + 1
        pdf += f"xref\n0 {total}\n0000000000 65535 f \n"
        for off in offsets:
            pdf += f"{off:010d} 00000 n \n"
        pdf += f"trailer\n<< /Size {total} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF"
        with open(path, "wb") as f:
            f.write(pdf.encode("latin-1", "replace"))


def build_pdf(blocks, out_path):
    pdf = PDF()
    for kind, payload in blocks:
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
    blocks = read_blocks(SRC)
    docx_path = os.path.join(HERE, "paper.docx")
    pdf_path = os.path.join(HERE, "paper.pdf")
    build_docx(blocks, docx_path)
    build_pdf(blocks, pdf_path)
    print("Parsed blocks :", len(blocks))
    print("Wrote         :", docx_path)
    print("Wrote         :", pdf_path)


if __name__ == "__main__":
    main()
