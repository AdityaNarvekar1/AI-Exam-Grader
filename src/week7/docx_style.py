"""Small python-docx helper that reproduces the look of the team's Week 1-6 reports
(navy title, grey italic subtitle, shaded key/value table, navy headings, compact tables)."""
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Emu, Pt, RGBColor, Inches

NAVY = RGBColor(0x1F, 0x38, 0x64)
GREY = RGBColor(0x59, 0x59, 0x59)
SHADE = "EEF2F7"


def _shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def _borders(table):
    tblPr = table._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "A6A6A6")
        b.append(e)
    tblPr.append(b)


class Report:
    def __init__(self, title, subtitle):
        self.d = Document()
        s = self.d.sections[0]
        s.page_width, s.page_height = Emu(7560310), Emu(10692130)
        for side in ("left_margin", "right_margin"):
            setattr(s, side, Inches(0.7))
        s.top_margin = s.bottom_margin = Inches(0.6)
        st = self.d.styles["Normal"]
        st.font.name = "Calibri"; st.font.size = Pt(10)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        st.paragraph_format.space_after = Pt(4)
        p = self.d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(title); r.bold = True; r.font.size = Pt(16); r.font.color.rgb = NAVY
        p.paragraph_format.space_after = Pt(1)
        p = self.d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(subtitle); r.italic = True; r.font.size = Pt(9.5); r.font.color.rgb = GREY
        p.paragraph_format.space_after = Pt(10)

    def kv(self, pairs):
        t = self.d.add_table(rows=0, cols=2); _borders(t)
        for k, v in pairs:
            c = t.add_row().cells
            c[0].width, c[1].width = Inches(1.6), Inches(5.2)
            _shade(c[0], SHADE)
            r = c[0].paragraphs[0].add_run(k); r.bold = True; r.font.size = Pt(9); r.font.color.rgb = NAVY
            r = c[1].paragraphs[0].add_run(v); r.font.size = Pt(9)
        self.d.add_paragraph()

    def h(self, text):
        p = self.d.add_paragraph()
        p.paragraph_format.space_before = Pt(12); p.paragraph_format.space_after = Pt(4)
        r = p.add_run(text); r.bold = True; r.font.size = Pt(12); r.font.color.rgb = NAVY
        p.style = self.d.styles["Heading 1"]
        for run in p.runs:
            run.font.color.rgb = NAVY; run.font.size = Pt(12); run.font.name = "Calibri"

    def p(self, text, italic=False, size=10):
        p = self.d.add_paragraph()
        self._rich(p, text, italic, size)
        return p

    def _rich(self, p, text, italic=False, size=10):
        # **bold** segments
        parts = text.split("**")
        for i, seg in enumerate(parts):
            if not seg:
                continue
            r = p.add_run(seg); r.bold = i % 2 == 1; r.italic = italic; r.font.size = Pt(size)

    def bullets(self, items):
        for it in items:
            p = self.d.add_paragraph(style="List Bullet")
            self._rich(p, it)
            p.paragraph_format.space_after = Pt(2)

    def table(self, header, rows, widths=None, size=8.5, note=None):
        t = self.d.add_table(rows=1, cols=len(header)); _borders(t)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, h in enumerate(header):
            c = t.rows[0].cells[i]; _shade(c, SHADE)
            r = c.paragraphs[0].add_run(str(h)); r.bold = True; r.font.size = Pt(size); r.font.color.rgb = NAVY
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                txt = str(v)
                bold = txt.startswith("**") and txt.endswith("**")
                r = cells[i].paragraphs[0].add_run(txt.strip("*")); r.font.size = Pt(size); r.bold = bold
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Inches(w)
        if note:
            p = self.d.add_paragraph(); r = p.add_run(note); r.italic = True; r.font.size = Pt(8); r.font.color.rgb = GREY
        else:
            self.d.add_paragraph().paragraph_format.space_after = Pt(2)
        return t

    def image(self, path, width_in=6.2, caption=None):
        self.d.add_picture(str(path), width=Inches(width_in))
        self.d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        if caption:
            p = self.d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(caption); r.italic = True; r.font.size = Pt(8.5); r.font.color.rgb = GREY

    def save(self, path):
        self.d.save(str(path))
