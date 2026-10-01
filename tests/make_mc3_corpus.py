"""Mock corpus shaped like the mc3 sample, plus the four hostile cases.

Run first, then tests/test_mc3.py.
"""

import os
import sys
import tempfile
import zipfile

OUT = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.join(tempfile.gettempdir(), "mc3_corpus")
)


def w(rel, text):
    path = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def make_docx(rel, paragraphs, table=None):
    """Hand-rolled docx: it is a zip of XML, no dependency needed."""
    from xml.sax.saxutils import escape

    body = "".join(f"<w:p><w:r><w:t>{escape(p)}</w:t></w:r></w:p>" for p in paragraphs)
    if table:
        rows = "".join(
            "<w:tr>" + "".join(f"<w:tc><w:p><w:r><w:t>{escape(c)}</w:t></w:r></w:p></w:tc>" for c in r) + "</w:tr>"
            for r in table
        )
        body += f"<w:tbl>{rows}</w:tbl>"
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    ct = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    path = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", ct)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document)
    return path


def make_xlsx(rel, sheets):
    from openpyxl import Workbook

    wb = Workbook()
    for i, (name, rows) in enumerate(sheets):
        ws = wb.active if i == 0 else wb.create_sheet()
        ws.title = name
        for row in rows:
            ws.append(row)
    path = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wb.save(path)
    return path


def make_pdf(rel, text, encrypted=False, broken=False):
    import fitz

    path = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if broken:
        with open(path, "wb") as fh:
            fh.write(b"%PDF-1.4\nthis is not a real pdf at all\n")
        return path

    doc = fitz.open()
    page = doc.new_page()
    y = 110
    for line in text.splitlines():
        page.insert_text((60, y), line, fontsize=11)
        y += 18
    if encrypted:
        doc.save(path, encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="")
    else:
        doc.save(path)
    doc.close()
    return path


def make_image(rel, lines):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (700, 260), (250, 250, 248))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arialbd.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    d.text((24, 40), lines, font=font, fill=(20, 20, 20))
    path = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    return path


def build():
    os.makedirs(OUT, exist_ok=True)

    # --- the four hostile cases from the spec ---
    os.makedirs(os.path.join(OUT, "archive"), exist_ok=True)  # empty directory
    w("vendor/internal_audit.txt", "Internal audit notes. Nothing useful here.")
    w("vendor/telemetry_capture.dat", "\x00\x01\x02binary\x00junk\x00")  # unknown type
    with open(os.path.join(OUT, "vendor", "supplier_agreement_ENCRYPTED.pdf"), "wb") as fh:
        fh.write(b"%PDF-1.4\nencrypted, no password given\n")
    os.makedirs(os.path.join(OUT, "specs"), exist_ok=True)
    w("specs/garbage.pdf", "not a pdf")  # later overwritten as broken

    # --- answerable content ---
    make_pdf(
        "specs/tq40_datasheet_r2.pdf",
        "TQ-40 Datasheet Revision 2\nMaximum junction temperature: 94 C\n"
        "Absolute maximum storage temperature: 125 C\nNominal supply: 12 V",
    )
    make_docx(
        "planning/roadmap_fy27.docx",
        ["Meridian FY27 Product Roadmap", "TQ-60 reaches customer sampling in Q3 FY27."],
        table=[["Programme", "Quarter"], ["TQ-60 sampling", "Q3 FY27"]],
    )
    make_xlsx(
        "support/rma_parts.xlsx",
        [
            ("Field Service", [["Part", "Assembly"], ["ORR-FAN-2214-B", "Fan assembly, TQ-40"]]),
            ("Costs", [["Part", "Unit price"], ["ORR-FAN-2214-B", "48.20"]]),
        ],
    )
    w(
        "support/bug_database.csv",
        "ticket,summary,firmware_fixed\n"
        "ORR-1847,Thermal throttle on sustained load,4.3.2\n"
        "ORR-1902,Spurious link fault,4.4.0\n",
    )
    w(
        "logs/prod_inference_2026-09-02.log",
        "2026-09-02T11:02:14Z WARN ingest: E7731 thermal throttle engaged on node-3\n"
        "2026-09-02T11:02:19Z INFO ingest: recovering\n"
        "2026-09-02T12:41:02Z WARN ingest: E7731 thermal throttle engaged on node-1\n",
    )
    w(
        "engineering/ingest_service.py",
        "DEFAULT_BATCH_TIMEOUT = 180\n"
        "MAX_QUEUE_BYTES = 1048576\n"
        "def flush(self):\n    return DEFAULT_BATCH_TIMEOUT\n",
    )
    make_image("specs/backplane_pinout.png", "B14 = THERM_ALERT#")
    make_image("support/asset_label.jpg", "BOARD REV-C2")

    print(f"built {OUT}")
    for dirpath, _dirs, files in os.walk(OUT):
        for f in sorted(files):
            p = os.path.join(dirpath, f)
            print(f"  {os.path.relpath(p, OUT).replace(os.sep, '/'):46} {os.path.getsize(p):>7} B")


if __name__ == "__main__":
    build()
