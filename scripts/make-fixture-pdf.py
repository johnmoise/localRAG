"""Write tests/fixtures/sample.pdf: a 2-page text PDF whose printed page labels start at 345.

Hand-built (no dependencies) so the fixture's content and page labels are known exactly.
"""
import sys

PAGES = [
    ["Sample Paper on Attention", "Self-attention takes quadratic time in sequence length.",
     "This sentence is used to check text extraction."],
    ["Second page of the sample paper.", "Linear attention reduces the cost to linear time."],
]


def content_stream(lines):
    ops = ["BT", "/F1 14 Tf", "72 720 Td", "18 TL"]
    for line in lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        ops.append(f"({escaped}) Tj T*")
    ops.append("ET")
    return "\n".join(ops).encode()


def build():
    objs = []  # index i -> object number i+1

    def add(body: bytes) -> int:
        objs.append(body)
        return len(objs)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    pages_num = 1 + len(PAGES) * 2 + 1  # font + (content, page) per page, then Pages
    page_nums = []
    for lines in PAGES:
        stream = content_stream(lines)
        c = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        p = add(b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] "
                b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>" % (pages_num, font, c))
        page_nums.append(p)
    kids = " ".join(f"{n} 0 R" for n in page_nums).encode()
    assert add(b"<< /Type /Pages /Kids [" + kids + b"] /Count %d >>" % len(page_nums)) == pages_num
    catalog = add(b"<< /Type /Catalog /Pages %d 0 R /PageLabels << /Nums [0 << /S /D /St 345 >>] >> >>" % pages_num)
    info = add(b"<< /Title (Sample Paper on Attention) /Author (A. Tester) >>")

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root %d 0 R /Info %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objs) + 1, catalog, info, xref)
    return bytes(out)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/sample.pdf"
    with open(path, "wb") as f:
        f.write(build())
