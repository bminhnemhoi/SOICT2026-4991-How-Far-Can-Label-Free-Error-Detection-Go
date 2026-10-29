"""IEEE style: for more than six authors list only the first author followed by et al.

Operates entry by entry (entries are separated by blank lines)."""
from pathlib import Path

p = Path(__file__).resolve().parents[2] / "paper_rivf2026" / "refs.tex"
entries = [e for e in p.read_text(encoding="utf-8").split("\n\n") if e.strip()]
out = []
for e in entries:
    key_end = e.index("} ") + 2
    head, body = e[:key_end], e[key_end:]
    marker = ", \\emph{et~al.},"
    quote = body.find("``")
    pos = body.find(marker)
    if pos != -1 and (quote == -1 or pos < quote):
        first = body[:pos].split(", ")[0]
        body = first + " \\emph{et~al.}," + body[pos + len(marker):]
    out.append(head + body)
p.write_text("\n\n".join(out) + "\n", encoding="utf-8")
for e in out:
    if "et~al" in e[:120]:
        print(e[:120])
