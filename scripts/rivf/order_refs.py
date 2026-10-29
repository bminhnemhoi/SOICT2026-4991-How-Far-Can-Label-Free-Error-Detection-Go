"""Write paper_rivf2026/refs.tex with the verified bibitems in order of first citation."""
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[2] / "paper_rivf2026"
BIB = Path(sys.argv[1])

INPUT_RE = re.compile(r"\\input\{([^}]+)\}")
CITE_RE = re.compile(r"\\cite\{([^}]+)\}")
ITEM_RE = re.compile(r"\\bibitem\{([^}]+)\}")


def expand(text: str) -> str:
    out, pos = [], 0
    for m in INPUT_RE.finditer(text):
        out.append(text[pos:m.start()])
        name = m.group(1)
        f = PAPER / (name if name.endswith(".tex") else name + ".tex")
        if f.exists() and "refs" not in name:
            out.append(expand(f.read_text(encoding="utf-8")))
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)


full = expand((PAPER / "main.tex").read_text(encoding="utf-8"))
order: list[str] = []
for m in CITE_RE.finditer(full):
    for k in m.group(1).split(","):
        k = k.strip()
        if k not in order:
            order.append(k)
items = {}
for blk in BIB.read_text(encoding="utf-8").split("\n\n"):
    blk = blk.strip()
    if blk:
        items[ITEM_RE.match(blk).group(1)] = blk
print("order:", order)
print("missing:", [k for k in order if k not in items])
print("unused:", [k for k in items if k not in order])
(PAPER / "refs.tex").write_text("\n\n".join(items[k] for k in order if k in items) + "\n", encoding="utf-8")
