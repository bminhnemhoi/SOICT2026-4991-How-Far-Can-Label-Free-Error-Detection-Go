"""Assemble the SOICT 2026 submission folder for paper 4991 from the final build.

SOICT2026/NOP_BAI_SOICT2026_4991/
  1_UPLOAD_EASYCHAIR/   4991.pdf (File A) and 4991.zip (File B), exactly what is uploaded
  2_THONG_TIN_EASYCHAIR/ title, author, abstract and keywords as plain text
  3_HO_SO_LUU/          signed licence, corresponding-author file, response to the reviews
  HUONG_DAN_NOP_BAI.md  step-by-step instructions (Vietnamese)
Run scripts/soict/build_package.py first.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "SOICT2026" / "camera_ready"
PKG = ROOT / "SOICT2026" / "camera_ready_package"
OUT = ROOT / "SOICT2026" / "NOP_BAI_SOICT2026_4991_FINAL"


def latex_to_text(s: str) -> str:
    s = s.replace(r"\sys\ ", "AutoCSV-LED ").replace(r"\sys", "AutoCSV-LED")
    s = s.replace(r"vs.\ ", "vs. ").replace("---", "\u2014").replace("--", "\u2013")
    s = s.replace("``", "\u201c").replace("''", "\u201d").replace(r"\%", "%")
    s = re.sub(r"\\texttt\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> None:
    final = PKG / "FINAL_UPLOAD"
    pdf, zp = final / "4991.pdf", final / "4991.zip"
    assert pdf.exists() and zp.exists(), "run build_package.py first"
    up, meta, keep = OUT / "1_UPLOAD_EASYCHAIR", OUT / "2_THONG_TIN_EASYCHAIR", OUT / "3_HO_SO_LUU"
    for d in (up, meta, keep):
        d.mkdir(parents=True, exist_ok=True)

    def put(src: Path, dst: Path) -> None:  # overwrite only when changed (a file open in a viewer stays untouched)
        if not dst.exists() or dst.read_bytes() != src.read_bytes():
            shutil.copy(src, dst)

    put(pdf, up / "4991.pdf")
    put(zp, up / "4991.zip")

    tex = (SRC / "main.tex").read_text(encoding="utf-8")
    title = re.search(r"\\title\{(.*?)\}\n", tex).group(1)
    abstract = latex_to_text(tex[tex.find(r"\begin{abstract}") + 16: tex.find(r"\keywords")])
    keywords = re.search(r"\\keywords\{(.*?)\}", tex).group(1).replace(r" \and ", "; ").replace("--", "\u2013")
    (meta / "EasyChair_metadata.txt").write_text(
        f"Paper ID: 4991\nTitle: {title}\nAuthor: Binh Minh Ngo (corresponding author)\n"
        f"Email: ngobinhminh.st@tdtu.edu.vn\nAffiliation: Faculty of Information Technology, Ton Duc Thang University, "
        f"Ho Chi Minh City, Vietnam\nORCID: 0009-0002-4141-3867\nKeywords: {keywords}\n\n"
        f"Abstract ({len(abstract.split())} words):\n{abstract}\n", encoding="utf-8")

    put(PKG / "4991_Licence_to_Publish_signed.pdf", keep / "4991_Licence_to_Publish_signed.pdf")
    put(PKG / "Corresponding_Author.txt", keep / "Corresponding_Author.txt")
    put(PKG / "4991_Licence_to_Publish_prefilled_PRINT.pdf", keep / "4991_Licence_to_Publish_prefilled_PRINT.pdf")
    put(ROOT / "SOICT2026" / "RESPONSE_TO_REVIEWERS.md", keep / "RESPONSE_TO_REVIEWERS.md")

    pages = re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout).group(1)
    with zipfile.ZipFile(zp) as z:
        listing = "\n".join(f"- `{n}`" for n in z.namelist())
    guide = (ROOT / "scripts" / "soict" / "huong_dan_template.md").read_text(encoding="utf-8")
    guide = guide.format(pages=pages, pdf_sha=sha(pdf), zip_sha=sha(zp), listing=listing,
                         pdf_kb=pdf.stat().st_size // 1024, zip_kb=zp.stat().st_size // 1024)
    (OUT / "HUONG_DAN_NOP_BAI.md").write_text(guide, encoding="utf-8")
    print("submission folder:", OUT)
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print("  ", p.relative_to(OUT))


if __name__ == "__main__":
    main()
