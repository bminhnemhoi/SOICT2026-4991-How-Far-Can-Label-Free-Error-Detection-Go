"""Build the SOICT 2026 camera-ready upload for paper 4991.

Produces in SOICT2026/camera_ready_package/:
  4991.pdf  final manuscript (File A)
  4991.zip  Source/ (everything needed to rebuild the PDF), Corresponding_Author.txt,
            and the signed, scanned Licence to Publish (File B)

Put the scanned licence at SOICT2026/camera_ready_package/4991_Licence_to_Publish_signed.pdf
before running; the script refuses to build the ZIP without it unless --draft is given.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "SOICT2026" / "camera_ready"
PKG = ROOT / "SOICT2026" / "camera_ready_package"
SIGNED = PKG / "4991_Licence_to_Publish_signed.pdf"
SOURCE_FILES = ["main.tex", "refs.bib", "main.bbl", "llncs.cls", "splncs04.bst"]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=SRC, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true", help="build without the signed licence")
    args = ap.parse_args()

    # class and bibliography style travel with the source
    for name in ("llncs.cls", "splncs04.bst"):
        found = subprocess.run(["kpsewhich", name], capture_output=True, text=True).stdout.strip()
        shutil.copy(found, SRC / name)
    for cmd in (["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"], ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
        run(cmd)
    log = (SRC / "main.log").read_text(encoding="latin-1")
    if "! " in log or "undefined" in log.lower().replace("undefined references", "undefined"):
        bad = [l for l in log.splitlines() if l.startswith("!") or "undefined" in l.lower()]
        if bad:
            print("LaTeX problems:\n" + "\n".join(bad[:10]))
            return 1

    PKG.mkdir(parents=True, exist_ok=True)
    shutil.copy(SRC / "main.pdf", PKG / "4991.pdf")
    if not SIGNED.exists() and not args.draft:
        print(f"Missing {SIGNED.name}: scan the ink-signed licence to this path, or use --draft.")
        return 2
    zpath = PKG / ("4991_DRAFT_without_licence.zip" if args.draft and not SIGNED.exists() else "4991.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for name in SOURCE_FILES:
            z.write(SRC / name, f"Source/{name}")
        for sub in ("figures", "tables"):
            for f in sorted((SRC / sub).iterdir()):
                if f.suffix in (".tex", ".pdf"):
                    z.write(f, f"Source/{sub}/{f.name}")
        z.write(PKG / "Corresponding_Author.txt", "Corresponding_Author.txt")
        if SIGNED.exists():
            z.write(SIGNED, SIGNED.name)
    print("wrote", PKG / "4991.pdf", "and", zpath)
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            print("  ", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
