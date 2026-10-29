"""Export the abstract of paper_rivf2026/main.tex as plain text for EDAS."""
import re
from pathlib import Path

paper = Path(__file__).resolve().parents[2] / "paper_rivf2026"
tex = (paper / "main.tex").read_text(encoding="utf-8")
abstract = tex.split("\\begin{abstract}")[1].split("\\end{abstract}")[0].strip()
title = re.search(r"\\title\{(.+?)\}\n", tex).group(1)
text = abstract.replace("\\sys\\ ", "AutoCSV-LED ").replace("\\sys", "AutoCSV-LED").replace("--", "\u2013").replace("\\%", "%")
text = text.replace("``", '"').replace("''", '"')
text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
text = re.sub(r"\\[a-zA-Z]+", "", text).replace("~", " ").replace("  ", " ")
out = f"TITLE:\n{title}\n\nABSTRACT ({len(text.split())} words):\n{text}\n\nKEYWORDS:\ndata quality; error detection; Dempster\u2013Shafer theory; large language models; human-in-the-loop; weak supervision\n"
(paper / "abstract_edas.txt").write_text(out, encoding="utf-8")
print(out)
