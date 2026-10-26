"""Assemble main.tex from main_template.tex and the generated tables.

    python build_paper.py
"""
import re
from pathlib import Path

D = Path(__file__).resolve().parent
t = (D / "main_template.tex").read_text(encoding="utf-8")


def table(m):
    return (D / (m.group(1) + ".tex")).read_text(encoding="utf-8").strip()


t = re.sub(r"%%TAB:(\w+)%%", table, t)
left = re.findall(r"%%TAB[^%]*%%", t)
assert not left, f"unfilled markers: {left}"
(D / "main.tex").write_text(t, encoding="utf-8")
print("wrote main.tex", len(t), "chars")
