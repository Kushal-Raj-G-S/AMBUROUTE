"""Structural checks on main.tex (no LaTeX compiler needed)."""
import re
from pathlib import Path

D = Path(__file__).resolve().parent
t = (D / "main.tex").read_text(encoding="utf-8")
body = re.sub(r"(?m)^\s*%.*$", "", t)  # drop comment lines

labels = set(re.findall(r"\\label\{([^}]+)\}", body))
refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", body))
print("undefined refs:", sorted(refs - labels))
print("unused labels:", sorted(labels - refs))

cites = set()
for m in re.findall(r"\\cite\{([^}]+)\}", body):
    cites |= {c.strip() for c in m.split(",")}
bibs = set(re.findall(r"\\bibitem\{([^}]+)\}", body))
print("cited but missing:", sorted(cites - bibs))
print("in bibliography but never cited:", sorted(bibs - cites))

figs = set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", body))
print("missing figure files:", sorted(f for f in figs if not (D / f).exists()))
print("figures to upload:", sorted(figs))

envs = re.findall(r"\\(begin|end)\{([^}]+)\}", body)
stack, bad = [], []
for kind, name in envs:
    if kind == "begin":
        stack.append(name)
    elif stack and stack[-1] == name:
        stack.pop()
    else:
        bad.append(name)
print("unbalanced environments:", bad, stack)
print("braces balance:", body.count("{") - body.count("}"), "(escaped braces counted; expect small)")
print("placeholders left:", re.findall(r"%%[^%]+%%|TODO|XX", t))
