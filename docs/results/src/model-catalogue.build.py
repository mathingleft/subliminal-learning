import html
import re
from pathlib import Path

import markdown

D = str(Path(__file__).resolve().parent) + "/"
md = open(D + "model-catalogue.md").read()
# split into the title block and "## " sections
parts = re.split(r"\n(?=## )", md)
head, sections = parts[0], parts[1:]
head = re.sub(r"^# .*\n", "", head)  # the page has its own h1
fam = [s for s in sections if re.match(r"## \d+\.", s)]
short = [s for s in sections if s.startswith("## Ranked shortlist")]
src = [s for s in sections if s.startswith("## Sources")]
assert len(short) == 1 and len(src) == 1 and len(fam) == 16, (len(short), len(src), len(fam))
conv = lambda t: markdown.markdown(t.replace("\n---\n", "\n"), extensions=["tables"])
def slug(s): return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]
def wrap_tables(h): return h.replace("<table>", '<div class="tw"><table>').replace("</table>", "</table></div>")
toc, fam_html = [], []
for s in fam:
    title = s.splitlines()[0][3:].strip()
    sid = slug(re.sub(r"^\d+\.\s*", "", title))
    toc.append(f'<a href="#{sid}">{html.escape(re.sub(r"^\d+\.\s*", "", title))}</a>')
    body = "\n".join(s.splitlines()[1:])
    fam_html.append(f'<section class="fam" id="{sid}"><h2>{html.escape(title)}</h2>{wrap_tables(conv(body))}</section>')
short_body = "\n".join(short[0].splitlines()[1:])
src_body = "\n".join(src[0].splitlines()[1:])
page = open(D + "model-catalogue.shell.html").read()
page = page.replace("__HEAD__", wrap_tables(conv(head))).replace("__SHORT__", conv(short_body)).replace("__TOC__", "\n".join(toc))
page = page.replace("__FAMILIES__", "\n".join(fam_html)).replace("__SOURCES__", conv(src_body))
open(D + "../model-catalogue.html", "w").write(page)
print("ok", len(page))
