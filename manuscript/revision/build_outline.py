"""Revised manuscript + SI outline -> JSON for build_docx.js (reuses the Markdown converter of build_content.py)."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md_blocks(path):
    out, para = [], []

    def flush():
        if para:
            out.append({"t": "p", "text": " ".join(para)})
            para.clear()
    for line in open(path, encoding="utf8"):
        l = line.rstrip("\n")
        conv = lambda t: re.sub(r"`([^`]+)`", r"\1", t)  # noqa: E731
        if not l.strip():
            flush(); continue
        if l.startswith("# "):
            flush(); out.append({"t": "h1", "text": conv(l[2:])})
        elif l.startswith("## "):
            flush(); out.append({"t": "h2", "text": conv(l[3:])})
        elif re.match(r"^\s{2,}[-*] ", l):
            flush(); out.append({"t": "bullets", "items": [conv(l.strip()[2:])], "level": 1})
        elif re.match(r"^[-*] ", l):
            flush(); out.append({"t": "bullets", "items": [conv(l[2:])], "level": 0})
        elif re.match(r"^\d+\. ", l):
            flush(); out.append({"t": "bullets", "items": [conv(l)], "level": 0})
        else:
            para.append(conv(l.strip()))
    flush()
    return out


title = json.load(open(HERE / "revised_manuscript.json"))["title"]
B = [{"t": "title", "text": "Revised manuscript and supplementary-material outline"}, {"t": "subtitle", "text": title, "italic": True}]
B += md_blocks(HERE / "revised_outline.md")
json.dump({"title": "Revised outline", "blocks": B}, open(HERE / "revised_outline.json", "w"), ensure_ascii=False, indent=0)
print(len(B), "blocks")
