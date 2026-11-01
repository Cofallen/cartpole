#!/usr/bin/env python3
"""把 rl-study/ 下的所有 markdown 拼成一本可离线阅读的 HTML 书。

输出：rl-study/index.html
- 左侧固定大纲（每篇文档的 H2/H3），可搜索、随滚动高亮
- 单页上下滚动，文档间用分隔线隔开
- 无外部依赖（不联网、不加载 CDN），可打印

用法：
    conda activate isaac          # 需要 markdown 包（Isaac 环境自带）
    python scripts/md_book.py
    # 然后双击 rl-study/index.html

依赖：python-markdown（`pip install markdown` 即可，Isaac 环境已有）。
"""

from __future__ import annotations

import html
import os
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    import markdown
except ImportError:  # pragma: no cover
    sys.exit("缺少依赖：pip install markdown（或 conda activate isaac 后再运行）")

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "rl-study"
OUT = SRC_DIR / "index.html"
BOOK_TITLE = "强化学习学习库"

CSS = r"""
:root{
  --bg:#ffffff; --fg:#1f2328; --muted:#59636e; --line:#d8dee4;
  --accent:#0969da; --soft:#f6f8fa; --quote:#fff8e6; --quoteline:#e3b341;
  --navw:320px;
}
@media (prefers-color-scheme: dark){
  :root{ --bg:#0d1117; --fg:#e6edf3; --muted:#8d96a0; --line:#30363d; --accent:#58a6ff;
         --soft:#161b22; --quote:#221a05; --quoteline:#9e6a03; }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:16px/1.75 -apple-system,"Segoe UI","Noto Sans CJK SC","Source Han Sans SC","PingFang SC",sans-serif;}
nav#toc{position:fixed;top:0;left:0;bottom:0;width:var(--navw);overflow-y:auto;
  background:var(--soft);border-right:1px solid var(--line);padding:16px 12px 60px;}
nav#toc .brand{font-weight:700;font-size:15px;margin:2px 6px 10px;}
nav#toc .brand small{display:block;font-weight:400;color:var(--muted);font-size:12px;margin-top:2px}
nav#toc input{width:100%;padding:7px 9px;margin:0 0 12px;border:1px solid var(--line);
  border-radius:6px;background:var(--bg);color:var(--fg);font-size:13px}
nav#toc details{margin:0 0 2px}
nav#toc summary{cursor:pointer;padding:5px 6px;border-radius:6px;font-size:14px;font-weight:600;color:var(--fg)}
nav#toc summary:hover{background:rgba(127,127,127,.12)}
nav#toc details.active>summary{color:var(--accent)}
nav#toc ul{list-style:none;margin:2px 0 8px;padding:0 0 0 14px}
nav#toc li a{display:block;padding:2px 6px;border-radius:5px;font-size:13px;
  color:var(--muted);text-decoration:none;border-left:2px solid transparent}
nav#toc li a:hover{background:rgba(127,127,127,.12);color:var(--fg)}
nav#toc li a.h3{padding-left:16px;font-size:12.5px}
nav#toc .empty{color:var(--muted);font-size:13px;padding:8px 6px}
main{margin-left:var(--navw);padding:40px 56px 120px;max-width:1040px}
main>h1.title{font-size:30px;margin:0 0 6px}
main .sub{color:var(--muted);font-size:13.5px;margin-bottom:34px}
section.doc{padding-top:8px}
section.doc+section.doc{border-top:1px solid var(--line);margin-top:56px;padding-top:40px}
h1,h2,h3,h4{line-height:1.35;scroll-margin-top:16px}
h1{font-size:26px;border-bottom:2px solid var(--line);padding-bottom:8px;margin:8px 0 18px}
h2{font-size:21px;margin:34px 0 12px;padding-top:4px}
h3{font-size:17.5px;margin:26px 0 8px;color:var(--accent)}
p{margin:10px 0}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
blockquote{margin:14px 0;padding:10px 16px;background:var(--quote);
  border-left:4px solid var(--quoteline);border-radius:0 6px 6px 0}
blockquote p{margin:6px 0}
blockquote blockquote{background:transparent;border-left-color:var(--line)}
code{font:13.5px/1.5 "JetBrains Mono",Consolas,"Noto Sans Mono CJK SC",monospace;
  background:rgba(127,127,127,.14);padding:.15em .38em;border-radius:4px}
pre{background:var(--soft);border:1px solid var(--line);border-radius:8px;
  padding:12px 14px;overflow-x:auto;margin:14px 0}
pre code{background:none;padding:0;font-size:13px;line-height:1.6}
.tablewrap{overflow-x:auto;margin:16px 0}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}
th{background:var(--soft);font-weight:600}
tr:nth-child(even) td{background:rgba(127,127,127,.045)}
hr{border:0;border-top:1px solid var(--line);margin:32px 0}
ul,ol{padding-left:24px}
li{margin:4px 0}
mark{background:#ffe08a;color:#111}
.katex-display{margin:14px 0;overflow-x:auto;overflow-y:hidden}
.katex{font-size:1.04em}
#top{position:fixed;right:22px;bottom:22px;width:42px;height:42px;border-radius:50%;
  border:1px solid var(--line);background:var(--bg);color:var(--fg);cursor:pointer;
  font-size:16px;display:none;box-shadow:0 2px 10px rgba(0,0,0,.12)}
h2:target,h3:target,h1:target{background:rgba(255,224,138,.25)}
@media print{
  nav#toc,#top{display:none} main{margin:0;padding:0;max-width:none}
  section.doc+section.doc{page-break-before:always}
}
@media (max-width:900px){
  nav#toc{position:static;width:auto;bottom:auto;border-right:0;border-bottom:1px solid var(--line)}
  main{margin-left:0;padding:20px}
}
"""

JS = r"""
const docs = [...document.querySelectorAll('section.doc')];
const details = [...document.querySelectorAll('nav#toc details')];
function openFor(id){
  details.forEach(d=>{
    const on = d.dataset.doc === id;
    d.classList.toggle('active', on);
    if(on) d.open = true;
  });
}
const io = new IntersectionObserver(entries=>{
  entries.forEach(e=>{ if(e.isIntersecting) openFor(e.target.id); });
},{rootMargin:'-10% 0px -80% 0px', threshold:0});
docs.forEach(s=>io.observe(s));

const q = document.getElementById('q');
q.addEventListener('input', ()=>{
  const v = q.value.trim().toLowerCase();
  details.forEach(d=>{
    const hit = !v || d.dataset.text.includes(v);
    d.style.display = hit ? '' : 'none';
    if(v) d.open = hit && v.length>0;
  });
});
document.getElementById('top').addEventListener('click', ()=>window.scrollTo({top:0,behavior:'smooth'}));
addEventListener('scroll', ()=>{ document.getElementById('top').style.display = scrollY>600?'block':'none'; });
"""


def slugify(text: str) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "-", text.strip().lower())
    return s.strip("-") or "sec"


def collect() -> list[dict]:
    docs = []
    for path in sorted(SRC_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        stem = path.stem
        num = stem.split("-")[0] if stem[:2].isdigit() else ""
        m = re.search(r"^#\s+(.+)$", raw, re.M)
        title = (m.group(1).strip() if m else stem)
        title = re.sub(r"[`*]", "", title)
        docs.append({"path": path, "num": num, "title": title, "title_full": stem,
                     "raw": raw, "id": f"doc-{num or slugify(stem)[:8]}"})
    return docs


MATH_RE = re.compile(r"(\$\$.+?\$\$|\$[^$\n]+?\$|\\\[.+?\\\]|\\\(.+?\\\))", re.S)


def protect_math(text: str) -> tuple[str, list[str]]:
    """把 LaTeX 挖出来换成占位符，避免 markdown 把 _ * 当成强调。"""
    store: list[str] = []

    def repl(m):
        store.append(m.group(1))
        return f"@@MATH{len(store) - 1}@@"

    return MATH_RE.sub(repl, text), store


def restore_math(html_text: str, store: list[str]) -> str:
    for i, expr in enumerate(store):
        html_text = html_text.replace(f"@@MATH{i}@@", html.escape(expr, quote=False))
    return html_text


def render(doc: dict) -> str:
    protected, math_store = protect_math(doc["raw"])
    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", "toc", "attr_list"],
                           extension_configs={"toc": {"permalink": False}})
    body = md.convert(protected)
    body = restore_math(body, math_store)
    tokens = md.toc_tokens

    # 1) 标题 id 加文档前缀，避免跨文档重名
    def fix_id(m):
        level, old = m.group(1), m.group(2)
        return f'<h{level} id="{doc["id"]}-{old}">'

    body = re.sub(r'<h([1-6]) id="([^"]+)">', fix_id, body)

    # 2) 文档间链接 name.md / name.md#anc -> #doc-xx
    for other in DOCS:
        pat = re.escape(other["path"].name)
        body = re.sub(r'href="' + pat + r'(#[^"]*)?"', f'href="#{other["id"]}"', body)

    # 3) 表格外面包一层，方便横向滚动
    body = re.sub(r"(<table>.*?</table>)", r'<div class="tablewrap">\1</div>', body, flags=re.S)

    doc["html"] = body
    doc["toc"] = tokens
    return body


def build_toc(doc: dict) -> str:
    parts = []

    def walk(items, level):
        out = ["<ul>"]
        for it in items:
            cls = ' class="h3"' if level >= 3 else ""
            inner = ""
            if it.get("children") and level < 3:
                inner = walk(it["children"], level + 1)
            out.append(f'<li><a{cls} href="#{doc["id"]}-{it["id"]}">'
                       f'{html.escape(it["name"])}</a>{inner}</li>')
        out.append("</ul>")
        return "".join(out)

    # 只显示 H2 及其下一层（H3）
    for it in doc["toc"]:
        if it["level"] != 1:
            parts.append(walk([it], 2))
    inner = "".join(parts)
    text = (doc["title"] + " " + re.sub(r"<[^>]+>", " ", inner)).lower()
    data = html.escape(text, quote=True)
    label = f'{doc["num"]} {html.escape(doc["title"])}' if doc["num"] else html.escape(doc["title"])
    return (f'<details data-doc="{doc["id"]}" data-text="{data}">'
            f'<summary>{label}</summary>{inner}</details>')


DOCS = collect()

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="把 rl-study/*.md 拼成带大纲的 HTML 书")
    ap.add_argument("--only", default=None, help="只包含文件名前缀匹配的文档，例如 --only 10")
    ap.add_argument("--out", default=None, help="输出路径（默认 rl-study/index.html）")
    args = ap.parse_args()

    if args.only:
        DOCS = [d for d in DOCS if d["num"] == args.only or d["path"].name.startswith(args.only)]
    if args.out:
        OUT = Path(args.out)

    if not DOCS:
        sys.exit(f"{SRC_DIR} 里没有 markdown 文件")

    for d in DOCS:
        render(d)

    toc_html = "\n".join(build_toc(d) for d in DOCS)
    body_html = "\n".join(
        f'<section class="doc" id="{d["id"]}">\n{d["html"]}\n</section>' for d in DOCS
    )
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{BOOK_TITLE}</title>
<link rel="stylesheet" href="assets/katex/katex.min.css">
<style>{CSS}</style></head>
<body>
<nav id="toc">
  <div class="brand">{BOOK_TITLE}<small>{len(DOCS)} 篇 · 生成于 {stamp}</small></div>
  <input id="q" type="search" placeholder="搜索标题 / 筛选文档…">
  {toc_html}
</nav>
<main>
  <h1 class="title">{BOOK_TITLE}</h1>
  <div class="sub">点击左侧大纲跳转；Ctrl+F 页内查找；Ctrl+P 打印。生成脚本：<code>scripts/md_book.py</code></div>
  {body_html}
</main>
<button id="top" title="回到顶部">↑</button>
<script src="assets/katex/katex.min.js"></script>
<script src="assets/katex/auto-render.min.js"></script>
<script>
renderMathInElement(document.querySelector('main'), {{
  delimiters: [
    {{left: '$$', right: '$$', display: true}},
    {{left: '\\[', right: '\\]', display: true}},
    {{left: '$', right: '$', display: false}},
    {{left: '\\(', right: '\\)', display: false}}
  ],
  throwOnError: false
}});
</script>
<script>{JS}</script>
</body></html>
"""
    OUT.write_text(page, encoding="utf-8")
    try:
        shown = OUT.relative_to(ROOT)
    except ValueError:
        shown = OUT
    print(f"已生成 {shown}  ({len(page) / 1024:.0f} KB, {len(DOCS)} 篇)")
    for d in DOCS:
        print(f"  - {d['num'] or '--'}  {d['title']}")
