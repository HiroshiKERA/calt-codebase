#!/usr/bin/env python
"""
Build a self-contained, navigable HTML page from a Markdown file.

Usage
-----
    python build_docs.py                              # DOCUMENTATION.md -> DOCUMENTATION.html
    python build_docs.py AI_CONTEXT.md                # -> AI_CONTEXT.html
    python build_docs.py DOCUMENTATION.md -o docs/index.html
    python build_docs.py --site                       # build the GitHub Pages site in docs/

The output is a single HTML file (CSS inlined, no external dependencies, works
offline) with a left navigation sidebar: clickable table of contents, a filter
box to find a section by keyword, and automatic highlighting of the section you
are currently reading.

The Markdown file stays the single source of truth. Re-run this script after
editing it to regenerate the HTML.

`--site` builds the files served by GitHub Pages:
    DOCUMENTATION.md -> docs/index.html
    AI_CONTEXT.md    -> docs/ai-context.html
and rewrites cross-document links so they work on the published site.
"""

import sys
from pathlib import Path

import markdown

# Cross-document links: rewrite Markdown links between docs so they resolve on
# the published site (and in the local docs/ folder).
LINK_MAP = {
    "DOCUMENTATION.md": "index.html",
    "AI_CONTEXT.md": "ai-context.html",
}

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{
    --bg: #ffffff;
    --sidebar-bg: #f5f6f8;
    --border: #e1e4e8;
    --text: #24292f;
    --muted: #57606a;
    --accent: #4f46e5;
    --accent-soft: #eef0fd;
    --code-bg: #f6f8fa;
  }}
  * {{ box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: var(--text);
    background: var(--bg);
    line-height: 1.6;
  }}
  /* ---- layout ---- */
  .sidebar {{
    position: fixed;
    top: 0; left: 0; bottom: 0;
    width: 300px;
    background: var(--sidebar-bg);
    border-right: 1px solid var(--border);
    overflow-y: auto;
    padding: 20px 14px 60px;
  }}
  .content {{
    margin-left: 300px;
    padding: 40px 56px 120px;
    max-width: 920px;
  }}
  /* ---- sidebar header + filter ---- */
  .sidebar h2.brand {{
    font-size: 15px;
    margin: 4px 8px 14px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .04em;
  }}
  .filter {{
    width: 100%;
    padding: 8px 10px;
    margin-bottom: 14px;
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 14px;
    background: #fff;
  }}
  .filter:focus {{ outline: 2px solid var(--accent-soft); border-color: var(--accent); }}
  /* ---- TOC (python-markdown emits nested <ul>) ---- */
  .toc ul {{ list-style: none; margin: 0; padding-left: 0; }}
  .toc > ul > li > a {{ font-weight: 600; }}
  .toc ul ul {{ padding-left: 14px; border-left: 1px solid var(--border); margin-left: 8px; }}
  .toc li {{ margin: 1px 0; }}
  .toc a {{
    display: block;
    padding: 5px 10px;
    border-radius: 6px;
    color: var(--text);
    text-decoration: none;
    font-size: 13.5px;
  }}
  .toc a:hover {{ background: #e9ebee; }}
  .toc a.active {{ background: var(--accent-soft); color: var(--accent); font-weight: 600; }}
  .toc li.hidden {{ display: none; }}
  /* ---- content typography ---- */
  .content h1 {{ font-size: 30px; padding-bottom: .3em; border-bottom: 2px solid var(--border); }}
  .content h2 {{ font-size: 23px; margin-top: 2.2em; padding-bottom: .25em; border-bottom: 1px solid var(--border); scroll-margin-top: 20px; }}
  .content h3 {{ font-size: 18px; margin-top: 1.6em; scroll-margin-top: 20px; }}
  .content a {{ color: var(--accent); }}
  .content table {{ border-collapse: collapse; width: 100%; margin: 1.2em 0; font-size: 14px; }}
  .content th, .content td {{ border: 1px solid var(--border); padding: 7px 12px; text-align: left; }}
  .content th {{ background: var(--code-bg); }}
  .content tr:nth-child(even) {{ background: #fafbfc; }}
  .content code {{
    background: var(--code-bg);
    padding: .15em .4em;
    border-radius: 5px;
    font-size: 85%;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  }}
  .content pre {{
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
    overflow-x: auto;
    line-height: 1.45;
  }}
  .content pre code {{ background: none; padding: 0; font-size: 13px; }}
  .content blockquote {{
    border-left: 4px solid var(--accent);
    background: var(--accent-soft);
    margin: 1.2em 0;
    padding: .6em 1em;
    color: var(--muted);
    border-radius: 0 8px 8px 0;
  }}
  .content blockquote p {{ margin: .3em 0; }}
  hr {{ border: none; border-top: 1px solid var(--border); margin: 2.4em 0; }}
  /* ---- responsive ---- */
  .menu-toggle {{ display: none; }}
  @media (max-width: 860px) {{
    .sidebar {{ transform: translateX(-100%); transition: transform .2s; z-index: 20; }}
    .sidebar.open {{ transform: translateX(0); }}
    .content {{ margin-left: 0; padding: 70px 20px 80px; }}
    .menu-toggle {{
      display: block; position: fixed; top: 12px; left: 12px; z-index: 30;
      background: var(--accent); color: #fff; border: none; border-radius: 8px;
      padding: 8px 12px; font-size: 16px; cursor: pointer;
    }}
  }}
</style>
</head>
<body>
  <button class="menu-toggle" onclick="document.querySelector('.sidebar').classList.toggle('open')">☰</button>
  <nav class="sidebar">
    <h2 class="brand">{title}</h2>
    <input class="filter" type="text" placeholder="Filter sections…" oninput="filterToc(this.value)">
    <div class="toc">
      {toc}
    </div>
  </nav>
  <main class="content">
    {body}
  </main>
<script>
  // ---- filter sidebar by keyword ----
  function filterToc(q) {{
    q = q.trim().toLowerCase();
    document.querySelectorAll('.toc li').forEach(function (li) {{
      var a = li.querySelector('a');
      var text = a ? a.textContent.toLowerCase() : '';
      li.classList.toggle('hidden', q !== '' && !text.includes(q));
    }});
  }}
  // ---- scroll-spy: highlight the section currently in view ----
  var links = {{}};
  document.querySelectorAll('.toc a').forEach(function (a) {{
    var id = decodeURIComponent(a.getAttribute('href').slice(1));
    links[id] = a;
  }});
  var headings = Array.from(document.querySelectorAll('.content h2, .content h3'))
    .filter(function (h) {{ return links[h.id]; }});
  function onScroll() {{
    var pos = window.scrollY + 90;
    var current = null;
    headings.forEach(function (h) {{ if (h.offsetTop <= pos) current = h; }});
    Object.values(links).forEach(function (a) {{ a.classList.remove('active'); }});
    if (current && links[current.id]) {{
      links[current.id].classList.add('active');
      links[current.id].scrollIntoView({{ block: 'nearest' }});
    }}
  }}
  window.addEventListener('scroll', onScroll, {{ passive: true }});
  onScroll();
</script>
</body>
</html>
"""


def _rewrite_links(html: str) -> str:
    for md_name, html_name in LINK_MAP.items():
        html = html.replace(f'href="{md_name}"', f'href="{html_name}"')
        html = html.replace(f'href="{md_name}#', f'href="{html_name}#')
    return html


def build(src_path: Path, out_path: Path | None = None) -> Path:
    text = src_path.read_text(encoding="utf-8")

    md = markdown.Markdown(
        extensions=["toc", "tables", "fenced_code", "sane_lists", "attr_list"],
        extension_configs={"toc": {"toc_depth": "2-3"}},
    )
    body = _rewrite_links(md.convert(text))
    toc = md.toc  # nested <ul> of links, ids already injected into headings

    # Title = first H1 line, else the filename.
    title = src_path.stem
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break

    html = TEMPLATE.format(title=title, toc=toc, body=body)
    out_path = out_path or src_path.with_suffix(".html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def build_site() -> list[Path]:
    """Build the GitHub Pages site in docs/ (index.html + ai-context.html)."""
    docs = Path("docs")
    outputs = []
    pages = [
        (Path("DOCUMENTATION.md"), docs / "index.html"),
        (Path("AI_CONTEXT.md"), docs / "ai-context.html"),
    ]
    for src, out in pages:
        if src.exists():
            outputs.append(build(src, out))
    return outputs


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--site":
        for out in build_site():
            print(f"Built {out}  ({out.stat().st_size // 1024} KB)")
    else:
        src = Path(args[0]) if args else Path("DOCUMENTATION.md")
        out = None
        if "-o" in args:
            out = Path(args[args.index("-o") + 1])
        if not src.exists():
            sys.exit(f"File not found: {src}")
        result = build(src, out)
        print(f"Built {result}  ({result.stat().st_size // 1024} KB)")
