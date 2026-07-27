#!/usr/bin/env python3
"""
generate_blog.py

Reads every markdown file in blog/posts/*.md (frontmatter + body), converts
each into a themed HTML page at blog/<slug>.html, and regenerates the
blog/index.html listing page.

No external dependencies (no pip installs) -- everything here is stdlib
plus a small hand-rolled markdown-to-HTML converter. Safe to re-run any
time; it fully regenerates output from the current markdown sources.

Usage:
    python3 blog/generate_blog.py          (run from anywhere)
    /usr/bin/python3 blog/generate_blog.py (if `python3` resolves to a
                                             broken shim on this machine)
"""

import html
import re
import sys
from pathlib import Path
from datetime import datetime

BLOG_DIR = Path(__file__).resolve().parent
POSTS_DIR = BLOG_DIR / "posts"

# ── Frontmatter parsing ──────────────────────────────────────────────

def parse_frontmatter(text):
    """Split a markdown file into (meta_dict, body_str).

    Expects a leading block delimited by lines that are exactly '---'.
    Frontmatter fields are simple 'key: value' pairs (no nested YAML).
    """
    meta = {}
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        # No frontmatter -- treat whole file as body.
        return meta, text

    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        # Malformed frontmatter (no closing ---) -- treat whole file as body.
        return meta, text

    for line in lines[1:end_idx]:
        if not line.strip() or ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip().lower()] = value.strip()

    body = "\n".join(lines[end_idx + 1:])
    return meta, body.lstrip("\n")


# ── Slug helper ──────────────────────────────────────────────────────

def slugify(value):
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "post"


# ── Inline markdown (bold/italic/code/links) ────────────────────────

def render_inline(text):
    """Escape HTML then apply inline markdown formatting."""
    # Pull out inline code spans first so their contents don't get
    # HTML-escaped twice or mangled by bold/italic/link regexes.
    placeholders = []

    def stash_code(m):
        placeholders.append(html.escape(m.group(1)))
        return "\x00%d\x00" % (len(placeholders) - 1)

    text = re.sub(r"`([^`]+)`", stash_code, text)

    # Escape remaining raw text.
    text = html.escape(text, quote=False)

    # Links: [text](url)
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: '<a href="%s">%s</a>' % (m.group(2), m.group(1)),
        text,
    )
    # Bold: **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"__(.+?)__", r"<strong>\1</strong>", text)
    # Italic: *text* or _text_
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<em>\1</em>", text)

    # Restore code spans.
    def restore_code(m):
        idx = int(m.group(1))
        return "<code>%s</code>" % placeholders[idx]

    text = re.sub(r"\x00(\d+)\x00", restore_code, text)
    return text


# ── Block-level markdown -> HTML ─────────────────────────────────────

def markdown_to_html(body):
    lines = body.splitlines()
    html_out = []
    i = 0
    n = len(lines)
    para_buf = []

    def flush_para():
        if para_buf:
            joined = " ".join(l.strip() for l in para_buf if l.strip())
            if joined:
                html_out.append("<p>%s</p>" % render_inline(joined))
            para_buf.clear()

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Fenced code block ```lang ... ```
        if stripped.startswith("```"):
            flush_para()
            lang = stripped[3:].strip()
            code_lines = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            code_escaped = html.escape("\n".join(code_lines))
            lang_class = ' class="language-%s"' % html.escape(lang) if lang else ""
            html_out.append(
                '<div class="code-card"><div class="code-card-head">'
                '<span class="code-tag">%s</span></div>'
                '<pre class="code-block"><code%s>%s</code></pre></div>'
                % (html.escape(lang.upper()) if lang else "CODE", lang_class, code_escaped)
            )
            continue

        # Headings
        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            flush_para()
            level = len(heading_match.group(1))
            # Shift down one level since the page already renders an <h1> title.
            tag = "h%d" % min(level + 1, 6)
            html_out.append("<%s>%s</%s>" % (tag, render_inline(heading_match.group(2)), tag))
            i += 1
            continue

        # Blockquote
        if stripped.startswith(">"):
            flush_para()
            quote_lines = []
            while i < n and lines[i].strip().startswith(">"):
                quote_lines.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            joined = " ".join(l.strip() for l in quote_lines if l.strip())
            html_out.append("<blockquote>%s</blockquote>" % render_inline(joined))
            continue

        # Unordered list
        if re.match(r"^[-*]\s+", stripped):
            flush_para()
            items = []
            while i < n and re.match(r"^[-*]\s+", lines[i].strip()):
                item_text = re.sub(r"^[-*]\s+", "", lines[i].strip())
                items.append(item_text)
                i += 1
            html_out.append(
                "<ul>%s</ul>" % "".join("<li>%s</li>" % render_inline(it) for it in items)
            )
            continue

        # Ordered list
        if re.match(r"^\d+\.\s+", stripped):
            flush_para()
            items = []
            while i < n and re.match(r"^\d+\.\s+", lines[i].strip()):
                item_text = re.sub(r"^\d+\.\s+", "", lines[i].strip())
                items.append(item_text)
                i += 1
            html_out.append(
                "<ol>%s</ol>" % "".join("<li>%s</li>" % render_inline(it) for it in items)
            )
            continue

        # Horizontal rule
        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", stripped):
            flush_para()
            html_out.append("<hr>")
            i += 1
            continue

        # Blank line -> paragraph break
        if not stripped:
            flush_para()
            i += 1
            continue

        # Otherwise: part of a paragraph
        para_buf.append(line)
        i += 1

    flush_para()
    return "\n".join(html_out)


# ── Date helpers ──────────────────────────────────────────────────────

def parse_date(value):
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d")
    except (ValueError, AttributeError):
        return datetime.min


def format_date_human(value):
    dt = parse_date(value)
    if dt == datetime.min:
        return value or ""
    return dt.strftime("%B %-d, %Y") if sys.platform != "win32" else dt.strftime("%B %d, %Y")


# ── Shared theme CSS (dark "luxury editorial" theme) ─────────────────

THEME_CSS = """
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#0a0a0b;--bg2:#111113;--bg3:#18181b;--bg4:#1e1e22;
  --text:#a1a1aa;--text-bright:#fafafa;--text-dim:#52525b;
  --accent:#c9a76a;--accent2:#e8c88a;--accent-glow:rgba(201,167,106,0.12);
  --green:#56d364;--blue:#58a6ff;
  --border:#27272a;--border-accent:rgba(201,167,106,0.2);
  --serif:Georgia,'Palatino Linotype','Book Antiqua',serif;
  --sans:'Inter','Segoe UI',system-ui,-apple-system,sans-serif;
  --mono:'SF Mono','Cascadia Code','Fira Code',monospace;
  --ease:cubic-bezier(.4,0,.2,1);
  --topbar-h:60px;
}
html{scroll-behavior:smooth;font-size:16px}
body{font-family:var(--sans);background:var(--bg);color:var(--text);line-height:1.7;-webkit-font-smoothing:antialiased;min-height:100vh}
::-webkit-scrollbar{width:8px;height:8px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}::-webkit-scrollbar-thumb:hover{background:var(--accent)}
a{color:var(--accent);text-decoration:none}
a:hover{color:var(--accent2)}

.topbar{position:fixed;top:0;left:0;right:0;height:var(--topbar-h);z-index:1000;display:flex;align-items:center;justify-content:space-between;padding:0 1.5rem;background:rgba(10,10,11,0.85);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border-bottom:1px solid var(--border)}
.topbar-left{display:flex;align-items:center;gap:1.2rem;min-width:0}
.home-link,.back-link{font-family:var(--mono);font-size:.78rem;color:var(--text-dim);text-decoration:none;letter-spacing:.05em;white-space:nowrap;transition:color .2s var(--ease)}
.home-link:hover,.back-link:hover{color:var(--accent)}
.topbar-title{font-family:var(--serif);font-size:1.05rem;color:var(--text-bright);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.topbar-title em{font-style:italic;color:var(--accent)}

.wrap{max-width:760px;margin:0 auto;padding:calc(var(--topbar-h) + 4rem) 1.5rem 6rem}

.section-label{font-family:var(--mono);font-size:.7rem;letter-spacing:.2em;text-transform:uppercase;color:var(--accent);margin-bottom:.8rem}
.page-title{font-family:var(--serif);font-size:clamp(2rem,4vw,3rem);font-weight:400;color:var(--text-bright);line-height:1.15;margin-bottom:1rem}
.page-desc{color:var(--text);max-width:640px;margin-bottom:2.5rem;font-size:1rem}

/* ── Post meta / tags ─────────────────────────────────────────── */
.post-meta{display:flex;align-items:center;gap:1rem;flex-wrap:wrap;font-family:var(--mono);font-size:.78rem;color:var(--text-dim);margin-bottom:2.5rem;padding-bottom:2rem;border-bottom:1px solid var(--border)}
.post-date{color:var(--accent)}
.tag{font-size:.68rem;padding:.25rem .65rem;border-radius:20px;background:var(--bg3);border:1px solid var(--border);color:var(--text-dim);letter-spacing:.03em}

/* ── Article body typography ──────────────────────────────────── */
.article h2{font-family:var(--serif);font-size:1.6rem;color:var(--text-bright);font-weight:400;margin:2.2rem 0 1rem}
.article h3{font-family:var(--serif);font-size:1.3rem;color:var(--text-bright);font-weight:400;margin:2rem 0 .8rem}
.article h4{font-family:var(--serif);font-size:1.1rem;color:var(--text-bright);font-weight:400;margin:1.6rem 0 .6rem}
.article p{font-size:.98rem;color:var(--text);line-height:1.85;margin-bottom:1.3rem}
.article ul,.article ol{margin:0 0 1.3rem 1.4rem;color:var(--text);font-size:.98rem;line-height:1.85}
.article li{margin-bottom:.4rem}
.article blockquote{border-left:3px solid var(--accent);background:var(--bg2);padding:.9rem 1.3rem;margin:0 0 1.3rem;border-radius:0 8px 8px 0;color:var(--text);font-style:italic}
.article hr{border:none;border-top:1px solid var(--border);margin:2.5rem 0}
.article code{font-family:var(--mono);font-size:.85em;background:var(--bg3);border:1px solid var(--border);border-radius:4px;padding:.1rem .4rem;color:var(--accent2)}
.article strong{color:var(--text-bright)}

.code-card{background:#0d0d0f;border:1px solid var(--border);border-radius:10px;margin:0 0 1.6rem;overflow:hidden}
.code-card-head{display:flex;align-items:center;padding:.5rem .9rem;background:var(--bg3);border-bottom:1px solid var(--border)}
.code-tag{font-family:var(--mono);font-size:.65rem;letter-spacing:.1em;color:var(--text-dim);text-transform:uppercase}
.code-block{padding:1rem 1.2rem;overflow-x:auto}
.code-block code{font-family:var(--mono);font-size:.83rem;line-height:1.65;color:var(--text);white-space:pre;background:none;border:none;padding:0}

.article-footer-nav{margin-top:3rem;padding-top:1.5rem;border-top:1px solid var(--border)}
.footer-note{padding:3rem 1.5rem;text-align:center;font-size:.75rem;color:var(--text-dim)}

/* ── Blog index listing ───────────────────────────────────────── */
.post-grid{display:grid;gap:1.3rem;margin-top:2.5rem}
.post-card{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:1.8rem 2rem;transition:all .3s var(--ease);text-decoration:none;display:block}
.post-card:hover{border-color:var(--border-accent);transform:translateY(-3px);box-shadow:0 16px 50px rgba(0,0,0,.35)}
.post-card-date{font-family:var(--mono);font-size:.72rem;letter-spacing:.05em;color:var(--accent);margin-bottom:.6rem}
.post-card h3{font-family:var(--serif);font-size:1.35rem;color:var(--text-bright);font-weight:400;margin-bottom:.6rem}
.post-card p{font-size:.9rem;color:var(--text);line-height:1.65;margin-bottom:1rem}
.post-card .post-tags{display:flex;gap:.5rem;flex-wrap:wrap}

.empty-state{text-align:center;padding:5rem 1.5rem;border:1px dashed var(--border);border-radius:12px;margin-top:2.5rem}
.empty-state p{font-size:1rem;color:var(--text-dim)}

@media(max-width:600px){
  .wrap{padding:calc(var(--topbar-h) + 2.5rem) 1.2rem 4rem}
  .post-card{padding:1.4rem 1.5rem}
}
"""

# ── Light theme override (per-post, via `theme: light` frontmatter) ──
# Approximates the clean white/sans-serif look of minimal engineering
# blogs (e.g. aleksagordic.com): white background, near-black text,
# system sans-serif + monospace, links underlined instead of colored.
THEME_CSS_LIGHT = """
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#ffffff;--bg2:#f7f7f8;--bg3:#f0f0f1;--bg4:#e4e4e7;
  --text:#3f3f46;--text-bright:#111111;--text-dim:#8a8a8f;
  --accent:#111111;--accent2:#000000;--accent-glow:rgba(17,17,17,0.05);
  --green:#1a7f37;--blue:#0969da;
  --border:#e4e4e7;--border-accent:#111111;
  --serif:Georgia,'Palatino Linotype','Book Antiqua',serif;
  --sans:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  --mono:'SF Mono','Menlo','Consolas','Liberation Mono',monospace;
  --ease:cubic-bezier(.4,0,.2,1);
  --topbar-h:60px;
}
html{scroll-behavior:smooth;font-size:16px}
body{font-family:var(--sans);background:var(--bg);color:var(--text);line-height:1.7;-webkit-font-smoothing:antialiased;min-height:100vh}
::-webkit-scrollbar{width:8px;height:8px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--bg4);border-radius:4px}::-webkit-scrollbar-thumb:hover{background:var(--text-dim)}
a{color:var(--text-bright);text-decoration:underline;text-decoration-color:var(--border-accent);text-underline-offset:2px}
a:hover{color:var(--blue)}

.topbar{position:fixed;top:0;left:0;right:0;height:var(--topbar-h);z-index:1000;display:flex;align-items:center;justify-content:space-between;padding:0 1.5rem;background:rgba(255,255,255,0.85);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border-bottom:1px solid var(--border)}
.topbar-left{display:flex;align-items:center;gap:1.2rem;min-width:0}
.home-link,.back-link{font-family:var(--mono);font-size:.78rem;color:var(--text-dim);text-decoration:none;letter-spacing:.05em;white-space:nowrap;transition:color .2s var(--ease)}
.home-link:hover,.back-link:hover{color:var(--text-bright)}
.topbar-title{font-family:var(--sans);font-weight:600;font-size:1.02rem;color:var(--text-bright);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.topbar-title em{font-style:normal;color:var(--text-dim)}

.wrap{max-width:760px;margin:0 auto;padding:calc(var(--topbar-h) + 4rem) 1.5rem 6rem}

.section-label{font-family:var(--mono);font-size:.7rem;letter-spacing:.15em;text-transform:uppercase;color:var(--text-dim);margin-bottom:.8rem}
.page-title{font-family:var(--sans);font-size:clamp(1.9rem,4vw,2.6rem);font-weight:700;color:var(--text-bright);line-height:1.2;margin-bottom:1rem;letter-spacing:-.02em}
.page-desc{color:var(--text);max-width:640px;margin-bottom:2.5rem;font-size:1rem}

.post-meta{display:flex;align-items:center;gap:1rem;flex-wrap:wrap;font-family:var(--mono);font-size:.78rem;color:var(--text-dim);margin-bottom:2.5rem;padding-bottom:2rem;border-bottom:1px solid var(--border)}
.post-date{color:var(--text-dim)}
.tag{font-size:.68rem;padding:.25rem .65rem;border-radius:20px;background:var(--bg2);border:1px solid var(--border);color:var(--text-dim);letter-spacing:.03em;font-family:var(--mono)}

.article h2{font-family:var(--sans);font-size:1.5rem;color:var(--text-bright);font-weight:700;margin:2.2rem 0 1rem;letter-spacing:-.01em}
.article h3{font-family:var(--sans);font-size:1.2rem;color:var(--text-bright);font-weight:700;margin:2rem 0 .8rem}
.article h4{font-family:var(--sans);font-size:1.05rem;color:var(--text-bright);font-weight:700;margin:1.6rem 0 .6rem}
.article p{font-size:.98rem;color:var(--text);line-height:1.85;margin-bottom:1.3rem}
.article ul,.article ol{margin:0 0 1.3rem 1.4rem;color:var(--text);font-size:.98rem;line-height:1.85}
.article li{margin-bottom:.5rem}
.article blockquote{border-left:3px solid var(--border-accent);background:var(--bg2);padding:.9rem 1.3rem;margin:0 0 1.3rem;border-radius:0 8px 8px 0;color:var(--text);font-style:italic}
.article hr{border:none;border-top:1px solid var(--border);margin:2.5rem 0}
.article code{font-family:var(--mono);font-size:.85em;background:var(--bg2);border:1px solid var(--border);border-radius:4px;padding:.1rem .4rem;color:#b5121b}
.article strong{color:var(--text-bright)}

.code-card{background:var(--bg2);border:1px solid var(--border);border-radius:10px;margin:0 0 1.6rem;overflow:hidden}
.code-card-head{display:flex;align-items:center;padding:.5rem .9rem;background:var(--bg3);border-bottom:1px solid var(--border)}
.code-tag{font-family:var(--mono);font-size:.65rem;letter-spacing:.1em;color:var(--text-dim);text-transform:uppercase}
.code-block{padding:1rem 1.2rem;overflow-x:auto}
.code-block code{font-family:var(--mono);font-size:.83rem;line-height:1.65;color:var(--text);white-space:pre;background:none;border:none;padding:0}

.article-footer-nav{margin-top:3rem;padding-top:1.5rem;border-top:1px solid var(--border)}
.footer-note{padding:3rem 1.5rem;text-align:center;font-size:.75rem;color:var(--text-dim)}

.post-grid{display:grid;gap:1.3rem;margin-top:2.5rem}
.post-card{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:1.8rem 2rem;transition:all .3s var(--ease);text-decoration:none;display:block}
.post-card:hover{border-color:var(--border-accent);transform:translateY(-2px);box-shadow:0 10px 30px rgba(0,0,0,.06)}
.post-card-date{font-family:var(--mono);font-size:.72rem;letter-spacing:.05em;color:var(--text-dim);margin-bottom:.6rem}
.post-card h3{font-family:var(--sans);font-size:1.25rem;color:var(--text-bright);font-weight:700;margin-bottom:.6rem}
.post-card p{font-size:.9rem;color:var(--text);line-height:1.65;margin-bottom:1rem}
.post-card .post-tags{display:flex;gap:.5rem;flex-wrap:wrap}

.empty-state{text-align:center;padding:5rem 1.5rem;border:1px dashed var(--border);border-radius:12px;margin-top:2.5rem}
.empty-state p{font-size:1rem;color:var(--text-dim)}

@media(max-width:600px){
  .wrap{padding:calc(var(--topbar-h) + 2.5rem) 1.2rem 4rem}
  .post-card{padding:1.4rem 1.5rem}
}
"""


def render_topbar(title_html, back_to_blog=True):
    left = '<a href="../index.html" class="home-link">&larr; punitvara.com</a>'
    if back_to_blog:
        left = (
            '<a href="index.html" class="back-link">&larr; Back to Blog</a>'
            '<span style="color:var(--border)" aria-hidden="true">|</span>'
            '<a href="../index.html" class="home-link">&larr; punitvara.com</a>'
        )
    return (
        '<nav class="topbar" aria-label="Blog">\n'
        '  <div class="topbar-left">%s</div>\n'
        '  <div class="topbar-title">%s</div>\n'
        "</nav>" % (left, title_html)
    )


def render_post_page(meta, slug, body_html):
    title = meta.get("title", slug)
    date_raw = meta.get("date", "")
    date_human = format_date_human(date_raw)
    summary = meta.get("summary", "")
    tags = [t.strip() for t in meta.get("tags", "").split(",") if t.strip()]
    css = THEME_CSS_LIGHT if meta.get("theme", "").strip().lower() == "light" else THEME_CSS

    tags_html = "".join('<span class="tag">%s</span>' % html.escape(t) for t in tags)
    meta_html = '<span class="post-date">%s</span>' % html.escape(date_human)
    if tags_html:
        meta_html += tags_html

    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — Blog — Punit Vara</title>
<meta name="description" content="{desc}">
<style>{css}</style>
</head>
<body>

{topbar}

<main class="wrap">
  <article class="reveal">
    <div class="section-label">Blog Post</div>
    <h1 class="page-title">{title}</h1>
    <div class="post-meta">{meta_html}</div>
    <div class="article">
{body}
    </div>
  </article>

  <div class="article-footer-nav">
    <a href="index.html" class="back-link">&larr; Back to all posts</a>
  </div>
</main>

<footer class="footer-note">
  <p>Built with curiosity. No frameworks, no dependencies — just code.</p>
</footer>

</body>
</html>
""".format(
        title=html.escape(title),
        desc=html.escape(summary or title),
        css=css,
        topbar=render_topbar('Blog &middot; <em>%s</em>' % html.escape(title), back_to_blog=True),
        meta_html=meta_html,
        body=body_html,
    )


def render_index_page(posts):
    if posts:
        cards = []
        for slug, meta in posts:
            title = meta.get("title", slug)
            date_human = format_date_human(meta.get("date", ""))
            summary = meta.get("summary", "")
            tags = [t.strip() for t in meta.get("tags", "").split(",") if t.strip()]
            tags_html = "".join('<span class="tag">%s</span>' % html.escape(t) for t in tags)
            cards.append(
                '<a href="{slug}.html" class="post-card">'
                '<div class="post-card-date">{date}</div>'
                "<h3>{title}</h3>"
                "<p>{summary}</p>"
                '<div class="post-tags">{tags}</div>'
                "</a>".format(
                    slug=slug,
                    date=html.escape(date_human),
                    title=html.escape(title),
                    summary=html.escape(summary),
                    tags=tags_html,
                )
            )
        grid = '<div class="post-grid">%s</div>' % "".join(cards)
    else:
        grid = (
            '<div class="empty-state"><p>No posts yet — check back soon.</p></div>'
        )

    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Blog — Punit Vara</title>
<meta name="description" content="Short, practical write-ups on GPU engineering, ML infrastructure, and systems programming from Punit Vara.">
<style>{css}</style>
</head>
<body>

{topbar}

<main class="wrap">
  <div class="section-label">Writing</div>
  <h1 class="page-title">Blog</h1>
  <p class="page-desc">Notes on GPU engineering, ML infrastructure, and systems programming — written as I learn and build.</p>

  {grid}
</main>

<footer class="footer-note">
  <p>Built with curiosity. No frameworks, no dependencies — just code.</p>
</footer>

</body>
</html>
""".format(
        css=THEME_CSS,
        topbar=render_topbar("Blog", back_to_blog=False),
        grid=grid,
    )


def main():
    if not POSTS_DIR.exists():
        print("No posts/ directory found at %s" % POSTS_DIR)
        POSTS_DIR.mkdir(parents=True, exist_ok=True)

    md_files = sorted(POSTS_DIR.glob("*.md"))
    posts = []  # list of (slug, meta)

    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        if "title" not in meta:
            meta["title"] = md_path.stem.replace("-", " ").title()
        slug = slugify(md_path.stem)
        body_html = markdown_to_html(body)

        out_path = BLOG_DIR / ("%s.html" % slug)
        out_path.write_text(render_post_page(meta, slug, body_html), encoding="utf-8")
        print("Generated %s" % out_path.relative_to(BLOG_DIR.parent))

        posts.append((slug, meta))

    # Sort by date descending (unparseable dates sort last).
    posts.sort(key=lambda item: parse_date(item[1].get("date", "")), reverse=True)

    index_path = BLOG_DIR / "index.html"
    index_path.write_text(render_index_page(posts), encoding="utf-8")
    print("Generated %s" % index_path.relative_to(BLOG_DIR.parent))
    print("Done: %d post(s)." % len(posts))


if __name__ == "__main__":
    main()
