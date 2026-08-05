"""Process UI deliverables from doc/local into committed assets."""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "doc" / "local" / "Next Story Trainer"
ICON_SRC = SRC / "app-icon.png"
if not ICON_SRC.is_file():
    ICON_SRC = SRC / "next-story-trainer-main-logo-transparent.png"
BANNER_SRC = SRC / "changelog-banner.png"
DIST = ROOT / "frontend" / "dist"
FAVICON_VERSION = "20260805-nst"


def cover_resize(img: Image.Image, size: tuple[int, int], bg: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    tw, th = size
    src = img.convert("RGBA")
    sw, sh = src.size
    scale = max(tw / sw, th / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    resized = src.resize((nw, nh), Image.Resampling.LANCZOS)
    x0 = (nw - tw) // 2
    y0 = (nh - th) // 2
    cropped = resized.crop((x0, y0, x0 + tw, y0 + th))
    out = Image.new("RGB", size, bg)
    out.paste(cropped, mask=cropped.split()[3])
    return out


def square_icon(img: Image.Image, size: int) -> Image.Image:
    src = img.convert("RGBA")
    sw, sh = src.size
    side = min(sw, sh)
    x0 = (sw - side) // 2
    y0 = (sh - side) // 2
    cropped = src.crop((x0, y0, x0 + side, y0 + side))
    return cropped.resize((size, size), Image.Resampling.LANCZOS)


def favicon_head_html() -> str:
    v = FAVICON_VERSION
    return (
        f'    <link rel="icon" href="/favicon.ico?v={v}" sizes="any">\n'
        f'    <link rel="icon" href="/assets/icon.65fd68ba.webp?v={v}" type="image/webp" sizes="512x512">\n'
        f'    <link rel="apple-touch-icon" href="/assets/icon.png?v={v}">'
    )


def patch_dist_favicon_links() -> None:
    head = favicon_head_html()
    icon_link_re = re.compile(r'\s*<link rel="(?:icon|apple-touch-icon)"[^>]*>\s*', re.I)
    for html_path in DIST.rglob("*.html"):
        text = html_path.read_text(encoding="utf-8")
        text = icon_link_re.sub("\n", text)
        marker = '<meta name="viewport"'
        idx = text.find(marker)
        if idx < 0:
            continue
        end = text.find(">", idx) + 1
        text = text[:end] + "\n" + head + text[end:]
        html_path.write_text(text, encoding="utf-8")
    print(f"patched favicon links in {DIST.relative_to(ROOT)}/**/*.html")


def patch_home_icon_cache_buster() -> None:
    index_js = DIST / "assets" / "index.html.c6ef684b.js"
    text = index_js.read_text(encoding="utf-8")
    text = text.replace(
        'var t="/assets/icon.65fd68ba.webp"',
        f'var t="/assets/icon.65fd68ba.webp?v={FAVICON_VERSION}"',
    )
    index_js.write_text(text, encoding="utf-8")

    index_html = DIST / "index.html"
    html = index_html.read_text(encoding="utf-8")
    html = html.replace(
        'src="/assets/icon.65fd68ba.webp"',
        f'src="/assets/icon.65fd68ba.webp?v={FAVICON_VERSION}"',
    )
    index_html.write_text(html, encoding="utf-8")
    print("patched home icon cache buster")


def patch_train_monitor_favicon() -> None:
    path = ROOT / "train_monitor" / "index.html"
    text = path.read_text(encoding="utf-8")
    text = re.sub(r'<link rel="icon" href="[^"]*"[^>]*>', "", text)
    text = text.replace(
        "<head>",
        f"<head>\n  <link rel=\"icon\" href=\"/favicon.ico?v={FAVICON_VERSION}\" sizes=\"any\">",
        1,
    )
    path.write_text(text, encoding="utf-8")
    print("patched train_monitor/index.html favicon")


def main() -> None:
    icon = Image.open(ICON_SRC)
    banner = Image.open(BANNER_SRC)

    readme_dir = ROOT / "assets" / "readme"
    readme_dir.mkdir(parents=True, exist_ok=True)

    cover = cover_resize(banner, (1280, 720))
    cover.save(readme_dir / "next-trainer-cover.png", optimize=True)
    cover.save(readme_dir / "anima-cover.png", optimize=True)

    social = cover_resize(banner, (1280, 640))
    social.save(readme_dir / "next-trainer-social.png", optimize=True)

    logo = square_icon(icon, 256)
    logo.save(ROOT / "assets" / "logo.png", optimize=True)
    logo.save(DIST / "assets" / "icon.png", optimize=True)

    webp_src = square_icon(icon, 512)
    webp_src.save(DIST / "assets" / "icon.65fd68ba.webp", format="WEBP", quality=90, method=6)

    ico_master = square_icon(icon, 256)
    ico_sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    for dest in (ROOT / "assets" / "favicon.ico", DIST / "favicon.ico"):
        ico_master.save(dest, format="ICO", sizes=ico_sizes)

    patch_dist_favicon_links()
    patch_home_icon_cache_buster()
    patch_train_monitor_favicon()

    print("Wrote:")
    for p in [
        readme_dir / "next-trainer-cover.png",
        readme_dir / "anima-cover.png",
        readme_dir / "next-trainer-social.png",
        ROOT / "assets" / "logo.png",
        DIST / "assets" / "icon.png",
        DIST / "assets" / "icon.65fd68ba.webp",
        ROOT / "assets" / "favicon.ico",
        DIST / "favicon.ico",
    ]:
        print(f"  {p.relative_to(ROOT)} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
