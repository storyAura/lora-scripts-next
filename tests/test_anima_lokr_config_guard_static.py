import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from spa_asset_cache import SPA_ASSET_CACHE_KEY  # noqa: E402
HTML = ROOT / "frontend" / "dist" / "lora" / "sd3.html"
GUARD = ROOT / "frontend" / "dist" / "assets" / "anima-lokr-config-guard.js"
NAV = ROOT / "frontend" / "dist" / "assets" / "sd-nav-i18n.js"
LAYOUT = ROOT / "frontend" / "dist" / "assets" / "layout.96d49288.js"


class AnimaLokrConfigGuardStaticTests(unittest.TestCase):
    def test_shared_nav_loads_guard_on_direct_and_spa_navigation(self):
        html = HTML.read_text(encoding="utf-8")
        nav = NAV.read_text(encoding="utf-8")

        self.assertNotIn('<script src="/assets/anima-lokr-config-guard.js', html)
        self.assertIn("function ensureAnimaLokrConfigGuard()", nav)
        self.assertIn("ensureAnimaLokrConfigGuard();", nav)
        self.assertIn("ANIMA_LOKR_GUARD_PATH", nav)
        self.assertIn("mikazukiAnimaLokrGuardLoaded", nav)
        self.assertLess(
            nav.index("ensureAnimaLokrConfigGuard();", nav.index("function boot()")),
            nav.index("applyNavLocale();", nav.index("function boot()")),
        )

        pages = [
            path
            for path in (ROOT / "frontend" / "dist").rglob("*.html")
            if "sd-nav-i18n.js" in path.read_text(encoding="utf-8")
        ]
        self.assertTrue(pages)
        # Track the shared dist cache key instead of pinning a literal: every dist
        # patch bumps it via scripts/bump_spa_asset_cache_key.py.
        for page in pages:
            self.assertIn(
                f"sd-nav-i18n.js?v={SPA_ASSET_CACHE_KEY}",
                page.read_text(encoding="utf-8"),
                page,
            )

    def test_guard_is_route_scoped_without_persistent_blob_replacement(self):
        script = GUARD.read_text(encoding="utf-8")
        layout = LAYOUT.read_text(encoding="utf-8")

        self.assertIn("sanitizeLycorisToml", script)
        self.assertIn("installDownloadBlobForCurrentClick", script)
        self.assertIn("window.Blob = NativeBlob", script)
        self.assertNotIn("class SanitizedConfigBlob extends NativeBlob", script)
        self.assertIn("MutationObserver", script)
        self.assertIn(".params-section", script)
        self.assertIn("undefined|null|nan", script)
        self.assertNotIn("anima-lokr-config-guard", layout)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for dist JS smoke")
    def test_guard_preserves_comments_and_unrelated_toml(self):
        sample = """\
network_module = "lycoris.kohya"
network_args = [
  # "commented_out=undefined",
  "conv_dim=undefined",
  "conv_alpha=null",
  "dropout=nan",
  "algo=lokr",
  "factor=-1"
]
note = "conv_dim=undefined"
"""
        node_script = f"""
global.window = globalThis;
global.location = {{ pathname: "/lora/sd3.html" }};
global.document = {{
  readyState: "loading",
  addEventListener() {{}}
}};
global.window.addEventListener = function () {{}};
global.Node = {{ TEXT_NODE: 3 }};
global.NodeFilter = {{ SHOW_TEXT: 4 }};
global.MutationObserver = class {{}};
require({json.dumps(str(GUARD))});
const input = {json.dumps(sample)};
const expected = ["algo=lokr", "factor=-1"];
const output = globalThis.mikazukiSanitizeLycorisTomlText(input);
if (output.includes('  "conv_dim=undefined",')) process.exit(2);
if (output.includes('  "conv_alpha=null",')) process.exit(3);
if (output.includes('  "dropout=nan",')) process.exit(4);
if (expected.some((item) => !output.includes(item))) process.exit(3);
if (!output.includes('# "commented_out=undefined",')) process.exit(5);
if (!output.includes('note = "conv_dim=undefined"')) process.exit(6);
if (globalThis.mikazukiSanitizeLycorisTomlText('note = "conv_dim=undefined"') !== 'note = "conv_dim=undefined"') process.exit(7);
"""
        subprocess.run(
            [shutil.which("node"), "-e", node_script],
            check=True,
            cwd=ROOT,
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for dist JS smoke")
    def test_guard_runs_after_spa_navigation_and_restores_blob(self):
        sample = """\
network_module = "lycoris.kohya"
network_args = [
  "conv_dim=undefined",
  "algo=lokr"
]
"""
        node_script = f"""
const nativeBlob = globalThis.Blob;
const listeners = {{}};
let observerCallback = null;
global.window = globalThis;
global.location = {{ pathname: "/lora/index.html" }};
global.Node = {{ TEXT_NODE: 3 }};
global.NodeFilter = {{ SHOW_TEXT: 4 }};

const previewParent = {{
  closest(selector) {{ return selector === ".params-section" ? {{}} : null; }}
}};
const previewText = {{
  nodeType: 3,
  nodeValue: {json.dumps(sample)},
  parentElement: previewParent
}};
const preview = {{ nodeType: 1 }};
const app = {{}};
global.document = {{
  readyState: "complete",
  querySelector(selector) {{
    if (selector === "#app") return app;
    if (selector === ".params-section") return preview;
    return null;
  }},
  createTreeWalker(root) {{
    let returned = false;
    return {{
      nextNode() {{
        if (root === preview && !returned) {{
          returned = true;
          return previewText;
        }}
        return null;
      }}
    }};
  }},
  addEventListener(type, callback) {{ listeners[type] = callback; }}
}};
global.window.addEventListener = function (type, callback) {{
  listeners["window:" + type] = callback;
}};
global.MutationObserver = class {{
  constructor(callback) {{ observerCallback = callback; }}
  observe(root, options) {{
    if (root !== app || !options.characterData || !options.childList || !options.subtree) {{
      process.exit(10);
    }}
  }}
}};

require({json.dumps(str(GUARD))});
if (!observerCallback || typeof listeners.click !== "function") process.exit(11);

// The guard is already resident, then the SPA changes route and Vue updates text.
location.pathname = "/lora/sd3.html";
observerCallback([{{ type: "characterData", target: previewText }}]);
if (previewText.nodeValue.includes('  "conv_dim=undefined",')) process.exit(12);
if (!previewText.nodeValue.includes('"algo=lokr"')) process.exit(13);

const button = {{
  textContent: "Download config",
  closest(selector) {{ return selector === ".right-container" ? {{}} : null; }}
}};
const target = {{
  closest(selector) {{ return selector === "button" ? button : null; }}
}};

(async function () {{
  const existingBlob = new nativeBlob(["existing"]);
  listeners.click({{ target }});
  if (globalThis.Blob === nativeBlob) process.exit(14);
  if (!(existingBlob instanceof globalThis.Blob)) process.exit(15);

  const downloaded = await new globalThis.Blob([{json.dumps(sample)}]).text();
  if (downloaded.includes('  "conv_dim=undefined",')) process.exit(16);
  if (!downloaded.includes('"algo=lokr"')) process.exit(17);

  await new Promise((resolve) => setTimeout(resolve, 10));
  if (globalThis.Blob !== nativeBlob) process.exit(18);

  location.pathname = "/lora/index.html";
  listeners.click({{ target }});
  if (globalThis.Blob !== nativeBlob) process.exit(19);
}})().catch(function (error) {{
  console.error(error);
  process.exit(20);
}});
"""
        subprocess.run(
            [shutil.which("node"), "-e", node_script],
            check=True,
            cwd=ROOT,
        )


if __name__ == "__main__":
    unittest.main()
