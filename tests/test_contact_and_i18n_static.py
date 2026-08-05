import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ABOUT_HTML = ROOT / "frontend" / "dist" / "other" / "about.html"
ABOUT_JS = ROOT / "frontend" / "dist" / "assets" / "about.html.b4807002.js"
NAV_I18N = ROOT / "frontend" / "dist" / "assets" / "sd-nav-i18n.js"
SCHEMA_DICT = ROOT / "frontend" / "dist" / "assets" / "sd-schema-i18n-en.json"
CHROME_DICT = ROOT / "frontend" / "dist" / "assets" / "sd-chrome-i18n-en.json"
HELP_DICT = ROOT / "frontend" / "dist" / "assets" / "sd-help-i18n-en.json"
SCHEMA_SRC = ROOT / "script" / "scratch" / "schema_zh_descs.json"


class ContactAndI18nStaticTests(unittest.TestCase):
    def test_about_contact_uses_storyaura_email_and_qq_group(self):
        html = ABOUT_HTML.read_text(encoding="utf-8")
        js = ABOUT_JS.read_text(encoding="utf-8")
        self.assertIn("storyaura@outlook.com", html)
        self.assertIn("917336925", html)
        self.assertNotIn("oulongchen273@outlook.com", html)
        self.assertNotIn("discord", html.lower())
        self.assertIn("storyaura@outlook.com", js)
        self.assertIn("917336925", js)
        self.assertNotIn("oulongchen273@outlook.com", js)
        self.assertNotIn("discord", js.lower())

    def test_nav_i18n_loads_external_dicts_and_covers_settings_title(self):
        nav = NAV_I18N.read_text(encoding="utf-8")
        self.assertIn("sd-schema-i18n-en.json", nav)
        self.assertIn("sd-chrome-i18n-en.json", nav)
        self.assertIn("sd-help-i18n-en.json", nav)
        self.assertIn('"训练 UI 设置": "Training UI Settings"', nav)
        self.assertIn("syncHelpIframeLocale", nav)
        self.assertIn("main.page .theme-default-content", nav)

    def test_sidebar_github_points_to_storyaura_repo(self):
        old = "https://github.com/wochenlong/lora-scripts-next"
        new = "https://github.com/storyAura/lora-scripts-story-next"
        index = (ROOT / "frontend" / "dist" / "index.html").read_text(encoding="utf-8")
        layout = (ROOT / "frontend" / "dist" / "assets" / "layout.96d49288.js").read_text(
            encoding="utf-8"
        )
        self.assertIn(f'Github <a class="icon" href="{new}"', index)
        bottom = index[index.find("sidebar-bottom") : index.find("</ul>", index.find("sidebar-bottom"))]
        self.assertNotIn(old, bottom)
        self.assertIn(f'createTextVNode(" Github "),createBaseVNode("a",{{class:"icon",href:"{new}"', layout)

    def test_next_page_mapping_does_not_collide_with_brand_name(self):
        nav = NAV_I18N.read_text(encoding="utf-8")
        chrome = json.loads(CHROME_DICT.read_text(encoding="utf-8"))
        # Bare "Next" reverse-maps to 下一页 and corrupts "Next Story Trainer".
        self.assertNotIn('下一页: "Next"', nav)
        self.assertIn('下一页: "Next page"', nav)
        self.assertIn('const BRAND = "Next Story Trainer"', nav)
        self.assertIn("AMBIGUOUS_EN", nav)
        self.assertEqual(chrome.get("下一页"), "Next page")
        self.assertNotIn("下一页 Story Trainer", nav)

    def test_changelog_hydration_template_matches_ssr_html(self):
        html = (ROOT / "frontend" / "dist" / "other" / "changelog.html").read_text(
            encoding="utf-8"
        )
        js = (
            ROOT / "frontend" / "dist" / "assets" / "changelog.html.e5f6a7b8.js"
        ).read_text(encoding="utf-8")
        inner = re.search(
            r'theme-default-content"><!--\[--><!--\]--><div>(.*?)</div><!--\[--><!--\]-->',
            html,
            re.S,
        ).group(1)
        tpl = re.search(r"const h=i\((`[\s\S]*`)\);function", js).group(1)[1:-1]
        body = tpl.replace("\\${", "${").replace("\\`", "`").replace("\\\\", "\\")
        self.assertEqual(body, inner)
        self.assertEqual(js.count("`"), 2)
        app_html = re.search(r"app\.547295de\.js\?v=[^\"]+", html).group(0)
        app_js = re.search(r"app\.547295de\.js\?v=[^\"]+", js).group(0)
        self.assertEqual(app_html, app_js)

    def test_schema_i18n_dict_covers_all_unique_descriptions(self):
        self.assertTrue(SCHEMA_DICT.is_file())
        self.assertTrue(CHROME_DICT.is_file())
        self.assertTrue(HELP_DICT.is_file())
        schema = json.loads(SCHEMA_DICT.read_text(encoding="utf-8"))
        chrome = json.loads(CHROME_DICT.read_text(encoding="utf-8"))
        help_dict = json.loads(HELP_DICT.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(schema), 407)
        self.assertTrue(all(str(v).strip() for v in schema.values()))
        self.assertIn("训练 UI 设置", chrome)
        self.assertIn("不懂的不要碰这个", chrome)
        self.assertGreaterEqual(len(help_dict), 100)
        if SCHEMA_SRC.is_file():
            src = json.loads(SCHEMA_SRC.read_text(encoding="utf-8"))
            self.assertTrue(set(src).issubset(set(schema)))

    def test_schema_markdown_descriptions_have_stripped_en_aliases(self):
        nav = NAV_I18N.read_text(encoding="utf-8")
        schema = json.loads(SCHEMA_DICT.read_text(encoding="utf-8"))
        self.assertIn("translateMarkdownBlocks", nav)
        self.assertIn("stripMarkdownMarkers", nav)
        # Markdown splits these into multiple text nodes; aliases must match textContent.
        cases = {
            "保存训练状态 配合 resume 参数可以继续从某个状态训练": "Save training state",
            "CLIP 跳过层数 玄学": "CLIP layers to skip",
            "危险 自定义参数，请输入 TOML 格式，将会直接覆盖当前界面内任何参数。实时更新，推荐写完后再粘贴过来": "Danger",
            "训练混合精度, RTX30系列以后也可以指定bf16": "mixed precision",
        }
        for key, needle in cases.items():
            self.assertIn(key, schema)
            self.assertIn(needle, schema[key])
            self.assertFalse(re.search(r"[\u4e00-\u9fff]", schema[key]), key)

    def test_lora_index_intro_synced_and_has_en_maps(self):
        html = (ROOT / "frontend" / "dist" / "lora" / "index.html").read_text(encoding="utf-8")
        js = (
            ROOT / "frontend" / "dist" / "assets" / "index.html.4896b94d.js"
        ).read_text(encoding="utf-8")
        chrome = json.loads(CHROME_DICT.read_text(encoding="utf-8"))
        self.assertIn("主推训练入口", html)
        self.assertIn("主推训练入口", js)
        self.assertNotIn("两种模式", html)
        self.assertNotIn("两种模式", js)
        self.assertNotIn("不要逞强", js)
        for key in (
            " — 主推训练入口（Anima DiT）",
            "打标、看日志、脚本工具等在侧栏 ",
            "本 LoRA 训练界面分为两种模式。",
            "如果你是新手，建议使用新手模式，不要逞强使用专家模式，否则可能会出现意想不到的问题。",
        ):
            self.assertIn(key, chrome)
            self.assertFalse(re.search(r"[\u4e00-\u9fff]", chrome[key]), key)

    def test_home_author_github_and_lead_en_maps(self):
        index = (ROOT / "frontend" / "dist" / "index.html").read_text(encoding="utf-8")
        home_js = (
            ROOT / "frontend" / "dist" / "assets" / "index.html.c6ef684b.js"
        ).read_text(encoding="utf-8")
        chrome = json.loads(CHROME_DICT.read_text(encoding="utf-8"))
        self.assertIn('n("storyAura")', home_js)
        self.assertIn('n("lora-scripts-story-next")', home_js)
        self.assertIn(
            'href="https://github.com/storyAura/lora-scripts-story-next"',
            index[index.find("Author") : index.find("Author") + 800],
        )
        self.assertIn(">storyAura<", index[index.find("Author") : index.find("Author") + 400])
        self.assertIn(
            ">lora-scripts-story-next<",
            index[index.find("Author") : index.find("Author") + 800],
        )
        # Lead paragraph is split across text nodes; chrome must cover Chinese fragments.
        for key in (
            "（Next Story Trainer）是 ",
            "（上游 wochenlong/lora-scripts-next）的下游 Story 分支，源自秋叶 ",
            "：主打 Anima DiT 训练与训练队列，在浏览器里配参数、一键开训。",
            "插件加速 · 进阶",
            "详细步骤见",
            "；秋叶用户迁移说明也在该页。参数释义 ·",
        ):
            self.assertIn(key, chrome)
            self.assertFalse(
                re.search(r"[\u4e00-\u9fff]", chrome[key]),
                f"Chinese left in EN value for {key!r}: {chrome[key]!r}",
            )


if __name__ == "__main__":
    unittest.main()
