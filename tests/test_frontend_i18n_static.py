import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAYOUT = ROOT / "frontend" / "dist" / "assets" / "layout.96d49288.js"
NAV_I18N = ROOT / "frontend" / "dist" / "assets" / "sd-nav-i18n.js"


class FrontendI18nStaticTests(unittest.TestCase):
    def test_chinese_locale_maps_hydrated_training_controls_exactly(self):
        layout = LAYOUT.read_text(encoding="utf-8")
        nav_i18n = NAV_I18N.read_text(encoding="utf-8")
        expected_pairs = (
            ('output_header:"Output"', '"参数预览": "Output"'),
            ('reset_all:"Reset All"', '全部重置: "Reset All"'),
            ('save_params:"Save Parameters"', '保存参数: "Save Parameters"'),
            ('read_params:"Read Parameters"', '读取参数: "Read Parameters"'),
            (
                'download_config:"Download Config File"',
                '下载配置文件: "Download Config File"',
            ),
            (
                'import_config:"Import Config File"',
                '导入配置文件: "Import Config File"',
            ),
            ('load_presets:"Load Presets"', '"✨加载训练预设✨": "Load Presets"'),
            ('start_train:"Start Training"', '开始训练: "Start Training"'),
            ('stop_train:"Stop Training"', '终止训练: "Stop Training"'),
        )

        self.assertIn("const EN_TO_ZH = Object.fromEntries", nav_i18n)
        for layout_fragment, nav_mapping in expected_pairs:
            with self.subTest(layout_fragment=layout_fragment):
                self.assertIn(layout_fragment, layout)
                self.assertIn(nav_mapping, nav_i18n)


if __name__ == "__main__":
    unittest.main()
