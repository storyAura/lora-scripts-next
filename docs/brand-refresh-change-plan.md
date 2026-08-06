# Next Story Trainer 品牌刷新代码变更清单

本清单用于把现有项目中的用户可见品牌统一为 `Next Story Trainer`。当前所选文件夹未包含完整仓库代码，因此本文件是“可直接放入仓库执行”的变更清单；进入真实仓库后按本清单逐项修改。

## 目标结果

- 软件内正式展示名统一为 `Next Story Trainer`。
- GitHub 展示、README 主仓库、clone 地址指向 `storyAura/lora-scripts-story-next`。
- favicon、首页 Logo、guide 引导插图、changelog banner、README 封面、社交预览图使用同一套线条化新品牌素材。
- `frontend/dist` 中 HTML SSR 与 JS hydration 的品牌文案一致。
- 重新运行品牌 patch 脚本后不会回退为 `Next Trainer` 或 `SD 训练 UI`。
- 上游致谢、许可证和历史来源仍保留 `wochenlong/lora-scripts-next`、`Akegarasu/lora-scripts`。

## 不改名契约

以下名称是运行路径、脚本入口、兼容目录或训练类型契约，不要因为品牌刷新而重命名：

```text
gui.py
run_gui.bat
run_gui.ps1
run_gui.sh
run_gui_cn.sh
run_gui_source.bat
run_gui_source.ps1
SD-Trainer/
/lora/sd3.html
sd3-lora
sd-trainer-brand.js
sd-trainer-ui-polish.css
sd-* CSS class
```

处理原则：

- 用户可见文案可以改为 `Next Story Trainer`。
- 文件名、目录名、路由名、训练类型名和 CSS class 不随品牌刷新重命名。
- `SD-Trainer` 可以作为兼容路径出现，但不作为产品正式自称。

## 链接策略

| 类型 | 处理 |
|---|---|
| Fork 自称 | 改为 `storyAura/lora-scripts-story-next` |
| README clone 地址 | 改为 `https://github.com/storyAura/lora-scripts-story-next.git` |
| 首页徽章 / GitHub badge | 改为 `storyAura/lora-scripts-story-next` |
| 上游致谢 | 保留 `wochenlong/lora-scripts-next` 与 `Akegarasu/lora-scripts` |
| Releases / 便携包下载源 | 已迁移至 `storyAura/lora-scripts-story-next`（徽章 / update_check / portable updater） |
| 历史 Issue / 历史 changelog | 不强行改写 |

## 素材源目录

建议使用：

```text
doc/local/Next Story Trainer/
```

本次已准备的素材：

```text
doc/local/Next Story Trainer/app-icon-line.jpg
doc/local/Next Story Trainer/home-logo-line.jpg
doc/local/Next Story Trainer/monitor-logo-line.jpg
doc/local/Next Story Trainer/guide-illustration-line.jpg
doc/local/Next Story Trainer/changelog-banner.jpg
doc/local/Next Story Trainer/logo.svg
assets/readme/next-story-trainer-cover-line.jpg
assets/readme/next-story-trainer-social-line.jpg
```

进入仓库后，如果脚本只接受 `png` 或 `webp`，先转换格式，再由脚本写入运行时路径。

## 先改源头脚本

### `scripts/process_next_trainer_assets.py`

做什么：

- 将素材源目录改为 `doc/local/Next Story Trainer/`。
- 将品牌名常量改为 `Next Story Trainer`。
- 输出 favicon、icon、README 封面、social 图时使用新素材。
- 保留脚本文件名，不重命名。

建议常量：

```python
BRAND_NAME = "Next Story Trainer"
BRAND_ASSET_DIR = ROOT / "doc" / "local" / "Next Story Trainer"
README_ASSET_DIR = ROOT / "assets" / "readme"
```

### `scripts/patch-brand-illustrations.py`

做什么：

- 将 `home-logo-line`、`guide-illustration-line`、`changelog-banner` 的输入源改为新目录。
- 输出仍写入：

```text
frontend/dist/assets/home-logo.webp
frontend/dist/assets/guide-mascot.webp
frontend/dist/assets/changelog-banner.webp
```

- 将 alt 文案改为 `Next Story Trainer`。
- 同步版本 query，避免浏览器缓存旧图。
- 如果后续愿意扩大改动，可把运行时文件名从 `guide-mascot.webp` 改为 `guide-illustration.webp`；若要最小化改动，则保留旧文件名但内容换成无角色线条插图。

### `scripts/patch-home-portals.py`

做什么：

- 首页主标题、lead、GitHub badge、repo 链接改为 `Next Story Trainer` 与 `storyAura/lora-scripts-story-next`。
- 功能入口名 `Anima`、`Fast`、`Flux`、`Stable Diffusion` 保持不变。
- 上游说明继续保留 `wochenlong/lora-scripts-next`。

### `scripts/patch-sidebar-nav.py`

做什么：

- 侧栏品牌名统一生成 `Next Story Trainer`。
- 不再生成 `Next Trainer`。
- 保留内部路由和 class 名。

### `scripts/patch-ui-brand-version.py`

做什么：

- `BRAND_TITLE`、chip title、repo 链接策略统一。
- 生成的 `frontend/dist/assets/sd-trainer-brand.js` 中应出现 `Next Story Trainer`。
- `sd-trainer-brand.js` 文件名保留。

建议 JS 结果：

```js
const BRAND_TITLE = "Next Story Trainer";
const BRAND_REPO = "storyAura/lora-scripts-story-next";
const UPSTREAM_REPO = "wochenlong/lora-scripts-next";
```

### `scripts/patch-nav-copy.py`

做什么：

- 页面标题后缀从 `SD 训练 UI` 改为 `Next Story Trainer`。
- 表格中的介绍文案统一正式名。
- 不改训练模式名、模型名、功能名。

### `scripts/patch-anima-fast-entry.py`

做什么：

- Fast 入口和 guide 品牌句中的 `Next Trainer` 改为 `Next Story Trainer`。
- `Anima Fast` 作为功能名保留。

### `scripts/spa_asset_cache.py` 和 `scripts/bump_spa_asset_cache_key.py`

做什么：

- dist 资源变化后 bump cache key。
- 确保 hard reload 后不会继续加载旧 `brand.js`、旧 Logo 或旧 favicon。

### `scripts/patch-home-changelog.py`

处理：

- 暂不重跑。
- 如果必须重跑，先修复其旧品牌输出，避免把 `other/changelog.html` 回退。

## 运行时图片文件

需要替换或生成：

```text
assets/favicon.ico
assets/logo.png
frontend/dist/favicon.ico
frontend/dist/assets/icon.png
frontend/dist/assets/icon.*.webp
frontend/dist/assets/home-logo.webp
frontend/dist/assets/guide-mascot.webp
frontend/dist/assets/changelog-banner.webp
assets/readme/logo.svg
assets/readme/next-story-trainer-cover-line.png
assets/readme/next-story-trainer-social-line.png
```

如果为了减少 README diff 继续沿用旧文件名，可使用：

```text
assets/readme/next-trainer-cover.png
assets/readme/next-trainer-social.png
```

但必须更新 README 中的 `alt="Next Story Trainer"`。

## HTML 和 JS 文案

必须成对修改 HTML SSR 与 JS hydration。重点路径：

```text
frontend/dist/index.html
frontend/dist/help/guide.html
frontend/dist/other/about.html
frontend/dist/other/changelog.html
frontend/dist/**/*.html
frontend/dist/assets/index.html.*.js
frontend/dist/assets/guide.html.*.js
frontend/dist/assets/changelog.html.*.js
frontend/dist/assets/app.*.js
frontend/dist/assets/sd-trainer-brand.js
```

修改规则：

- `Next Trainer` → `Next Story Trainer`，仅限用户可见品牌文案。
- `SD 训练 UI` → `Next Story Trainer`，用于页面标题后缀。
- `wochenlong/lora-scripts-next` 不全局替换；只在 GitHub 展示归属位置改为 `storyAura/lora-scripts-story-next`。
- `SD-Trainer` 不全局替换；只在窗口标题、启动日志等用户展示字样中改。

建议页面标题：

```html
<title>Next Story Trainer</title>
<title>训练监控 — Next Story Trainer</title>
<title>训练参数 | Next Story Trainer</title>
```

## 训练监控页

需要修改：

```text
train_monitor/index.html
train_monitor/monitor.css
train_monitor/server.py
mikazuki/static/train_log.html
```

做法：

- `assets/logo.png` 替换为新监控页 Logo。
- `logo alt` 改为 `Next Story Trainer`。
- `title` 改为 `训练监控 — Next Story Trainer`。
- `monitor.css` 保留暗色科技风，只同步少量品牌变量：

```css
:root {
  --nst-monitor-bg: #0F172A;
  --nst-monitor-panel: #111827;
  --nst-monitor-accent: #A78BFA;
  --nst-monitor-chart: #38BDF8;
}
```

## UI polish CSS

主要修改：

```text
frontend/dist/assets/sd-trainer-ui-polish.css
```

谨慎修改：

```text
frontend/dist/assets/style.*.css
```

做法：

- 在 `sd-trainer-ui-polish.css` 顶部加入 `--nst-*` 变量。
- 使用变量覆盖首页、guide、changelog、brand chip 的颜色和阴影。
- 保留所有 `sd-*` selector。
- 注释中可写 `Next Story Trainer brand polish`。

不要做：

- 不重命名 `sd-trainer-ui-polish.css`。
- 不把 `style.*.css` 当作长期维护源头。
- 不为了品牌刷新大规模重构 Element 组件样式。

## README 与公开文档

需要修改：

```text
README.md
README-zh.md
NOTICE.md
CHANGELOG.md
CONTRIBUTORS.md
docs/anima-fast.md
docs/anima-training.md
docs/portable-getting-started.md
docs/portable-upgrade.md
docs/repo-layout.md
docs/train-monitor.md
docs/flash-attention.md
docs/tagger-models.md
docs/docker.md
docs/cli-args.md
```

README 英文建议：

```md
# Next Story Trainer

One-click LoRA & full finetune training GUI for Windows.

<sub>
A story-focused fork maintained at `storyAura/lora-scripts-story-next`,
based on `wochenlong/lora-scripts-next` and the Akegarasu-style GUI.
</sub>
```

README 中文建议：

```md
# Next Story Trainer

面向 Windows 的一键 LoRA 与全量微调训练图形界面。

<sub>
当前仓库由 `storyAura/lora-scripts-story-next` 维护，基于
`wochenlong/lora-scripts-next` 继续扩展，并保留 Akegarasu 风格 GUI 与上游致谢。
</sub>
```

CHANGELOG 建议新增当前条目：

```md
## Unreleased

- Brand: 统一产品展示名为 `Next Story Trainer`，更新 Logo、favicon、README 封面、监控页标题和 WebUI 品牌插图。
```

NOTICE 处理：

- 不删除上游版权和许可证。
- 可以补充 `Next Story Trainer` 是当前 fork 展示名。

## 启动器与便携包展示字

需要修改：

```text
scripts/portable/launch_portable.bat
Download-Anima-Model.bat
run_gui.bat
run_gui.ps1
run_gui_source.bat
run_gui_source.ps1
```

可改：

```bat
title Next Story Trainer
echo Starting Next Story Trainer...
```

不可改：

```bat
set APP_DIR=SD-Trainer
```

PowerShell 同理，只改展示字，不改脚本名和目录名。

## 配置预设

需要检查：

```text
config/presets/*.toml
```

只改用户可见 `author` 或 `display_name`，不改训练参数。

示例：

```toml
author = "Next Story Trainer"
```

## 建议执行顺序

1. 复制品牌源素材到 `doc/local/Next Story Trainer/`。
2. 写入 `docs/brand-design-tokens.md`。
3. 修改 `scripts/process_next_trainer_assets.py` 和 `scripts/patch-brand-illustrations.py`。
4. 修改首页、侧栏、导航、Fast 入口相关 patch 脚本。
5. 修改 UI brand version 与 cache key 脚本。
6. 运行素材处理脚本，生成运行时图标和插图。
7. 修改 HTML SSR 与 JS hydration 文案。
8. 修改 README、README-zh、NOTICE、CHANGELOG、CONTRIBUTORS 和 docs。
9. 修改启动器展示字和 preset author。
10. bump `spa_asset_cache.py`。
11. 运行测试和品牌搜索验证。

## 搜索验证

搜索旧品牌：

```powershell
Select-String -Path "README.md","README-zh.md","NOTICE.md","CHANGELOG.md","CONTRIBUTORS.md" -Pattern "Next Trainer","SD 训练 UI" -CaseSensitive
```

建议覆盖：

```text
frontend/dist/**/*.html
frontend/dist/assets/*.js
frontend/dist/assets/*.css
train_monitor/index.html
mikazuki/static/train_log.html
docs/**/*.md
scripts/**/*.py
scripts/**/*.bat
scripts/**/*.ps1
config/presets/*.toml
```

判断规则：

- 用户可见自称里不应再出现 `Next Trainer` 或 `SD 训练 UI`。
- 技术兼容名可以保留。
- 上游致谢可以保留。

## 页面验证

重点检查：

```text
frontend/dist/index.html
frontend/dist/help/guide.html
frontend/dist/other/about.html
frontend/dist/other/changelog.html
train_monitor/index.html
mikazuki/static/train_log.html
```

验收：

- 页面标题显示 `Next Story Trainer`。
- 训练监控标题显示 `训练监控 — Next Story Trainer`。
- Logo alt 显示 `Next Story Trainer`。
- 浏览器强刷后不出现旧 Logo 或旧 brand.js。

## 脚本幂等验证

重新运行后不能回退旧品牌：

```text
scripts/process_next_trainer_assets.py
scripts/patch-brand-illustrations.py
scripts/patch-home-portals.py
scripts/patch-sidebar-nav.py
scripts/patch-ui-brand-version.py
scripts/patch-nav-copy.py
scripts/patch-anima-fast-entry.py
```

验收：

- `frontend/dist` 中仍是 `Next Story Trainer`。
- `home-logo.webp`、`guide-mascot.webp` 或 `guide-illustration.webp`、`changelog-banner.webp` 仍是新素材。
- `sd-trainer-brand.js` 中品牌常量正确。
- cache key 已变化。

## 测试建议

运行已有测试：

```powershell
pytest tests/test_frontend_dist_cache.py
```

如果存在相关测试，也运行：

```powershell
pytest tests -k "brand or frontend or static"
```

可新增轻量断言：

```python
from pathlib import Path


def test_homepage_brand_title_is_next_story_trainer():
    html = Path("frontend/dist/index.html").read_text(encoding="utf-8")
    assert "Next Story Trainer" in html
    assert "SD 训练 UI" not in html


def test_runtime_brand_script_uses_current_name():
    js = Path("frontend/dist/assets/sd-trainer-brand.js").read_text(encoding="utf-8")
    assert "Next Story Trainer" in js
```

## 最终验收

- 侧栏名、首页 Logo、favicon 三处一致。
- 所有 `<title>` 不再把 `SD 训练 UI` 作为产品后缀。
- `README.md` 和 `README-zh.md` 首屏品牌一致。
- GitHub 展示归属为 `storyAura/lora-scripts-story-next`。
- 上游致谢和 AGPL 相关说明仍保留。
- patch 脚本重新运行不会把品牌倒退回 `Next Trainer`。
- `tests/test_frontend_dist_cache.py` 和相关 static 测试通过。
