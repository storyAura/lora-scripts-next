# Next Story Trainer 品牌设计 Token

本文件定义 `Next Story Trainer` 品牌刷新所需的轻量设计 Token。目标是统一产品名、色彩、圆角、阴影、字体和动效，而不是重建完整设计系统。

## 品牌命名

| Token | Value | 用途 |
|---|---|---|
| `brand.name` | `Next Story Trainer` | 产品正式展示名 |
| `brand.repo` | `storyAura/lora-scripts-story-next` | GitHub 展示、clone 地址、README 主仓库 |
| `brand.upstream.parent` | `wochenlong/lora-scripts-next` | 上游 fork 来源、Releases 渠道说明 |
| `brand.upstream.source` | `Akegarasu/lora-scripts` | 原始项目致谢 |
| `brand.technical.sdTrainer` | `SD-Trainer` | 便携包目录、兼容路径，不作为新品牌展示名 |
| `brand.trainType.sd3` | `sd3-lora` | 训练类型契约，不改名 |

## 命名边界

应该改为 `Next Story Trainer` 的位置：

- 软件自称、首页标题、侧栏品牌名、页面 `<title>`。
- Logo、favicon、README 封面、社交预览图的 `alt` 文案。
- 用户可见的启动器标题和启动日志。
- 新增文档、CHANGELOG 的当前 rebrand 条目。

不应该全局替换的位置：

- `SD-Trainer/` 便携目录。
- `sd-trainer-brand.js`、`sd-trainer-ui-polish.css` 等技术文件名。
- `sd-*` CSS class。
- `/lora/sd3.html`、`sd3-lora` 等路由与训练类型。
- `wochenlong/lora-scripts-next`、`Akegarasu/lora-scripts` 的上游致谢、许可证和历史说明。

## 视觉边界

本次品牌刷新使用线条化方向：

- 使用 monoline 线条、书签轮廓、Loss 曲线、LoRA 轨道、训练节点、参数卡片作为核心视觉母题。
- 不使用吉祥物、看板娘、动物角色或人物角色。
- Logo 优先保持轻量、清晰、可在 favicon、README、监控页中缩放使用。
- 插图可以保留温暖浅色背景，但图形主体应以线条和轻量节点为主。

## 色彩

### 原始色

| Token | Value | 用途 |
|---|---|---|
| `color.raw.violet.500` | `#8B5CF6` | 主品牌紫 |
| `color.raw.violet.300` | `#C4B5FD` | 浅紫辅助 |
| `color.raw.pink.500` | `#EC4899` | 渐变和强调 |
| `color.raw.amber.300` | `#FCD34D` | 轨道节点、里程碑、温暖强调 |
| `color.raw.peach.300` | `#FDBA74` | 引导流程、节点和轻量强调 |
| `color.raw.cream.50` | `#FFF7ED` | 主站暖背景 |
| `color.raw.slate.800` | `#1F2937` | 主文本 |
| `color.raw.slate.500` | `#6B7280` | 次级文本 |
| `color.raw.navy.950` | `#0F172A` | 监控页暗背景 |
| `color.raw.navy.900` | `#111827` | 监控页面板 |
| `color.raw.sky.400` | `#38BDF8` | 监控曲线和数据强调 |

### 语义色

| Token | Value | 用途 |
|---|---|---|
| `color.brand.primary` | `{color.raw.violet.500}` | 主按钮、链接、徽章、关键高亮 |
| `color.brand.secondary` | `{color.raw.pink.500}` | 次级强调、渐变终点 |
| `color.brand.warmBg` | `{color.raw.cream.50}` | 首页、帮助页浅色背景 |
| `color.brand.surface` | `#FFFFFF` | 卡片、表单、浮层 |
| `color.brand.text` | `{color.raw.slate.800}` | 主文本 |
| `color.brand.muted` | `{color.raw.slate.500}` | 辅助文本、说明文字 |
| `color.brand.border` | `rgba(139, 92, 246, 0.18)` | 轻量卡片边框 |
| `color.brand.focus` | `rgba(139, 92, 246, 0.32)` | focus ring |
| `color.monitor.bg` | `{color.raw.navy.950}` | 训练监控页背景 |
| `color.monitor.panel` | `{color.raw.navy.900}` | 训练监控卡片 |
| `color.monitor.accent` | `#A78BFA` | 监控页品牌高亮 |
| `color.monitor.chart` | `{color.raw.sky.400}` | 监控曲线 |

## CSS 变量建议

```css
:root {
  --nst-brand-primary: #8B5CF6;
  --nst-brand-secondary: #EC4899;
  --nst-brand-warm-bg: #FFF7ED;
  --nst-surface: #FFFFFF;
  --nst-text: #1F2937;
  --nst-muted: #6B7280;
  --nst-border-soft: rgba(139, 92, 246, 0.18);
  --nst-focus-ring: rgba(139, 92, 246, 0.32);
}

.train-monitor {
  --nst-monitor-bg: #0F172A;
  --nst-monitor-panel: #111827;
  --nst-monitor-accent: #A78BFA;
  --nst-monitor-chart: #38BDF8;
}
```

## 字体

| Token | Value | 用途 |
|---|---|---|
| `font.family.sans` | `system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif` | 默认界面字体 |
| `font.family.mono` | `"SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace` | 日志、命令、配置预览 |
| `font.weight.regular` | `400` | 正文 |
| `font.weight.medium` | `500` | 表单标签、卡片标题 |
| `font.weight.semibold` | `600` | 导航、按钮 |
| `font.weight.bold` | `700` | 首页标题、README 封面 |

建议继续使用系统字体，不引入外部字体依赖，避免离线便携包加载失败。

## 字号与行高

| Token | Value | 用途 |
|---|---|---|
| `font.size.xs` | `12px` | tag、徽章、辅助说明 |
| `font.size.sm` | `14px` | 表单说明、次级文本 |
| `font.size.base` | `16px` | 默认正文 |
| `font.size.lg` | `18px` | 卡片标题 |
| `font.size.xl` | `24px` | 页面标题 |
| `font.size.hero` | `40px` | 首页 hero 标题 |
| `lineHeight.tight` | `1.2` | 标题 |
| `lineHeight.normal` | `1.55` | 正文 |
| `lineHeight.code` | `1.45` | 日志和代码 |

## 圆角

| Token | Value | 用途 |
|---|---|---|
| `radius.sm` | `8px` | tag、小按钮 |
| `radius.md` | `12px` | 输入框、普通按钮 |
| `radius.lg` | `18px` | 首页入口卡片、信息卡片 |
| `radius.xl` | `24px` | hero 区块、大卡片 |
| `radius.full` | `999px` | pill、徽章、进度胶囊 |

## 间距

| Token | Value | 用途 |
|---|---|---|
| `spacing.1` | `4px` | 紧凑间隔 |
| `spacing.2` | `8px` | 小组件内部 |
| `spacing.3` | `12px` | 表单组 |
| `spacing.4` | `16px` | 卡片内边距 |
| `spacing.5` | `20px` | 内容块间距 |
| `spacing.6` | `24px` | 页面区块 |
| `spacing.8` | `32px` | hero 和分组 |
| `spacing.12` | `48px` | 大区块 |

## 阴影

| Token | Value | 用途 |
|---|---|---|
| `shadow.card` | `0 12px 30px rgba(31, 41, 55, 0.08)` | 主站浅色卡片 |
| `shadow.cardHover` | `0 18px 42px rgba(139, 92, 246, 0.16)` | 卡片 hover |
| `shadow.popover` | `0 20px 50px rgba(31, 41, 55, 0.16)` | 浮层、下拉菜单 |
| `shadow.monitor` | `0 18px 40px rgba(0, 0, 0, 0.32)` | 监控页暗色卡片 |

## 动效

| Token | Value | 用途 |
|---|---|---|
| `motion.duration.fast` | `120ms` | hover、按钮反馈 |
| `motion.duration.normal` | `180ms` | 卡片、菜单 |
| `motion.duration.slow` | `260ms` | hero 轻动效 |
| `motion.easing.standard` | `cubic-bezier(0.2, 0, 0, 1)` | 默认缓动 |
| `motion.translate.hover` | `-2px` | 卡片 hover 上浮 |

实现时应尊重 `prefers-reduced-motion`：

```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    scroll-behavior: auto !important;
    transition-duration: 0.01ms !important;
  }
}
```

## 实现约束

- 新增品牌变量可使用 `--nst-*` 前缀。
- 不重命名现有 `sd-*` class。
- 不重命名 `sd-trainer-brand.js` 和 `sd-trainer-ui-polish.css`。
- 主要在 `frontend/dist/assets/sd-trainer-ui-polish.css` 做品牌覆盖。
- `frontend/dist/assets/style.*.css` 是构建产物，除非必要，不作为品牌源头直接维护。
- HTML SSR 和 JS hydration 中同一品牌字符串必须成对修改。
