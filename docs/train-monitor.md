# 训练监控页

## 概述

训练监控页是一个独立的实时状态面板（默认端口 6008，可自动回退），随 GUI 自动启动并在浏览器中打开。

## 功能

- 实时训练状态卡片（状态 / 进度 / Epoch / ETA / Loss / LR / it/s）
- TensorBoard 同源 Loss / LR 曲线：`loss/average`、`loss/current`、`loss/epoch_average`、`lr/unet`
- 曲线范围按钮：全部 / 最近 50% / 最近 20% / 最近 10% / 恢复最新；保留滚轮缩放和拖拽平移
- 训练参数速查：总步数、学习率、优化器、分辨率、保存频率、精度、seed 等关键参数
- 训练预览图展示
- `./output` 下最新输出文件列表
- 训练日志同步显示在 CMD 终端与监控页；上翻看历史不会被新日志拖回去
- 静默正常轮询的 200/304 access log，终端后台只保留关键输出

### v2.9.2 数据对齐与防双训练

- 参数区优先读取**当前运行任务**的 `metadata.config_path`，避免被更新的其它 autosave 抢绑。
- TensorBoard 优先当前配置的 `logging_dir`；顶部 / 卡片 Loss、学习率与 TB scalar 同源回填。
- 停止训练时会等待整棵进程树退出后，队列才允许启动下一场，避免旧进程与新任务同时写 `output` / events（Loss、Epoch 乱跳）。

## 访问方式

- **自动打开** — GUI 启动后自动在浏览器打开实际监控端口
- **手动访问** — 优先从主 WebUI 打开 `/train-monitor`，后端会跳转到实际 `TRAIN_MONITOR_PORT`
- **指定任务** — URL 后加 `?task_id=<uuid>`
- **嵌入** — 同机嵌入优先使用 `/train-monitor`；直接访问子服务时以启动日志中的实际端口为准

## SSE 原始流接口

WebUI 启动训练时，后端会通过 Server-Sent Events 转发训练日志：

- **监控页面** — `http://127.0.0.1:28000/train-log`（自动发现当前运行任务）
- **原始流** — `GET /api/train/log/stream/{task_id}` 返回 `text/event-stream`

`task_id` 来自 `POST /api/run` 的返回值。如果 URL 里没带 `task_id`，监控页会自动查询 `GET /api/train/tasks` 选取最新任务。

## 配置

| 环境变量 / 参数 | 默认值 | 说明 |
|----------------|--------|------|
| `--train-monitor-port` | 6008 | 监控页端口 |
| `--disable-train-monitor` | false | 禁用训练监控 |
