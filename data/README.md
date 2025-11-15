# data/

- `raw/`: 挂载/存放原始 Ego-Exo4D 下载（目前计划用 symlink 指向根目录已有的 `egoexo_cache/`、`egoexo_cache_test/`）。
- `staging/`: 对齐/QA 等中间产物（例如 `processed_windows/qa`）。
- `processed/`: 窗口化输出、弱标签等经过整理的可复用数据；完成迁移后和 `outputs/` 的 run 目录互通。

> 清理阶段仅创建骨架，实际文件/链接会在后续步骤逐步迁移。
