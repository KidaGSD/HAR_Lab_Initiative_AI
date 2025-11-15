# runs/

Run-level 产物及 manifest 将集中保存在此目录，命名建议 `runs/<YYYYMMDD_HHMMSS>_<tag>/`，包含：

- `manifest.json`: 记录 git commit、脚本版本、输入 UID、窗口参数、弱标签规则等。
- `state/`: 运行时使用的 `candidate_takes.*`, `ready_for_*`, 下载日志快照。
- `qa/`, `windows/`, `weak_labels/`, `notebook_reports/`: 各阶段输出。

当前仅创建骨架，等待新的 run-id 流程接入。
