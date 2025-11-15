# state/

- `active/`: 当前正在使用的状态文件（`candidate_takes.*`, `ready_for_alignment.txt`, `ready_for_windowing.txt`, `logs/download_*.json` 等）。文件已迁入，并在根目录保留指向这些文件的 symlink 以兼容旧脚本。
- 如需引用 legacy 下载日志，可通过 `state/active/logs -> outputs/legacy_run/logs` 的 symlink 访问。
- 未来运行完成后，可将 `active/` 快照复制到 `runs/<run_id>/state/`，以便复现实验。
