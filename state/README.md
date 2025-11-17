
# `state/`

* `active/`: The directory containing the **currently in-use state files** (e.g., `candidate_takes.*`, `ready_for_alignment.txt`, `ready_for_windowing.txt`, `logs/download_*.json`, etc.).
  These files have been migrated here, and **symlinks** are preserved at the root directory to maintain compatibility with older scripts.

* If you need to reference **legacy download logs**, you can access them via the symlink:
  `state/active/logs -> outputs/legacy_run/logs`.

* After future runs are completed, you may **copy the snapshot of `active/` into `runs/<run_id>/state/`** so the experiment can be fully reproduced later.

