# `runs/`

Run-level artifacts and manifests will be consolidated in this directory.
Recommended naming convention: `runs/<YYYYMMDD_HHMMSS>_<tag>/`, containing:

* **`manifest.json`**: Records the git commit, script versions, input UIDs, windowing parameters, weak-labeling rules, and other metadata.
* **`state/`**: Snapshot of runtime state, including `candidate_takes.*`, `ready_for_*` flags, and download logs.
* **`qa/`, `windows/`, `weak_labels/`, `notebook_reports/`**: Outputs from each stage of the pipeline.

Currently, only the skeleton structure is created; it will be populated once the new run-ID workflow is integrated.

