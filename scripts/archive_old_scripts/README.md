# Archive: Old/Unused Scripts

These scripts are from previous experiments and approaches that have been superseded by the current hierarchical methodology documented in `RESEARCH_DESIGN.md`.

## Scripts Moved Here

1. **debug_s3_paths.py**: S3 path debugging (no longer needed after switching to direct download)
2. **diagnose_training.py**: Training diagnostics for old SSL+SVDD approach
3. **generate_candidates.py**: Legacy video candidate selection
4. **label_with_narrations.py**: Old narration labeling method (superseded by `map_narrations_to_actions.py`)
5. **manage_disk_space.py**: Disk space management utility
6. **prepare_vlm_clips.py**: VLM clip preparation (future work, not current priority)
7. **run_ego4d_pipeline.py**: Old data processing pipeline (being replaced by `process_imu_data.py`)
8. **train_ego4d_model.py**: SSL+SVDD training approach (replaced by hierarchical LLE+HLA training)

## Why Archived?

These scripts implemented earlier approaches before we finalized the research design. Keeping them for reference but they are not part of the active pipeline.

## If You Need Them

Code is preserved as-is. To use, copy back to `scripts/` and modify as needed. However, the current pipeline (see `README.md`) is recommended.
