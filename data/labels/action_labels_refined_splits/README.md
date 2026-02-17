# Action Labels — Split for Label Studio

Chunks of `action_labels_llm_clean_refined.csv` for import into Label Studio (SQLite limit ~100k tasks).

- **Max 10k rows per chunk** — each chunk stays under Label Studio SQLite limits
- **Videos never split** — all rows for a `video_uid` are in the same chunk

## 1. Create chunks (already done)

```bash
python scripts/split_labels_for_labelstudio.py
```

## 2. Import into HAR_dataset project

```bash
# Install Label Studio SDK
pip install label-studio

# Set your Label Studio URL and API key
# IMPORTANT: Use a Legacy Token (Account → Legacy Tokens), NOT a Personal Access Token
export LABEL_STUDIO_URL="http://localhost:8080"   # or your instance URL
export LABEL_STUDIO_API_KEY="your_legacy_token"

# Import all chunks (or specific ones: --chunks 1 2 3)
python scripts/import_labelstudio_chunks.py

# Dry run to preview
python scripts/import_labelstudio_chunks.py --dry-run
```

Import one chunk at a time to avoid overloading:

```bash
python scripts/import_labelstudio_chunks.py --chunks 1
python scripts/import_labelstudio_chunks.py --chunks 2
# etc.
```

## Task format

Each row becomes a Label Studio task with `data`:

- `video_uid`, `timestamp_sec`, `narration_text`, `scenario`, `action`, `reasoning`, `status`

Configure your HAR_dataset labeling interface to display these fields (e.g. `$narration_text`, `$action` for verification).
