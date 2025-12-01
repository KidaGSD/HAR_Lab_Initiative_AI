#!/bin/bash
# Label Validator Quick Start

echo "Generating validator for video: 9b49246f-30e9-476f-ab8c-56a1bcb1d936"
python tools/label_validator.py \
    --video-uid "9b49246f-30e9-476f-ab8c-56a1bcb1d936" \
    --video-manifest data/ego4d_data/ego4d.json \
    --output validator.html

echo ""
echo "✓ Validator HTML created: validator.html"
echo ""
echo "Next step: Start HTTP server in another terminal:"
echo "  python3 -m http.server 8000"
echo ""
echo "Then open: http://localhost:8000/validator.html"