#!/usr/bin/env python3
"""
Quick Moondream2 VLM probe harness.

Goals:
- Try prompt templates for "frame understanding" and structured JSON output.
- Support single image, or a small list of images (multi-frame by per-frame QA).

Example:
  /shared/ssd_14T/home/leopolddas/miniconda3/envs/har_ssh/bin/python scripts/vlm_moondream2_probe.py \
    --image data/anomalies/anomaly_47249b6c-5d9c-44dd-9ab9-a68fa59ccb3f_47.jpg \
    --scenario "Cooking" --action "Locomotion"
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "vikhyatk/moondream2"


def _load_images(args: argparse.Namespace) -> list[Path]:
    if args.image:
        return [Path(args.image)]
    if args.image_dir:
        p = Path(args.image_dir)
        if not p.exists():
            raise FileNotFoundError(f"--image-dir not found: {p}")
        images = sorted([x for x in p.rglob(args.pattern) if x.is_file()])
        if not images:
            raise FileNotFoundError(f"No images found in {p} matching pattern {args.pattern!r}")
        rng = random.Random(args.seed)
        rng.shuffle(images)
        return images[: args.limit]
    raise ValueError("Provide either --image or --image-dir")


def _get_dtype(dtype_str: str) -> torch.dtype | None:
    if dtype_str == "auto":
        return None
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def _safe_json_prompt() -> str:
    return (
        "You MUST respond with a single valid JSON object and nothing else. "
        "No markdown, no code fences."
    )


def build_questions(scenario: str | None, action: str | None) -> list[dict[str, str]]:
    # Keep these short; moondream2 tends to do better with compact, concrete questions.
    base: list[dict[str, str]] = [
        {"name": "caption", "q": "Describe this image in one sentence."},
        {"name": "setting", "q": "Where is the person and what are they doing?"},
        {"name": "objects", "q": "List the 5 most important objects you see."},
        {"name": "risk", "q": "Is there anything dangerous happening? If yes, what? If no, say no."},
    ]

    if scenario or action:
        sc = scenario if scenario else "Unknown"
        ac = action if action else "Unknown"
        base.append(
            {
                "name": "imu_context_check",
                "q": (
                    f"Context: IMU predicted scenario={sc} and action={ac}. "
                    "Based on the image, does this seem plausible? Answer: plausible / unclear / implausible, and why."
                ),
            }
        )
        base.append(
            {
                "name": "imu_context_json",
                "q": (
                    f"{_safe_json_prompt()} "
                    f"Given scenario={sc} and action={ac}, analyze the image and return JSON with keys: "
                    '{"status":"OK|ANOMALY|UNCLEAR","confidence":0-1,"reason":"...","evidence":["..."]}.'
                ),
            }
        )
        base.append(
            {
                "name": "labels_reasoning_json",
                "q": (
                    f"{_safe_json_prompt()} "
                    f"Given labels scenario={sc} and action={ac}, decide if the labels match the image. "
                    "Return JSON with keys: "
                    '{"scenario_action":"plausible|implausible|unclear","why":"..."}'
                ),
            }
        )
    else:
        base.append(
            {
                "name": "context_json",
                "q": (
                    f"{_safe_json_prompt()} "
                    'Analyze the image and return JSON with keys: '
                    '{"summary":"...","risk":"none|low|medium|high","evidence":["..."]}.'
                ),
            }
        )
    return base


def _extract_frames_ffmpeg(video_path: Path, seconds: list[float]) -> list[Path]:
    """
    Extract specific timestamps as JPG frames using ffmpeg.
    Returns paths to extracted images (in a temporary directory).
    """
    if not video_path.exists():
        raise FileNotFoundError(f"--video not found: {video_path}")

    tmpdir = Path(tempfile.mkdtemp(prefix="vlm_frames_"))
    out_paths: list[Path] = []

    for i, t in enumerate(seconds):
        out_path = tmpdir / f"frame_{i:03d}_{t:.2f}s.jpg"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(t),
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        out_paths.append(out_path)

    return out_paths


def _parse_json_maybe(s: str) -> tuple[bool, Any | None, str | None]:
    s = (s or "").strip()
    if not s:
        return False, None, "empty"
    try:
        return True, json.loads(s), None
    except Exception as e:
        return False, None, str(e)


def _normalize_verdict_json(js: Any) -> dict[str, Any]:
    """
    Make the model's JSON easier to consume by normalizing common schema violations.
    """
    if not isinstance(js, dict):
        return {"status": "UNCLEAR", "confidence": 0.0, "reason": "Invalid JSON schema (not an object).", "evidence": []}

    status = str(js.get("status", "UNCLEAR")).upper()
    if status not in {"OK", "ANOMALY", "UNCLEAR"}:
        status = "UNCLEAR"

    conf = js.get("confidence", 0.0)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    reason = js.get("reason", "")
    if isinstance(reason, list):
        reason = "; ".join(str(x) for x in reason if str(x).strip())
    elif not isinstance(reason, str):
        reason = str(reason)

    evidence = js.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        evidence = []
    evidence = [str(x) for x in evidence if str(x).strip()]

    return {"status": status, "confidence": conf, "reason": reason, "evidence": evidence}


def _normalize_plausibility(js: Any) -> dict[str, str]:
    if not isinstance(js, dict):
        return {"scenario_action": "unclear", "why": "Invalid JSON schema."}
    sa = str(js.get("scenario_action", "unclear")).lower().strip()
    if sa not in {"plausible", "implausible", "unclear"}:
        sa = "unclear"
    why = js.get("why", "")
    if not isinstance(why, str):
        why = str(why)
    return {"scenario_action": sa, "why": why.strip()}


def _fallback_plausibility_from_text(s: str) -> str:
    s = (s or "").strip().lower()
    if s.startswith("plausible"):
        return "plausible"
    if s.startswith("implausible"):
        return "implausible"
    if s.startswith("unclear"):
        return "unclear"
    return "unclear"


def _aggregate_json_verdict(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate per-frame JSON verdicts into a single decision.

    Uses `imu_context_json` when scenario/action are provided, otherwise uses `context_json`.

    Policy (simple + conservative):
    - If any frame says ANOMALY -> overall ANOMALY, confidence = max(conf) among ANOMALY frames
    - Else if any frame says UNCLEAR -> overall UNCLEAR
    - Else OK
    """
    per_frame: list[dict[str, Any]] = []

    for r in results:
        qa = r.get("qa", [])
        # Prefer imu_context_json if present, else fallback to context_json.
        item = next((x for x in qa if x.get("name") == "imu_context_json"), None)
        if item is None:
            item = next((x for x in qa if x.get("name") == "context_json"), None)

        if not item or not item.get("json_ok"):
            per_frame.append({"image": r.get("image"), "status": "UNCLEAR", "confidence": 0.0})
            continue

        js = item.get("json") or {}
        js_norm = _normalize_verdict_json(js)
        status = js_norm["status"]
        conf = js_norm["confidence"]

        per_frame.append({"image": r.get("image"), "status": status, "confidence": conf, "json": js_norm})

    statuses = [x["status"] for x in per_frame]
    if "ANOMALY" in statuses:
        overall = "ANOMALY"
        conf = max((x["confidence"] for x in per_frame if x["status"] == "ANOMALY"), default=0.0)
    elif "UNCLEAR" in statuses:
        overall = "UNCLEAR"
        conf = max((x["confidence"] for x in per_frame if x["status"] == "UNCLEAR"), default=0.0)
    else:
        overall = "OK"
        conf = max((x["confidence"] for x in per_frame), default=0.0)

    return {"overall_status": overall, "overall_confidence": conf, "per_frame": per_frame}


def _aggregate_labels_reasoning(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate per-frame `labels_reasoning` (plausible/implausible/unclear) into one label-level decision.

    Policy:
    - If any frame => implausible -> overall implausible
    - Else if any frame => unclear -> overall unclear
    - Else plausible
    """
    per_frame: list[dict[str, Any]] = []
    for r in results:
        lr = r.get("labels_reasoning") or {}
        sa = str(lr.get("scenario_action", "unclear")).lower().strip()
        if sa not in {"plausible", "implausible", "unclear"}:
            sa = "unclear"
        why = lr.get("why", "")
        if not isinstance(why, str):
            why = str(why)
        per_frame.append({"image": r.get("image"), "scenario_action": sa, "why": why.strip()})

    labels = [x["scenario_action"] for x in per_frame]
    if "implausible" in labels:
        overall = "implausible"
    elif "unclear" in labels:
        overall = "unclear"
    else:
        overall = "plausible"

    return {"overall_scenario_action": overall, "per_frame": per_frame}


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe Moondream2 prompt behavior on frames.")
    ap.add_argument("--model", default=DEFAULT_MODEL_ID, help="HuggingFace model id")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument(
        "--device-map",
        default="auto",
        help='HF device_map, e.g. "auto". If `accelerate` is not installed, this is ignored.',
    )
    ap.add_argument("--max-new-tokens", type=int, default=256)

    ap.add_argument("--image", type=str, default=None, help="Single image path")
    ap.add_argument("--image-dir", type=str, default=None, help="Directory to sample images from")
    ap.add_argument("--pattern", type=str, default="*.jpg", help="Glob pattern under --image-dir")
    ap.add_argument("--limit", type=int, default=3, help="How many images to run if using --image-dir")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--video", type=str, default=None, help="Optional: extract frames from a video file")
    ap.add_argument(
        "--seconds",
        type=str,
        default=None,
        help='Comma-separated seconds to extract from --video, e.g. "0,1,2.5,10"',
    )
    ap.add_argument(
        "--t-sec",
        type=float,
        default=None,
        help="If set with --video, defines the start of a window (in seconds).",
    )
    ap.add_argument(
        "--window-sec",
        type=float,
        default=1.0,
        help="Window size in seconds for --t-sec mode (default 1.0s).",
    )
    ap.add_argument(
        "--nframes",
        type=int,
        default=3,
        help="How many frames to sample inside the window for --t-sec mode (default 3).",
    )

    ap.add_argument("--scenario", type=str, default=None, help="Optional IMU-predicted scenario name")
    ap.add_argument("--action", type=str, default=None, help="Optional IMU-predicted action name")
    ap.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate per-frame JSON verdicts into a single decision (recommended for window mode).",
    )
    ap.add_argument(
        "--aggregate-only",
        action="store_true",
        help="When used with --aggregate, print only the aggregate JSON (suppresses per-frame outputs).",
    )

    args = ap.parse_args()

    if args.video:
        if args.t_sec is not None:
            if args.nframes <= 0:
                raise ValueError("--nframes must be >= 1")
            if args.window_sec <= 0:
                raise ValueError("--window-sec must be > 0")

            # Sample inside [t, t+window), avoiding exactly the endpoint.
            if args.nframes == 1:
                secs = [float(args.t_sec)]
            else:
                step = args.window_sec / args.nframes
                secs = [float(args.t_sec + i * step) for i in range(args.nframes)]

            image_paths = _extract_frames_ffmpeg(Path(args.video), secs)
        else:
            if not args.seconds:
                raise ValueError(
                    'When using --video, provide either --t-sec (window mode) or --seconds "0,1,2,3"'
                )
            secs = [float(x.strip()) for x in args.seconds.split(",") if x.strip()]
            image_paths = _extract_frames_ffmpeg(Path(args.video), secs)
    else:
        image_paths = _load_images(args)

    dtype = _get_dtype(args.dtype)
    if dtype is None:
        # Good default for small VLMs: fp16 on CUDA, fp32 on CPU.
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # `device_map` requires `accelerate`. If not installed, fall back to a simple `.to(device)`.
    try:
        import accelerate as _accelerate  # type: ignore  # noqa: F401

        accelerate_ok = True
    except Exception:
        accelerate_ok = False

    load_kwargs: dict[str, Any] = dict(
        trust_remote_code=True,
        dtype=dtype,
    )
    if accelerate_ok and args.device_map and args.device_map != "none":
        load_kwargs["device_map"] = args.device_map

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if not accelerate_ok:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    model.eval()

    questions = build_questions(args.scenario, args.action)

    all_results: list[dict[str, Any]] = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        per_q: list[dict[str, Any]] = []
        for item in questions:
            q = item["q"]
            # Moondream2 exposes `caption(image, length=...)` and `query(image=..., question=...)`.
            if item["name"] == "caption":
                out = model.caption(img, length="short")
                ans = out.get("caption", "")
            else:
                out = model.query(image=img, question=q)
                ans = out.get("answer", "")

            answer_str = str(ans).strip()
            rec: dict[str, Any] = {"name": item["name"], "question": q, "answer": answer_str}
            if item["name"].endswith("_json") or item["name"] == "context_json":
                ok, parsed, err = _parse_json_maybe(answer_str)
                rec["json_ok"] = ok
                if ok:
                    # Normalize expected schemas for easier downstream use.
                    if item["name"] == "imu_context_json":
                        rec["json"] = _normalize_verdict_json(parsed)
                    elif item["name"] == "labels_reasoning_json":
                        rec["json"] = _normalize_plausibility(parsed)
                    else:
                        rec["json"] = parsed
                else:
                    rec["json_error"] = err
            per_q.append(rec)

        # Attach a single "labels_reasoning" field that answers:
        # "Given [scenario, action], is that plausible in this frame?"
        labels_reasoning = None
        lr_item = next((x for x in per_q if x.get("name") == "labels_reasoning_json" and x.get("json_ok")), None)
        if lr_item and isinstance(lr_item.get("json"), dict):
            labels_reasoning = lr_item["json"]
        else:
            check_item = next((x for x in per_q if x.get("name") == "imu_context_check"), None)
            labels_reasoning = {
                "scenario_action": _fallback_plausibility_from_text((check_item or {}).get("answer", "")),
                "why": "Derived from imu_context_check (model did not return valid labels_reasoning_json).",
            }

        all_results.append(
            {
                "image": str(img_path),
                "scenario": args.scenario,
                "action": args.action,
                "qa": per_q,
                "labels_reasoning": labels_reasoning,
            }
        )

    output: dict[str, Any] = {"model": args.model, "results": all_results}
    if args.aggregate and len(all_results) >= 1:
        output["aggregate"] = {
            "verdict": _aggregate_json_verdict(all_results),
            "labels_reasoning": _aggregate_labels_reasoning(all_results),
        }

    if args.aggregate and args.aggregate_only:
        print(json.dumps(output.get("aggregate", {}), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


