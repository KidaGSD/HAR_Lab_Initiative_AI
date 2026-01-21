#!/usr/bin/env python3
"""
Super-simple workflow router:
Input image + IMU predictions + anomaly_reason -> VLM JSON -> routing -> (optional) LLM chat.

Design goal: tiny, readable, easy to replace with real components later.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple


Decision = str  # "shutdown_camera" | "ask_user" | "give_help"
CameraState = Literal["OFF", "ON"]


@dataclass
class ImuEvent:
    scenario_pred: str
    action_pred: str
    scenario_conf: float = 0.0
    action_conf: float = 0.0
    anomaly_reason: str = ""


def _safe_json_loads(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {"raw": s}


def _extract_json_object(s: str) -> Optional[str]:
    """
    Best-effort extraction of a JSON object from a model answer that may contain extra text.
    """
    if not s:
        return None
    i = s.find("{")
    j = s.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    return s[i : j + 1]


def _print_block(tag: str, text: str) -> None:
    bar = "=" * max(10, len(tag))
    print(f"\n[{tag}]\n{bar}\n{text}\n{bar}\n", flush=True)


def _normalize_vlm_json(d: Dict[str, Any], raw_answer: str = "") -> Dict[str, Any]:
    """
    Enforce a minimal schema so routing is stable even if the VLM deviates.
    """
    out = dict(d)
    out.setdefault("image_caption", "")
    out.setdefault("labels_plausibility", "unclear")
    out.setdefault("labels_reasoning", "")
    out.setdefault("scene_summary", "")
    out.setdefault("risk_level", "unclear")
    out.setdefault("risk_reason", "")
    if raw_answer and "raw_answer" not in out:
        out["raw_answer"] = raw_answer
    return out


def call_vlm_moondream2(
    image_path: str,
    event: ImuEvent,
    model_id: str = "vikhyatk/moondream2",
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Single-image Moondream2 call returning a *structured JSON* verdict.
    """
    from PIL import Image  # lazy import
    import torch  # lazy import
    from transformers import AutoModelForCausalLM  # lazy import

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, dtype=dtype).to(device).eval()

    img = Image.open(image_path).convert("RGB")
    # Short caption acts as a compact "anchor" for downstream LLM chat.
    try:
        cap_out = model.caption(img, length="short")
        image_caption = str(cap_out.get("caption", "")).strip()
    except Exception:
        image_caption = ""

    prompt = (
        "You are a safety assistant verifying IMU predictions from smart glasses.\n"
        "Return ONLY valid JSON. No markdown.\n"
        "Required keys:\n"
        '  - "image_caption": a single short caption of the image\n'
        '  - "labels_plausibility": one of ["plausible","implausible","unclear"]\n'
        '  - "labels_reasoning": short reason\n'
        '  - "scene_summary": 1-2 sentences describing what you see\n'
        '  - "risk_level": one of ["none","low","medium","high","unclear"]\n'
        '  - "risk_reason": short reason\n\n'
        f'IMU scenario_pred="{event.scenario_pred}" (conf={event.scenario_conf:.3f})\n'
        f'IMU action_pred="{event.action_pred}" (conf={event.action_conf:.3f})\n'
        f'IMU anomaly_reason="{event.anomaly_reason}"\n\n'
        "Task:\n"
        f"0) Here is a suggested caption you can reuse if accurate: {json.dumps(image_caption)}\n"
        "1) Are the IMU labels plausible in this image?\n"
        "2) Assess any safety risk in the scene.\n"
    )
    if debug:
        _print_block("VLM_PROMPT", prompt)
    out = model.query(image=img, question=prompt)
    ans = str(out.get("answer", out)).strip()
    if debug:
        _print_block("VLM_RAW_OUTPUT", ans)

    parsed = _safe_json_loads(ans)
    if "raw" in parsed:
        maybe = _extract_json_object(ans)
        if maybe:
            parsed = _safe_json_loads(maybe)

    normalized = _normalize_vlm_json(parsed, raw_answer=ans)
    # Ensure caption is always available to the LLM even if JSON parsing fails.
    if image_caption and not normalized.get("image_caption"):
        normalized["image_caption"] = image_caption
    if debug:
        _print_block("VLM_PARSED_JSON", json.dumps(normalized, indent=2))
    return normalized


def call_vlm_mock(image_path: str, event: ImuEvent) -> Dict[str, Any]:
    # Replace with Moondream2 / other VLM call. This keeps the router runnable anywhere.
    return {
        "image_caption": "mock_vlm: caption not implemented",
        "labels_plausibility": "unclear",
        "labels_reasoning": "mock_vlm: not implemented",
        "scene_summary": "mock_vlm: no scene summary",
        "risk_level": "unclear",
        "risk_reason": "mock_vlm: no risk assessment",
        "image": image_path,
        "scenario_pred": event.scenario_pred,
        "action_pred": event.action_pred,
        "anomaly_reason": event.anomaly_reason,
    }


def call_llm_ollama(prompt: str, model: str) -> str:
    """
    Requires `ollama` installed + model pulled locally.
    Example model (light): qwen2.5:1.5b-instruct (name depends on your Ollama registry).
    """
    p = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError(f"ollama failed: {p.stderr.decode('utf-8', errors='ignore')}")
    return p.stdout.decode("utf-8", errors="ignore").strip()


def route(event: ImuEvent, vlm_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal routing rules:
    - implausible => likely FP => shutdown unless risk is medium/high/unclear => ask_user
    - plausible => check risk => shutdown/ask_user/give_help
    - unclear => ask_user (conservative)
    """
    plaus = str(vlm_json.get("labels_plausibility", "unclear")).lower()
    risk = str(vlm_json.get("risk_level", "unclear")).lower()

    if plaus == "implausible":
        if risk in {"high", "medium", "unclear"}:
            return {"decision": "ask_user", "why": "VLM says labels implausible, but risk is not clearly none"}
        return {"decision": "shutdown_camera", "why": "Likely IMU false positive; VLM sees no risk"}

    if plaus == "plausible":
        if risk == "high":
            return {"decision": "give_help", "why": "VLM sees high risk"}
        if risk in {"medium", "unclear"}:
            return {"decision": "ask_user", "why": "VLM uncertain/medium risk"}
        return {"decision": "shutdown_camera", "why": "VLM sees plausible labels and no risk"}

    # plaus == unclear or anything else
    return {"decision": "ask_user", "why": "VLM uncertain about label plausibility"}


def build_llm_prompt(
    event: ImuEvent,
    vlm_json: Dict[str, Any],
    route_json: Dict[str, Any],
    chat_history: List[Dict[str, str]],
) -> str:
    payload = {
        "imu": {
            "scenario_pred": event.scenario_pred,
            "action_pred": event.action_pred,
            "scenario_conf": event.scenario_conf,
            "action_conf": event.action_conf,
            "anomaly_reason": event.anomaly_reason,
        },
        "vlm": vlm_json,
        "router": route_json,
    }
    # Note: we embed history as plain text to keep this dependency-free + backend-agnostic.
    history_txt = "\n".join([f'{m["role"]}: {m["content"]}' for m in chat_history])
    return (
        "SYSTEM:\n"
        "You are an assistive AI for smart glasses.\n"
        "You have access to a camera snapshot summary and IMU anomaly context.\n"
        "Stay grounded in the provided VLM fields. Do not mention probabilities unless asked.\n"
        "Be concise, calm, and practical. If you need info, ask ONE short question.\n"
        "CONTEXT_JSON:\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        "CHAT_HISTORY:\n"
        f"{history_txt}\n"
    )


def _print_camera(state: CameraState, reason: str = "") -> None:
    msg = f"[CAMERA] state={state}"
    if reason:
        msg += f" | {reason}"
    print(msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to a keyframe image to verify.")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--action", required=True)
    ap.add_argument("--scenario-conf", type=float, default=0.0)
    ap.add_argument("--action-conf", type=float, default=0.0)
    ap.add_argument("--anomaly-reason", default="")

    ap.add_argument("--vlm-backend", choices=["moondream2", "mock"], default="moondream2")
    ap.add_argument("--vlm-model", default="vikhyatk/moondream2")
    ap.add_argument("--vlm-json", default="", help="If provided, skip VLM call and use this JSON string.")
    ap.add_argument("--debug", action="store_true", help="Print VLM prompt + VLM output + parsed JSON (and LLM prompt).")

    ap.add_argument("--llm-backend", choices=["none", "ollama"], default="none")
    ap.add_argument("--llm-model", default="qwen2.5:1.5b-instruct")
    ap.add_argument(
        "--interactive",
        action="store_true",
        help="If LLM is used, enter a terminal chat loop (type /shutdown to stop camera).",
    )
    args = ap.parse_args()

    event = ImuEvent(
        scenario_pred=args.scenario,
        action_pred=args.action,
        scenario_conf=args.scenario_conf,
        action_conf=args.action_conf,
        anomaly_reason=args.anomaly_reason,
    )

    camera: CameraState = "OFF"
    _print_camera(camera, "idle")

    # This script assumes the IMU anomaly logic already triggered; we turn on the camera for VLM verification.
    camera = "ON"
    _print_camera(camera, f"activated due to anomaly_reason='{event.anomaly_reason}'")

    if args.vlm_json:
        vlm = _normalize_vlm_json(_safe_json_loads(args.vlm_json))
    else:
        if args.vlm_backend == "mock":
            vlm = call_vlm_mock(args.image, event)
        else:
            vlm = call_vlm_moondream2(args.image, event, model_id=args.vlm_model, debug=args.debug)

    r = route(event, vlm)
    out: Dict[str, Any] = {"decision": r["decision"], "why": r["why"], "camera_state": camera, "vlm": vlm}

    if r["decision"] == "shutdown_camera":
        camera = "OFF"
        _print_camera(camera, "shutdown (no issue detected)")
        out["camera_state"] = camera
        print(json.dumps(out, indent=2))
        return

    # ask_user / give_help: keep camera on unless user shuts it down.
    _print_camera(camera, f"keeping ON (decision={r['decision']})")

    if args.llm_backend != "none":
        chat: List[Dict[str, str]] = []
        # Kick off with a grounded "first turn" so the assistant starts in-context.
        chat.append(
            {
                "role": "user",
                "content": (
                    "Start the conversation.\n"
                    "1) Briefly explain what you see (from the camera) and whether there is an issue.\n"
                    "2) If needed, ask ONE clarifying question.\n"
                ),
            }
        )
        prompt = build_llm_prompt(event, vlm, r, chat)
        if args.debug:
            _print_block("LLM_PROMPT", prompt)
        if args.llm_backend == "ollama":
            first = call_llm_ollama(prompt, args.llm_model)
            out["llm_text"] = first
            print(f"[LLM] {first}", flush=True)
            chat.append({"role": "assistant", "content": first})

        if args.interactive:
            print("[CHAT] Type your reply. Commands: /shutdown, /exit", flush=True)
            while True:
                _print_camera(camera, "chat_active")
                try:
                    user_msg = input("you> ").strip()
                except EOFError:
                    user_msg = "/exit"
                if not user_msg:
                    continue
                if user_msg in {"/shutdown", "/stop"}:
                    camera = "OFF"
                    _print_camera(camera, "user requested shutdown")
                    out["camera_state"] = camera
                    break
                if user_msg in {"/exit", "/quit"}:
                    _print_camera(camera, "exiting chat (camera unchanged)")
                    out["camera_state"] = camera
                    break

                chat.append({"role": "user", "content": user_msg})
                prompt = build_llm_prompt(event, vlm, r, chat)
                if args.debug:
                    _print_block("LLM_PROMPT", prompt)
                if args.llm_backend == "ollama":
                    ans = call_llm_ollama(prompt, args.llm_model)
                    chat.append({"role": "assistant", "content": ans})
                    print(f"assistant> {ans}", flush=True)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


