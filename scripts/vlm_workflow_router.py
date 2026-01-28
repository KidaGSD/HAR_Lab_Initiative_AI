#!/usr/bin/env python3
"""
Super-simple workflow router:
Input image + IMU predictions + anomaly_reason -> VLM JSON -> routing -> (optional) LLM chat.

Design goal: tiny, readable, easy to replace with real components later.
"""

from __future__ import annotations

import argparse
import json
import re
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


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


_USER_DONE_PAT = re.compile(
    r"\b("
    r"thank(s| you)?|thx|ty|"
    r"got it|"
    r"ok(ay)?( thanks)?|"
    r"that's all|that is all|nothing else|no more|"
    r"bye|goodbye|see you"
    r")\b",
    flags=re.IGNORECASE,
)

_USER_NEGATIVE_PAT = re.compile(
    r"^\s*(no|nope|nah|nothing|nothing else|that's all|that is all|all good|done)\s*[.!]?\s*$",
    flags=re.IGNORECASE,
)

_ASSISTANT_CLOSING_Q_PAT = re.compile(
    r"\b(anything else|any other (questions|things)|is that all|is there anything else|"
    r"can i help with anything else)\b",
    flags=re.IGNORECASE,
)


def _user_signals_done(user_msg: str) -> bool:
    """
    Conservative: only triggers on very explicit "done" language.
    """
    return bool(_USER_DONE_PAT.search(user_msg or ""))


def _user_says_no_to_closing(user_msg: str) -> bool:
    return bool(_USER_NEGATIVE_PAT.match(user_msg or ""))


def _assistant_asked_closing_q(assistant_msg: str) -> bool:
    return bool(_ASSISTANT_CLOSING_Q_PAT.search(assistant_msg or ""))


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


def _load_moondream2(model_id: str):
    import torch  # lazy import
    from transformers import AutoModelForCausalLM  # lazy import

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    return AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, dtype=dtype).to(device).eval()


def call_vlm_moondream2(model: Any, img: Any, event: ImuEvent, debug: bool = False) -> Dict[str, Any]:
    """
    Single-image Moondream2 call returning a *structured JSON* verdict.
    """
    # Short caption acts as a compact "anchor" for downstream chat.
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
    # Ensure caption is always available to downstream chat even if JSON parsing fails.
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


def build_vlm_chat_prompt(
    event: ImuEvent,
    vlm_json: Dict[str, Any],
    route_json: Dict[str, Any],
    chat_history: List[Dict[str, str]],
) -> str:
    """
    Keep this prompt SHORT. Moondream2 does best with compact, concrete instructions.
    We include only a compact context summary + a short chat history.
    """
    """
    NOTE:
    For Moondream2, shorter is better. Large "system prompts" and multi-turn transcripts
    often cause it to re-caption instead of answering the actual question.

    We therefore build a *minimal* prompt focused on the last user message only.
    """
    image_caption = str(vlm_json.get("image_caption", "")).strip()
    last_user = ""
    for m in reversed(chat_history):
        if m.get("role") == "user":
            last_user = str(m.get("content", "")).strip()
            break

    # Minimal question-focused prompt.
    return (
        "Answer the user's question about the image in 1-2 sentences. Plain English only.\n"
        "If the user asks where something is, use relative directions (left/right/center, near/far).\n"
        "If you cannot locate it, say so and ask ONE brief follow-up question.\n\n"
        "If you think the user is satisfied, you may end with: \"Anything else I can help with?\"\n\n"
        f"Caption hint: {image_caption}\n"
        f"User question: {last_user}\n"
    )


def vlm_chat_turn_moondream2(model: Any, img: Any, prompt: str, debug: bool = False) -> str:
    if debug:
        _print_block("VLM_CHAT_PROMPT", prompt)
    out = model.query(image=img, question=prompt)
    ans = str(out.get("answer", out)).strip()
    if debug:
        _print_block("VLM_CHAT_OUTPUT", ans)
    return ans


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
    ap.add_argument("--debug", action="store_true", help="Print VLM prompt + VLM output + parsed JSON + chat prompts.")
    ap.add_argument(
        "--history-turns",
        type=int,
        default=6,
        help="How many chat messages to include in the prompt (kept small for better VLM focus).",
    )
    ap.add_argument(
        "--interactive",
        action="store_true",
        help="Enter a terminal chat loop with the VLM (type /shutdown to stop camera).",
    )
    ap.add_argument(
        "--no-auto-shutdown",
        action="store_true",
        help="Disable automatic camera shutdown when the conversation appears finished (e.g., user says thanks).",
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

    model = None
    img = None
    if args.vlm_json:
        vlm = _normalize_vlm_json(_safe_json_loads(args.vlm_json))
    else:
        if args.vlm_backend == "mock":
            vlm = call_vlm_mock(args.image, event)
        else:
            from PIL import Image  # lazy import

            model = _load_moondream2(args.vlm_model)
            img = Image.open(args.image).convert("RGB")
            vlm = call_vlm_moondream2(model, img, event, debug=args.debug)

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

    if args.interactive:
        chat: List[Dict[str, str]] = []
        auto_shutdown = not args.no_auto_shutdown
        assistant_recently_asked_closing = False

        if args.vlm_backend == "mock":
            first = "mock_vlm: I cannot see the image (mock), but the system detected a potential anomaly. What do you need?"
        else:
            assert model is not None and img is not None
            # First turn: ask a simple, concrete question (avoid multi-step instructions).
            first_q = "Describe the image in 1-2 sentences and mention any obvious safety risk."
            first = vlm_chat_turn_moondream2(model, img, first_q, debug=args.debug)
        print(f"assistant> {first}", flush=True)
        out["assistant_text"] = first
        chat.append({"role": "assistant", "content": first})
        assistant_recently_asked_closing = _assistant_asked_closing_q(first)

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

            if auto_shutdown:
                # If the assistant just asked "anything else?" and the user said "no", shut down.
                if assistant_recently_asked_closing and _user_says_no_to_closing(user_msg):
                    camera = "OFF"
                    _print_camera(camera, "auto-shutdown (user indicated conversation finished)")
                    out["camera_state"] = camera
                    chat.append({"role": "user", "content": user_msg})
                    break
                # If the user explicitly signals they're done (thanks/bye/etc.), shut down.
                if _user_signals_done(user_msg):
                    camera = "OFF"
                    _print_camera(camera, "auto-shutdown (user said thanks/bye/done)")
                    out["camera_state"] = camera
                    chat.append({"role": "user", "content": user_msg})
                    break

            chat.append({"role": "user", "content": user_msg})
            if args.vlm_backend == "mock":
                ans = "mock_vlm: (no image) I can't answer visually in mock mode."
            else:
                assert model is not None and img is not None
                # Moondream2 works best with short, concrete prompts; use only the last user message.
                prompt = build_vlm_chat_prompt(event, vlm, r, chat[-args.history_turns :])
                ans = vlm_chat_turn_moondream2(model, img, prompt, debug=args.debug)
            chat.append({"role": "assistant", "content": ans})
            print(f"assistant> {ans}", flush=True)
            assistant_recently_asked_closing = _assistant_asked_closing_q(ans)

        # Keep the full transcript in the final JSON for debugging / downstream logging.
        out["chat_history"] = chat

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


