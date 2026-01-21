#!/usr/bin/env python3
"""
Super-simple workflow router:
IMU anomaly -> (optional) VLM verification -> decision -> (optional) LLM call.

Design goal: tiny, readable, easy to replace with real components later.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional


Decision = str  # "shutdown_camera" | "ask_user" | "give_help"


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


def call_vlm_mock(image_path: str, event: ImuEvent) -> Dict[str, Any]:
    # Replace with Moondream2 / other VLM call. This keeps the router runnable anywhere.
    return {
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


def build_llm_prompt(event: ImuEvent, vlm_json: Dict[str, Any], route_json: Dict[str, Any]) -> str:
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
    return (
        "You are an assistive AI for smart glasses.\n"
        "Given IMU anomaly context and a VLM report, decide what to tell the user.\n"
        "Return a short, direct sentence to say out loud.\n\n"
        f"JSON:\n{json.dumps(payload, indent=2)}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to a keyframe image to verify.")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--action", required=True)
    ap.add_argument("--scenario-conf", type=float, default=0.0)
    ap.add_argument("--action-conf", type=float, default=0.0)
    ap.add_argument("--anomaly-reason", default="")

    ap.add_argument("--vlm-json", default="", help="If provided, skip VLM call and use this JSON string.")
    ap.add_argument("--use-mock-vlm", action="store_true", help="Use mock VLM output (default if no --vlm-json).")

    ap.add_argument("--llm-backend", choices=["none", "ollama"], default="none")
    ap.add_argument("--llm-model", default="qwen2.5:1.5b-instruct")
    args = ap.parse_args()

    event = ImuEvent(
        scenario_pred=args.scenario,
        action_pred=args.action,
        scenario_conf=args.scenario_conf,
        action_conf=args.action_conf,
        anomaly_reason=args.anomaly_reason,
    )

    if args.vlm_json:
        vlm = _safe_json_loads(args.vlm_json)
    else:
        if args.use_mock_vlm:
            vlm = call_vlm_mock(args.image, event)
        else:
            raise NotImplementedError(
                "Real VLM call not wired yet. Use --vlm-json to pass a JSON response or --use-mock-vlm."
            )

    r = route(event, vlm)
    out: Dict[str, Any] = {"decision": r["decision"], "why": r["why"], "vlm": vlm}

    if r["decision"] in {"ask_user", "give_help"} and args.llm_backend != "none":
        prompt = build_llm_prompt(event, vlm, r)
        if args.llm_backend == "ollama":
            out["llm_text"] = call_llm_ollama(prompt, args.llm_model)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


