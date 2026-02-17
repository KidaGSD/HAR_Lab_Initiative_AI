#!/usr/bin/env python3
"""
Print gold/(gold+bad) ratio from Label Studio annotations.
Fetches annotated tasks from HAR_dataset and counts Gold vs Bad choices.
"""

import argparse
import os
import requests


def _get_bearer_token(url, refresh_token):
    base = url.rstrip("/")
    r = requests.post(
        base + "/api/token/refresh/",
        json={"refresh": refresh_token},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access"]


def _ls_request(method, url, api_key, path, body=None, params=None):
    """Make Label Studio API request. Auto-refreshes PAT when Legacy disabled."""
    base = url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    r = requests.request(
        method, base + path,
        headers={**headers, "Authorization": "Token " + api_key},
        json=body, params=params, timeout=120,
    )
    if r.status_code == 401 and "legacy" in (r.text or ""):
        access = _get_bearer_token(url, api_key)
        r = requests.request(
            method, base + path,
            headers={**headers, "Authorization": "Bearer " + access},
            json=body, params=params, timeout=120,
        )
    r.raise_for_status()
    return r


def _extract_choices_from_annotation(ann):
    """Extract chosen labels (e.g. Gold, Bad) from a Label Studio annotation."""
    labels = []
    for item in ann.get("result", []):
        val = item.get("value", {})
        choices = val.get("choices", [])
        if isinstance(choices, list):
            labels.extend(c for c in choices if isinstance(c, str))
        elif isinstance(choices, str):
            labels.append(choices)
    return labels


def main():
    parser = argparse.ArgumentParser(description="Gold/(gold+bad) ratio from Label Studio HAR_dataset")
    parser.add_argument("--url", default=os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080"))
    parser.add_argument("--api-key", default=os.environ.get("LABEL_STUDIO_API_KEY"))
    parser.add_argument("--project", default="HAR_dataset")
    parser.add_argument("--project-id", type=int, default=None)
    args = parser.parse_args()

    if not args.api_key:
        print("Error: Set LABEL_STUDIO_API_KEY or pass --api-key")
        return 1

    # Get project ID
    r = _ls_request("GET", args.url, args.api_key, "/api/projects", params={"page_size": 100})
    data = r.json()
    projects = data.get("results", data) if isinstance(data, dict) else data
    if not isinstance(projects, list):
        projects = [projects] if projects else []
    match = next((p for p in projects if p.get("title") == args.project), None)
    if not match:
        print(f"Error: Project '{args.project}' not found.")
        return 1
    pid = match["id"]
    print(f"Project: {args.project} (id={pid})")

    # Fetch annotated tasks via /api/tasks/ with project filter
    gold, bad = 0, 0
    page = 1
    page_size = 250

    while True:
        params = {
            "project": pid,
            "only_annotated": "true",
            "fields": "all",
            "page": page,
            "page_size": page_size,
        }
        r = _ls_request("GET", args.url, args.api_key, "/api/tasks/", params=params)
        data = r.json()
        tasks = data.get("tasks", data) if isinstance(data, dict) else data
        if not isinstance(tasks, list):
            tasks = []
        if not tasks:
            break
        for task in tasks:
            for ann in task.get("annotations", []):
                labels = _extract_choices_from_annotation(ann)
                for lb in labels:
                    lb_lower = lb.strip().lower()
                    if lb_lower == "gold":
                        gold += 1
                    elif lb_lower == "bad":
                        bad += 1
        if len(tasks) < page_size:
            break
        page += 1

    total = gold + bad
    if total == 0:
        print("No Gold or Bad annotations found.")
        return 0
    ratio = gold / total
    print(f"\nGold: {gold} | Bad: {bad}")
    print(f"gold/(gold+bad) = {gold}/{total} = {ratio:.4f} ({ratio*100:.2f}%)")
    return 0


if __name__ == "__main__":
    exit(main())
