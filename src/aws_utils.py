from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import boto3


@dataclass(frozen=True)
class AwsCreds:
    access_key_id: str
    secret_access_key: str
    session_token: Optional[str] = None


def parse_access_key_file(path: str | Path) -> AwsCreds:
    """
    Parse a simple "Access ID / Access Key" file, optionally with a session token.

    Supported formats (case-insensitive; extra whitespace ok):
      Access ID: <id>
      Access Key: <secret>
      Session Token: <token>        (optional)

    This matches the format used by some Ego(Exo) tooling and is convenient for sharing creds
    without exporting env vars.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"credentials file not found: {p}")

    access_id = None
    access_secret = None
    session_token = None
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("access id"):
            access_id = line.split(":", 1)[1].strip()
        elif low.startswith("access key"):
            access_secret = line.split(":", 1)[1].strip()
        elif low.startswith("session token") or low.startswith("aws session token"):
            session_token = line.split(":", 1)[1].strip()

    if not access_id or not access_secret:
        raise ValueError(f"Failed to parse Access ID/Key from {p}")
    return AwsCreds(access_key_id=access_id, secret_access_key=access_secret, session_token=session_token)


def make_boto3_session(
    *,
    region: str = "us-west-1",
    profile: Optional[str] = None,
    creds_file: Optional[str | Path] = None,
    access_key_id: Optional[str] = None,
    secret_access_key: Optional[str] = None,
    session_token: Optional[str] = None,
) -> boto3.Session:
    """
    Create a boto3 Session with flexible credential sources.

    Priority:
      1) Explicit args (access_key_id/secret/session_token)
      2) creds_file (Access ID/Key[/Session Token])
      3) AWS_PROFILE / profile arg
      4) Default boto3 credential chain (env, ~/.aws/credentials, instance role, etc.)

    Notes:
      - Supports temporary credentials via session_token (common in newer Ego4D grants).
      - If you rely on env vars, boto3 already reads AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_PROFILE, etc.
    """
    if access_key_id and secret_access_key:
        return boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            aws_session_token=session_token,
            region_name=region,
        )

    if creds_file:
        creds = parse_access_key_file(creds_file)
        return boto3.Session(
            aws_access_key_id=creds.access_key_id,
            aws_secret_access_key=creds.secret_access_key,
            aws_session_token=creds.session_token,
            region_name=region,
        )

    # If profile is provided explicitly, prefer it over AWS_PROFILE env.
    if profile:
        return boto3.Session(profile_name=profile, region_name=region)

    # Fall back to default chain (may still use AWS_PROFILE env var).
    return boto3.Session(region_name=region)


def get_aws_identity(session: boto3.Session) -> Tuple[bool, str]:
    """
    Best-effort identity check used by download scripts.
    Returns (ok, message).
    """
    try:
        sts = session.client("sts")
        ident = sts.get_caller_identity()
        arn = ident.get("Arn") or str(ident)
        return True, arn
    except Exception as e:
        return False, str(e)



