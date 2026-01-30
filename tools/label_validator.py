#!/usr/bin/env python3
"""
Create HTML validator with remote video access via signed S3 URLs.
Usage: python scripts/create_label_validator.py --video-uid <uid> --output validator.html
"""

import pandas as pd
import argparse
from pathlib import Path
import json
import boto3
import os
from urllib.parse import quote
from datetime import timedelta
from botocore.exceptions import ClientError

from src.aws_utils import make_boto3_session


def get_s3_client(*, region: str = "us-west-1", profile: str | None = None, creds_file: str | None = None):
    """Initialize S3 client with flexible AWS credentials (env/profile/creds file)."""
    session = make_boto3_session(region=region, profile=profile, creds_file=creds_file)
    return session.client("s3")

def get_video_s3_path(video_uid, manifest_path=None):
    """
    Get S3 path for video. Tries ego4d.json first, then constructs path.
    """
    # Try ego4d.json (metadata file)
    if manifest_path and Path(manifest_path).exists():
        try:
            import json
            with open(manifest_path, 'r') as f:
                data = json.load(f)
            
            videos = data.get("videos", [])
            for video in videos:
                if video.get("video_uid") == video_uid:
                    s3_path = video.get("s3_path")
                    if s3_path:
                        print(f"✓ Found video path in ego4d.json: {s3_path}")
                        return s3_path
        except Exception as e:
            print(f"Warning: Could not read manifest as JSON: {e}")
            # Try as CSV manifest instead
            try:
                manifest_df = pd.read_csv(manifest_path)
                video_row = manifest_df[manifest_df['video_uid'] == video_uid]
                if len(video_row) > 0:
                    s3_path = video_row.iloc[0]['s3_path']
                    print(f"✓ Found video path in CSV manifest: {s3_path}")
                    return s3_path
            except Exception as e2:
                print(f"Warning: Could not read manifest as CSV either: {e2}")
    
    # Return None to indicate we should try all possible paths
    return None

def generate_signed_url(s3_path, expiration_hours=24):
    """
    Generate a presigned S3 URL for video streaming.
    Handles paths with or without file extensions.
    """
    s3_client = get_s3_client()
    
    # Parse S3 path
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_path}")
    
    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    base_key = parts[1]
    
    # First, check if we can access the bucket
    try:
        s3_client.head_bucket(Bucket=bucket)
        print(f"✓ Can access bucket: {bucket}")
    except Exception as e:
        print(f"ERROR: Cannot access bucket {bucket}: {e}")
        print("This might be a permissions issue. Check your AWS credentials.")
        return None
    
    # Try to list files in the directory to see what's actually there
    directory = "/".join(base_key.split("/")[:-1])  # Get directory path
    video_uid = base_key.split("/")[-1]  # Get video UID
    
    print(f"Searching in directory: s3://{bucket}/{directory}/")
    try:
        # List objects with this prefix
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=f"{directory}/{video_uid}",
            MaxKeys=10
        )
        
        if 'Contents' in response:
            print(f"Found {len(response['Contents'])} objects with prefix:")
            for obj in response['Contents']:
                print(f"  - {obj['Key']}")
        else:
            print(f"No objects found with prefix: {directory}/{video_uid}")
    except Exception as e:
        print(f"Warning: Could not list objects: {e}")
    
    # Check if path already has an extension
    has_extension = base_key.endswith(('.mp4', '.mov', '.mkv', '.avi'))
    
    # Build possible keys to try
    possible_keys = []
    
    if has_extension:
        # Path already has extension, use as-is
        possible_keys.append(base_key)
    else:
        # Path doesn't have extension, try common video extensions
        possible_keys.extend([
            f"{base_key}.mp4",
            f"{base_key}.mov",
            f"{base_key}.mkv",
            f"{base_key}.webm",
        ])
    
    # Find which key actually exists
    found_key = None
    for key in possible_keys:
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            found_key = key
            print(f"✓ Found video at: s3://{bucket}/{key}")
            break
        except s3_client.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '403':
                print(f"  Access denied for: s3://{bucket}/{key}")
            elif error_code == '404':
                # File doesn't exist, continue trying
                pass
            else:
                print(f"  Error checking {key}: {error_code}")
        except Exception as e:
            # Other error, continue trying
            pass
    
    if not found_key:
        print(f"\nERROR: Video not found in S3. Tried:")
        for key in possible_keys:
            print(f"  - s3://{bucket}/{key}")
        print(f"\nPossible issues:")
        print(f"  1. Video file doesn't exist at this path")
        print(f"  2. AWS credentials don't have access to bucket '{bucket}'")
        print(f"  3. Video might be in a different location")
        print(f"  4. Video might need to be downloaded via Ego4D CLI first")
        return None
    
    # Generate presigned URL
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': found_key},
            ExpiresIn=int(timedelta(hours=expiration_hours).total_seconds())
        )
        return url
    except Exception as e:
        print(f"ERROR: Could not generate signed URL: {e}")
        return None

def create_validator_html(video_uid, action_labels_path, scenario_labels_path, 
                         video_manifest_path=None, output_path="validator.html",
                         url_expiration_hours=24):
    """
    Create HTML page for validating labels with remote video access.
    """
    
    # Load labels
    print(f"Loading labels for video {video_uid}...")
    action_df = pd.read_csv(action_labels_path)
    scenario_df = pd.read_csv(scenario_labels_path)
    
    if args.video_manifest is None:
        # Try to find ego4d.json in common locations
        possible_paths = [
            "data/ego4d_data/ego4d.json",
            "data/ego4d_data/v2/ego4d.json",
            "data/ego4d_data/ego4d/v2/ego4d.json",
        ]
        for path in possible_paths:
            if Path(path).exists():
                args.video_manifest = path
                print(f"Using ego4d.json at: {path}")
                break

    # Filter for this video
    video_actions = action_df[action_df['video_uid'] == video_uid].sort_values('timestamp_sec')
    scenario_row = scenario_df[scenario_df['video_uid'] == video_uid]
    
    if len(video_actions) == 0:
        print(f"No action labels found for {video_uid}")
        return
    
    if len(scenario_row) == 0:
        print(f"No scenario label found for {video_uid}")
        return
    
    scenario = scenario_row.iloc[0]['scenario']

    
    # Skip S3 URL generation - we're using Ego4D visualization tool iframe
    print(f"Using Ego4D visualization tool for video playback")
    
    # Create label timeline (1-second intervals)
    max_time = int(video_actions['timestamp_sec'].max()) + 1
    timeline = []
    
    # Create label timeline (1-second intervals)
    max_time = int(video_actions['timestamp_sec'].max()) + 1
    timeline = []
    
    for t in range(max_time):
        # Find closest action label
        closest = video_actions.iloc[(video_actions['timestamp_sec'] - t).abs().argsort()[:1]]
        if len(closest) > 0 and abs(closest.iloc[0]['timestamp_sec'] - t) < 2.0:  # Within 2 seconds
            row = closest.iloc[0]
            timeline.append({
                'time': t,
                'action': row['action'],
                'narration': row.get('narration_text', ''),
                'reasoning': row.get('reasoning', '')
            })
        else:
            timeline.append({
                'time': t,
                'action': 'Unknown',
                'narration': '',
                'reasoning': ''
            })
    
    # Generate HTML
        # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Label Validator - {video_uid}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        .video-info {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 5px;
            flex-wrap: wrap;
        }}
        .info-item {{
            flex: 1;
            min-width: 150px;
        }}
        .info-label {{
            font-weight: bold;
            color: #666;
            font-size: 0.9em;
        }}
        .info-value {{
            font-size: 1.2em;
            color: #333;
            margin-top: 5px;
        }}
        .scenario-badge {{
            display: inline-block;
            padding: 8px 16px;
            background: #4CAF50;
            color: white;
            border-radius: 20px;
            font-weight: bold;
        }}
        .video-container {{
            position: relative;
            margin-bottom: 20px;
            background: #000;
            border-radius: 5px;
            overflow: hidden;
        }}
        iframe {{
            width: 100%;
            height: 600px;
            border: none;
        }}
                .labels-panel {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;  /* Reduced from 20px */
            margin-top: 10px;  /* Reduced from 20px */
        }}
        .label-box {{
            padding: 8px 12px;  /* Reduced from 20px */
            border-radius: 5px;
            border: 2px solid #ddd;
            transition: all 0.3s;
        }}
        .label-box.active {{
            border-color: #2196F3;
            background: #E3F2FD;
            box-shadow: 0 2px 4px rgba(33, 150, 243, 0.2);  /* Reduced shadow */
        }}
        .label-title {{
            font-size: 0.75em;  /* Reduced from 0.9em */
            color: #666;
            margin-bottom: 4px;  /* Reduced from 10px */
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .label-value {{
            font-size: 1.1em;  /* Reduced from 2em */
            font-weight: bold;
            color: #333;
            margin-bottom: 4px;  /* Reduced from 10px */
        }}
        .action-badge {{
            display: inline-block;
            padding: 4px 12px;  /* Reduced from 10px 20px */
            background: #2196F3;
            color: white;
            border-radius: 15px;  /* Reduced from 25px */
            font-size: 0.9em;  /* Reduced from 1.5em */
            font-weight: bold;
        }}
        .narration-box {{
            margin-top: 8px;  /* Reduced from 15px */
            padding: 8px;  /* Reduced from 15px */
            background: #fff;
            border-left: 3px solid #2196F3;  /* Reduced from 4px */
            border-radius: 3px;
        }}
        .narration-text {{
            font-style: italic;
            color: #555;
            margin-bottom: 4px;  /* Reduced from 10px */
            font-size: 0.85em;  /* Make narration text smaller */
        }}
        .reasoning-text {{
            font-size: 0.8em;  /* Reduced from 0.9em */
            color: #777;
        }}
        .timeline {{
            margin-top: 20px;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 5px;
            max-height: 300px;
            overflow-y: auto;
        }}
        .timeline-item {{
            padding: 8px;
            border-bottom: 1px solid #eee;
            font-size: 0.9em;
            cursor: pointer;
            transition: background 0.2s;
        }}
        .timeline-item:hover {{
            background: #f0f0f0;
        }}
        .timeline-item.current {{
            background: #E3F2FD;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Label Validator</h1>
        
        <div class="video-info">
            <div class="info-item">
                <div class="info-label">Video UID</div>
                <div class="info-value">{video_uid}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Scenario</div>
                <div class="info-value">
                    <span class="scenario-badge">{scenario}</span>
                </div>
            </div>
            <div class="info-item">
                <div class="info-label">Duration</div>
                <div class="info-value">{max_time} seconds</div>
            </div>
            <div class="info-item">
                <div class="info-label">Action Labels</div>
                <div class="info-value">{len(video_actions)} labels</div>
            </div>
        </div>
        
        <div class="video-container">
            <iframe 
                id="ego4dPlayer"
                src="https://visualize.ego4d-data.org/?video_uid={video_uid}"
                width="100%"
                height="600px"
                frameborder="0"
                allowfullscreen>
            </iframe>
            <div style="margin-top: 10px; padding: 10px; background: #fff3cd; border-radius: 5px;">
                <strong>Note:</strong> Video loaded from Ego4D visualization tool.
                <a href="https://visualize.ego4d-data.org/?video_uid={video_uid}" target="_blank">Open in new tab</a>
            </div>
        </div>
        
        
        <div class="labels-panel">
            <div class="label-box active" id="scenarioBox">
                <div class="label-title">Scenario</div>
                <div class="label-value">{scenario}</div>
            </div>
            
            <div class="label-box active" id="actionBox">
                <div class="label-title">Current Action</div>
                <div class="label-value">
                    <span class="action-badge" id="actionLabel">Loading...</span>
                </div>
                <div class="narration-box" id="narrationBox" style="display: none;">
                    <div class="narration-text" id="narrationText"></div>
                    <div class="reasoning-text" id="reasoningText"></div>
                </div>
            </div>
        </div>
        
        <div class="timeline">
            <h3>Label Timeline (Click to jump to time)</h3>
            <div id="timelineContent"></div>
        </div>
    </div>
    
    <script>
        // Label timeline data
        const timeline = {json.dumps(timeline)};
        
        const actionLabel = document.getElementById('actionLabel');
        const narrationBox = document.getElementById('narrationBox');
        const narrationText = document.getElementById('narrationText');
        const reasoningText = document.getElementById('reasoningText');
        const timelineContent = document.getElementById('timelineContent');
        
        // Action color mapping
        const actionColors = {{
            'Locomotion': '#4CAF50',
            'Manual Work': '#FF9800',
            'Essential Operation': '#9C27B0',
            'Object Transfer': '#2196F3',
            'Search': '#F44336',
            'Stationary': '#9E9E9E',
            'Error / Correction': '#E91E63',
            'Unknown': '#CCCCCC'
        }};
        
        // Render timeline
        timeline.forEach(item => {{
            const div = document.createElement('div');
            div.className = 'timeline-item';
            div.id = `timeline-${{item.time}}`;
            div.innerHTML = `
                <strong>${{item.time}}s</strong>: ${{item.action}}
                ${{item.narration ? '<br><em>' + item.narration + '</em>' : ''}}
            `;
            div.onclick = () => {{
                updateLabels(item.time);
            }};
            timelineContent.appendChild(div);
        }});
        
        function updateLabels(time) {{
            const currentTime = time || 0;
            
            // Find label for current time
            const label = timeline.find(l => l.time === currentTime) || timeline[0];
            
            if (label) {{
                // Update action label
                actionLabel.textContent = label.action;
                actionLabel.style.background = actionColors[label.action] || '#CCCCCC';
                
                // Update narration
                if (label.narration) {{
                    narrationBox.style.display = 'block';
                    narrationText.textContent = label.narration;
                    reasoningText.textContent = label.reasoning || '';
                }} else {{
                    narrationBox.style.display = 'none';
                }}
                
                // Highlight timeline item
                document.querySelectorAll('.timeline-item').forEach(el => {{
                    el.classList.remove('current');
                }});
                const timelineEl = document.getElementById(`timeline-${{currentTime}}`);
                if (timelineEl) {{
                    timelineEl.classList.add('current');
                    timelineEl.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                }}
            }}
        }}
        
        
        
        // Initial update
        // Initial update (show first label)
        if (timeline.length > 0) {{
            updateLabels(timeline[0].time);
        }}
    </script>
</body>
</html>
"""
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Validator HTML created: {output_path}")
    print(f"  Open in browser: file://{Path(output_path).absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create label validator HTML with remote video access")
    parser.add_argument("--video-uid", type=str, required=True, help="Video UID to visualize")
    parser.add_argument("--action-labels", type=str, 
                       default="data/labels/action_labels_llm_validated.csv",
                       help="Path to action labels CSV")
    parser.add_argument("--scenario-labels", type=str,
                       default="data/labels/scenario_labels.csv",
                       help="Path to scenario labels CSV")
    parser.add_argument("--video-manifest", type=str, default=None,
                       help="Path to video manifest CSV (optional, will construct path if not provided)")
    parser.add_argument("--output", type=str, default="validator.html",
                       help="Output HTML file path")
    parser.add_argument("--url-expiration", type=int, default=24,
                       help="Signed URL expiration time in hours (default: 24)")
    
    args = parser.parse_args()
    
    # Verify AWS credentials (best-effort). Credentials may come from:
    # - env vars: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY [/ AWS_SESSION_TOKEN]
    # - profiles: AWS_PROFILE / ~/.aws/credentials
    # - instance role / workload identity
    if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")) and not os.environ.get(
        "AWS_PROFILE"
    ):
        print("NOTE: No explicit AWS env vars/profile detected; relying on boto3 default credential chain.")
    
    create_validator_html(
        args.video_uid,
        args.action_labels,
        args.scenario_labels,
        args.video_manifest,
        args.output,
        args.url_expiration
    )