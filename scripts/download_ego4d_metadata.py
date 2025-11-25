import argparse
import json
import pandas as pd
from pathlib import Path
import subprocess
import sys

def install_ego4d():
    try:
        import ego4d
    except ImportError:
        print("ego4d package not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ego4d"])

def download_metadata(output_dir):
    """
    Downloads ego4d.json and narrations.json using the Ego4D CLI.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading metadata to {output_dir}...")
    
    # Download ego4d.json
    cmd_ego4d = [
        "ego4d", "--output_directory", str(output_dir), "--datasets", "ego4d_json", "--benchmarks", "moments", "--yes"
    ]
    # Note: 'moments' benchmark usually includes the main metadata. 
    # Alternatively, we can just try to download the specific files if we knew the S3 paths, 
    # but the CLI is the standard way.
    # However, the CLI might download a lot. 
    # Let's try to use the CLI to download specific metadata if possible.
    # The 'ego4d' CLI has a 'download' command.
    
    # Actually, for metadata, it's often just 'ego4d --output_directory <dir> --datasets ego4d.json'
    # Let's try the standard command.
    
    # Check if file already exists
    ego4d_json_path = output_dir / "ego4d.json"
    if not ego4d_json_path.exists():
        # Try to find it in subdirs
        found = list(output_dir.rglob("ego4d.json"))
        if found:
            ego4d_json_path = found[0]

    try:
        subprocess.check_call(["ego4d", "--output_directory", str(output_dir), "--datasets", "metadata", "--yes"])
    except subprocess.CalledProcessError:
        print("Failed to download metadata via CLI.")
        if ego4d_json_path.exists():
            print(f"Found existing metadata at {ego4d_json_path}. Proceeding...")
            return True
        else:
            print("No local metadata found. Cannot proceed.")
            return False
        
    return True

def filter_metadata(metadata_dir, output_file, scenarios=None, keywords=None):
    metadata_dir = Path(metadata_dir)
    ego4d_json_path = metadata_dir / "ego4d.json"
    
    if not ego4d_json_path.exists():
        # It might be in a subdirectory depending on how the CLI downloads it
        # usually it's in <output_dir>/ego4d/v2/ego4d.json or similar
        found = list(metadata_dir.rglob("ego4d.json"))
        if found:
            ego4d_json_path = found[0]
        else:
            print(f"Error: ego4d.json not found in {metadata_dir}")
            return

    print(f"Loading {ego4d_json_path}...")
    with open(ego4d_json_path, 'r') as f:
        ego4d_data = json.load(f)
    
    videos = ego4d_data.get("videos", [])
    print(f"Total videos in metadata: {len(videos)}")
    
    filtered_videos = []
    
    # Filter by scenario
    if scenarios:
        scenarios_lower = [s.lower() for s in scenarios]
        for v in videos:
            v_scenarios = v.get("scenarios", [])
            if any(s.lower() in [vs.lower() for vs in v_scenarios] for s in scenarios_lower):
                filtered_videos.append(v)
    else:
        filtered_videos = videos
        
    print(f"Videos after scenario filtering: {len(filtered_videos)}")
    
    # Filter by narration keywords (if we had narrations.json downloaded and linked)
    # For now, we will just output the videos that match the scenario.
    # We can add narration filtering later if we download narrations.json.
    
    # Create DataFrame
    data = []
    for v in filtered_videos:
        data.append({
            "video_uid": v["video_uid"],
            "scenarios": str(v.get("scenarios", [])),
            "duration_sec": v.get("duration_sec", 0)
        })
        
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Saved filtered UIDs to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Download and filter Ego4D metadata")
    parser.add_argument("--output-dir", type=str, default="data/ego4d_meta", help="Directory to save metadata")
    parser.add_argument("--target-uids-file", type=str, default="target_uids.csv", help="Output CSV file for filtered UIDs")
    parser.add_argument("--scenarios", nargs="+", default=["Cooking", "Carpenter", "Gardening", "Bike"], help="List of scenarios to filter by")
    
    args = parser.parse_args()
    
    # Ensure ego4d is installed
    install_ego4d()
    
    # Download metadata
    if download_metadata(args.output_dir):
        # Filter
        filter_metadata(args.output_dir, args.target_uids_file, args.scenarios)

if __name__ == "__main__":
    main()
