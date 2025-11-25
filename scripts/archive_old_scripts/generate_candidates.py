import torch
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
from train_ego4d_model import SSLModel, DeepSVDD, CONFIG, Ego4DDataset

def generate_candidates(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load UIDs
    df = pd.read_csv(args.target_uids_file)
    uids = df['video_uid'].tolist()
    
    # Load Dataset
    ds = Ego4DDataset(uids, args.processed_dir)
    loader = torch.utils.data.DataLoader(ds, batch_size=CONFIG['ssl']['batch_size'], shuffle=False)
    
    # Load Models
    model = SSLModel(CONFIG).to(device)
    model.load_state_dict(torch.load(Path(args.model_dir) / "best_model.pth", map_location=device))
    model.eval()
    
    svdd_state = torch.load(Path(args.model_dir) / "svdd_model.pth", map_location=device)
    svdd = DeepSVDD()
    svdd.center = svdd_state['center']
    svdd.R = svdd_state['R']
    
    print(f"Loaded SVDD Model. Center: {svdd.center[:5]}..., R: {svdd.R}")
    
    # Inference
    results = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Scoring Windows"):
            traj = batch['traj'].to(device)
            batch_input = {'traj': traj}
            
            # Get Embedding
            emb, _ = model(batch_input)
            emb_np = emb.cpu().numpy()
            
            # Get Anomaly Score
            scores = svdd.score(emb_np)
            
            # Store results
            # We need to map back to UID and Window Index
            # Since loader is sequential and not shuffled, we can just iterate
            # But batch['take_uid'] is a list of strings, batch['window_idx'] is tensor
            
            take_uids = batch['take_uid']
            window_idxs = batch['window_idx'].cpu().numpy()
            
            for i in range(len(scores)):
                results.append({
                    'video_uid': take_uids[i],
                    'window_idx': window_idxs[i],
                    'svdd_score': scores[i],
                    'is_candidate': scores[i] > args.threshold
                })
    
    # Save Results
    results_df = pd.read_json(pd.DataFrame(results).to_json()) # Handle mixed types if any
    results_df = pd.DataFrame(results)
    
    output_path = Path(args.output_dir) / "candidates.csv"
    results_df.to_csv(output_path, index=False)
    
    n_candidates = results_df['is_candidate'].sum()
    print(f"\nProcessed {len(results_df)} windows.")
    print(f"Found {n_candidates} candidates (Score > {args.threshold})")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-uids-file", type=str, required=True)
    parser.add_argument("--processed-dir", type=str, default="data/processed_ego4d")
    parser.add_argument("--model-dir", type=str, default="models/ego4d_ssl")
    parser.add_argument("--output-dir", type=str, default="data/candidates")
    parser.add_argument("--threshold", type=float, default=1.0, help="SVDD score threshold for candidacy")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    generate_candidates(args)
