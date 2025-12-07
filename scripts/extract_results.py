#!/usr/bin/env python3
"""
Extract and compare all experiment results from checkpoint logs.
"""

import os
import re
import pandas as pd
from pathlib import Path

def parse_probe_log(log_path):
    """Extract best Action F1 from probe training log."""
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Find all "Probe Val Action F1" entries
    pattern = r'Probe Val Action F1: ([\d.]+)'
    matches = re.findall(pattern, content)
    
    if matches:
        f1s = [float(m) for m in matches]
        return max(f1s)
    return None

def parse_training_log(log_path):
    """Extract final Scenario F1 and Action F1 from training log."""
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    scenario_f1s = []
    action_f1s = []
    
    for line in lines:
        # Scenario F1
        match = re.search(r'Val Scenario F1: ([\d.]+)', line)
        if match:
            scenario_f1s.append(float(match.group(1)))
        
        # Action F1
        match = re.search(r'Val Action F1: ([\d.]+)', line)
        if match:
            action_f1s.append(float(match.group(1)))
    
    best_scenario = max(scenario_f1s) if scenario_f1s else None
    best_action = max(action_f1s) if action_f1s else None
    
    return best_scenario, best_action

def main():
    base_dir = Path("checkpoints/checkpoints")
    
    results = []
    
    # Check all backbone runs
    for run_dir in base_dir.glob("backbone_*"):
        log_dir = run_dir / "logs"
        
        if not log_dir.exists():
            continue
            
        print(f"\nAnalyzing {run_dir.name}...")
        
        # β=0.0 results
        beta0_log = log_dir / "beta0.log"
        if beta0_log.exists():
            scenario_f1, _ = parse_training_log(beta0_log)
            if scenario_f1:
                results.append({
                    'Run': run_dir.name,
                    'Model': 'β=0.0 Backbone',
                    'Scenario F1': f"{scenario_f1:.4f}",
                    'Action F1': 'N/A',
                    'Notes': 'Scenario-only training'
                })
        
        # β=0.0 Probe results
        probe0_log = log_dir / "beta0_probe.log"
        if probe0_log.exists():
            action_f1 = parse_probe_log(probe0_log)
            if action_f1:
                results.append({
                    'Run': run_dir.name,
                    'Model': 'β=0.0 + Probe',
                    'Scenario F1': f"{scenario_f1:.4f}" if scenario_f1 else 'N/A',
                    'Action F1': f"{action_f1:.4f}",
                    'Notes': 'Frozen backbone + linear probe'
                })
        
        # β=0.3 results
        beta03_log = log_dir / "beta03.log"
        if beta03_log.exists():
            scenario_f1, action_f1 = parse_training_log(beta03_log)
            if scenario_f1:
                results.append({
                    'Run': run_dir.name,
                    'Model': 'β=0.3 Joint',
                    'Scenario F1': f"{scenario_f1:.4f}",
                    'Action F1': f"{action_f1:.4f}" if action_f1 else 'N/A',
                    'Notes': 'Joint training'
                })
        
        # β=0.3 Probe results
        probe03_log = log_dir / "beta03_probe.log"
        if probe03_log.exists():
            action_f1 = parse_probe_log(probe03_log)
            if action_f1:
                results.append({
                    'Run': run_dir.name,
                    'Model': 'β=0.3 + Probe',
                    'Scenario F1': 'N/A',
                    'Action F1': f"{action_f1:.4f}",
                    'Notes': 'Probe on joint model'
                })
    
    # Check baselines
    for baseline_dir in base_dir.glob("baselines_*"):
        csv_file = baseline_dir / "baseline_results.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                results.append({
                    'Run': baseline_dir.name,
                    'Model': row['model'].upper(),
                    'Scenario F1': f"{row['best_scenario_f1']:.4f}",
                    'Action F1': f"{row.get('final_action_f1', 0):.4f}",
                    'Notes': 'Baseline'
                })
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Display
    print("\n" + "="*80)
    print("COMPLETE RESULTS SUMMARY")
    print("="*80)
    print(df_results.to_string(index=False))
    
    # Save
    output_file = "results_summary.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
    
    # Find best models
    print("\n" + "="*80)
    print("BEST PERFORMERS")
    print("="*80)
    
    # Best Scenario
    scenario_rows = df_results[df_results['Scenario F1'] != 'N/A'].copy()
    scenario_rows['Scenario F1 Num'] = scenario_rows['Scenario F1'].astype(float)
    best_scenario = scenario_rows.loc[scenario_rows['Scenario F1 Num'].idxmax()]
    print(f"Best Scenario F1: {best_scenario['Model']} - {best_scenario['Scenario F1']}")
    
    # Best Action
    action_rows = df_results[df_results['Action F1'] != 'N/A'].copy()
    action_rows['Action F1 Num'] = action_rows['Action F1'].astype(float)
    best_action = action_rows.loc[action_rows['Action F1 Num'].idxmax()]
    print(f"Best Action F1: {best_action['Model']} - {best_action['Action F1']}")

if __name__ == "__main__":
    main()
