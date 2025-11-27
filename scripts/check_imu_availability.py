#!/usr/bin/env python3
"""
Check IMU coverage against Ego4D manifest (all available IMU data).
This tells us what % of our labeled videos have IMU data available on Ego4D servers.
"""

import pandas as pd
from pathlib import Path

# Paths
LABELS_PATH = 'data/labels/action_labels_llm_clean.csv'
MANIFEST_PATH = 'data/ego4d_data/v2/imu/manifest.csv'
SCENARIO_LABELS_PATH = 'data/labels/scenario_labels.csv'

def main():
    print("=" * 80)
    print("IMU COVERAGE ANALYSIS: Checking Against Ego4D Manifest")
    print("=" * 80)
    
    # 1. Load labeled videos
    print("\n1. Loading labeled videos...")
    df_labels = pd.read_csv(LABELS_PATH)
    labeled_uids = set(df_labels['video_uid'].unique())
    
    print(f"   Labeled narrations: {len(df_labels):,}")
    print(f"   Unique labeled videos: {len(labeled_uids):,}")
    
    # 2. Load IMU manifest (what's available on Ego4D)
    print("\n2. Loading Ego4D IMU manifest...")
    df_manifest = pd.read_csv(MANIFEST_PATH)
    manifest_uids = set(df_manifest['video_uid'].unique())
    
    print(f"   Total IMU files on Ego4D: {len(manifest_uids):,}")
    
    # 3. Calculate coverage
    print("\n3. Calculating coverage...")
    covered_uids = labeled_uids.intersection(manifest_uids)
    missing_uids = labeled_uids - manifest_uids
    
    coverage_pct = len(covered_uids) / len(labeled_uids) * 100 if labeled_uids else 0
    
    print("\n" + "=" * 80)
    print("RESULTS: IMU Data Availability on Ego4D")
    print("=" * 80)
    print(f"✅ Videos with IMU available: {len(covered_uids):,} ({coverage_pct:.1f}%)")
    print(f"❌ Videos without IMU: {len(missing_uids):,} ({100-coverage_pct:.1f}%)")
    print("=" * 80)
    
    # 4. Breakdown by scenario
    if Path(SCENARIO_LABELS_PATH).exists():
        print("\n4. Coverage by scenario...")
        df_scenario = pd.read_csv(SCENARIO_LABELS_PATH)
        df_scenario = df_scenario[df_scenario['video_uid'].isin(labeled_uids)]
        
        for scenario in df_scenario['scenario'].unique():
            scenario_uids = set(df_scenario[df_scenario['scenario'] == scenario]['video_uid'])
            scenario_covered = scenario_uids.intersection(manifest_uids)
            scenario_pct = len(scenario_covered) / len(scenario_uids) * 100 if scenario_uids else 0
            
            print(f"   {scenario:25s}: {len(scenario_covered):4d}/{len(scenario_uids):4d} ({scenario_pct:5.1f}%)")
    
    # 5. Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    covered_narrations = len(df_labels[df_labels['video_uid'].isin(covered_uids)])
    total_narrations = len(df_labels)
    
    print(f"📊 Narrations with IMU available: {covered_narrations:,}/{total_narrations:,} ({covered_narrations/total_narrations*100:.1f}%)")
    
    if coverage_pct >= 90:
        print("\n✅ EXCELLENT: >90% of videos have IMU data available!")
        print("   You can proceed with downloading and using IMU for training.")
    elif coverage_pct >= 70:
        print("\n✅ GOOD: >70% coverage.")
        print("   Most videos have IMU data available.")
    elif coverage_pct >= 50:
        print("\n⚠️ MODERATE: 50-70% coverage.")
        print("   Some videos may not have IMU data.")
    else:
        print("\n❌ WARNING: <50% coverage!")
        print("   Many of your labeled videos don't have IMU data on Ego4D.")
        print("   This may limit IMU-based training.")
    
    # 6. Show missing videos
    if len(missing_uids) > 0 and len(missing_uids) <= 20:
        print(f"\n📋 Videos without IMU data ({len(missing_uids)} total):")
        for uid in sorted(missing_uids):
            print(f"   - {uid}")
    elif len(missing_uids) > 20:
        print(f"\n📋 Sample of videos without IMU data (showing 20/{len(missing_uids)}):")
        for uid in sorted(list(missing_uids))[:20]:
            print(f"   - {uid}")
    
    print("\n" + "=" * 80)
    
    # 7. Next steps
    print("\nNEXT STEPS:")
    print("=" * 80)
    if coverage_pct > 80:
        print("1. Download IMU data for your labeled videos:")
        print("   cd Ego4d")
        print("   python -m ego4d.cli.cli \\")
        print("       --output_directory ../data/ego4d_data \\")
        print("       --datasets imu \\")
        print("       --video_uids ../target_uids.csv")
        print("\n2. Use notebooks/imu_coverage_analysis.ipynb to visualize examples")
        print("\n3. Proceed with IMU-based training")
    else:
        print("⚠️ Coverage is low. Consider:")
        print("1. Filtering your labeled dataset to only use videos with IMU")
        print("2. Or accepting that not all narrations will have IMU validation")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
