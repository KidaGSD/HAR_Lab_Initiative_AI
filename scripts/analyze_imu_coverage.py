#!/usr/bin/env python3
"""
Analyze IMU data coverage and correlation with action labels.

This script fixes the issues in the notebook by:
1. Using correct column names (canonical_timestamp_ms, accl_x/y/z)
2. Adding comprehensive IMU coverage analysis
3. Generating proper visualizations

Run this independently or use the code to fix the notebook cells.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

# Paths
LABELS_PATH = '../data/labels/action_labels_llm_clean.csv'
IMU_DIR = Path('../data/ego4d_data/v2/imu')

def load_imu_data(video_uid):
    """
    Load and standardize IMU data for a specific video.
    
    Handles the actual Ego4D IMU format:
    - canonical_timestamp_ms (milliseconds)
    - accl_x, accl_y, accl_z (acceleration)
    - gyro_x, gyro_y, gyro_z (gyroscope)
    """
    imu_file = IMU_DIR / f"{video_uid}.csv"
    if not imu_file.exists():
        return None
    
    try:
        imu = pd.read_csv(imu_file)
        
        # Standardize column names and units
        # 1. Convert timestamp from milliseconds to seconds
        if 'canonical_timestamp_ms' in imu.columns:
            imu['timestamp'] = imu['canonical_timestamp_ms'] / 1000.0
        elif 'component_timestamp_ms' in imu.columns:
            imu['timestamp'] = imu['component_timestamp_ms'] / 1000.0
        else:
            print(f"Warning: No timestamp column found in {video_uid}")
            return None
        
        # 2. Rename acceleration columns (accl -> accel)
        rename_map = {}
        if 'accl_x' in imu.columns:
            rename_map.update({'accl_x': 'accel_x', 'accl_y': 'accel_y', 'accl_z': 'accel_z'})
        
        if rename_map:
            imu = imu.rename(columns=rename_map)
        
        # 3. Calculate magnitude of acceleration
        if all(col in imu.columns for col in ['accel_x', 'accel_y', 'accel_z']):
            imu['accel_mag'] = np.sqrt(
                imu['accel_x']**2 + imu['accel_y']**2 + imu['accel_z']**2
            )
        
        # 4. Calculate magnitude of gyroscope
        if all(col in imu.columns for col in ['gyro_x', 'gyro_y', 'gyro_z']):
            imu['gyro_mag'] = np.sqrt(
                imu['gyro_x']**2 + imu['gyro_y']**2 + imu['gyro_z']**2
            )
        
        return imu
        
    except Exception as e:
        print(f"Error loading {video_uid}: {e}")
        return None


def analyze_coverage(labels_df, imu_dir):
    """Analyze what percentage of labeled videos have IMU data"""
    
    # Get all available IMU files
    imu_files = list(imu_dir.glob('*.csv'))
    imu_files = [f for f in imu_files if f.stem not in ['manifest', 'manifest.ver']]
    
    # Extract UIDs
    labeled_uids = set(labels_df['video_uid'].unique())
    imu_uids = set(f.stem for f in imu_files)
    
    # Calculate coverage
    covered_uids = labeled_uids.intersection(imu_uids)
    missing_uids = labeled_uids - imu_uids
    
    coverage_stats = {
        'total_labeled': len(labeled_uids),
        'have_imu': len(covered_uids),
        'missing_imu': len(missing_uids),
        'coverage_pct': len(covered_uids) / len(labeled_uids) * 100 if labeled_uids else 0,
        'covered_uids': covered_uids,
        'missing_uids': missing_uids
    }
    
    return coverage_stats


def visualize_imu_with_labels(video_uid, labels_df, imu_dir):
    """Create comprehensive visualization of IMU + labels for a video"""
    
    # Load data
    imu_data = load_imu_data(video_uid)
    if imu_data is None:
        print(f"No IMU data for {video_uid}")
        return
    
    video_labels = labels_df[labels_df['video_uid'] == video_uid].sort_values('timestamp_sec')
    
    if len(video_labels) == 0:
        print(f"No labels for {video_uid}")
        return
    
    print(f"Found {len(video_labels)} labels for this video")
    print(f"IMU time range: {imu_data['timestamp'].min():.2f}s - {imu_data['timestamp'].max():.2f}s")
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    
    # Define label colors
    label_colors = {
        'Locomotion': 'red',
        'Essential Operation': 'orange',
        'Object Transfer': 'yellow',
        'Search': 'green',
        'Error / Correction': 'purple',
        'Stationary': 'gray'
    }
    
    # Plot 1: Acceleration magnitude
    if 'accel_mag' in imu_data.columns:
        axes[0].plot(imu_data['timestamp'], imu_data['accel_mag'], 
                    linewidth=0.5, alpha=0.7, color='blue', label='Accel Magnitude')
        axes[0].set_ylabel('Acceleration (m/s²)', fontsize=12)
        axes[0].set_title(f'IMU Data + Action Labels: {video_uid}', fontweight='bold', fontsize=14)
        axes[0].grid(alpha=0.3)
        
        # Add label markers
        for _, row in video_labels.iterrows():
            color = label_colors.get(row['action'], 'black')
            axes[0].axvline(row['timestamp_sec'], color=color, alpha=0.4, linewidth=2)
    
    # Plot 2: Gyroscope magnitude
    if 'gyro_mag' in imu_data.columns:
        axes[1].plot(imu_data['timestamp'], imu_data['gyro_mag'], 
                    linewidth=0.5, alpha=0.7, color='green', label='Gyro Magnitude')
        axes[1].set_ylabel('Gyroscope (rad/s)', fontsize=12)
        axes[1].grid(alpha=0.3)
        
        # Add label markers
        for _, row in video_labels.iterrows():
            color = label_colors.get(row['action'], 'black')
            axes[1].axvline(row['timestamp_sec'], color=color, alpha=0.4, linewidth=2)
    
    # Plot 3: Label timeline
    unique_labels = video_labels['action'].unique()
    y_positions = {label: i for i, label in enumerate(unique_labels)}
    
    for _, row in video_labels.iterrows():
        color = label_colors.get(row['action'], 'black')
        y_pos = y_positions[row['action']]
        axes[2].scatter(row['timestamp_sec'], y_pos, c=color, s=100, alpha=0.7, 
                       edgecolors='black', linewidths=0.5)
    
    axes[2].set_yticks(range(len(unique_labels)))
    axes[2].set_yticklabels(unique_labels)
    axes[2].set_ylabel('Action Label', fontsize=12)
    axes[2].set_xlabel('Time (seconds)', fontsize=12)
    axes[2].grid(alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=label) 
                      for label, color in label_colors.items() 
                      if label in video_labels['action'].values]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=9, ncol=2)
    
    plt.tight_layout()
    plt.savefig(f'imu_visualization_{video_uid}.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to imu_visualization_{video_uid}.png")
    plt.show()


def analyze_motion_by_label(labels_df, imu_dir, max_videos=20):
    """Analyze motion patterns for different action labels"""
    
    # Get available IMU files
    imu_files = list(imu_dir.glob('*.csv'))
    imu_files = [f for f in imu_files if f.stem not in ['manifest']]
    
    # Get videos that have both labels and IMU
    labeled_uids = set(labels_df['video_uid'].unique())
    imu_uids = set(f.stem for f in imu_files)
    available_uids = list(labeled_uids.intersection(imu_uids))
    
    # Sample videos
    sample_uids = available_uids[:max_videos]
    
    motion_data = []
    
    print(f"Analyzing motion for {len(sample_uids)} videos...")
    
    for video_uid in sample_uids:
        imu = load_imu_data(video_uid)
        if imu is None or 'accel_mag' not in imu.columns:
            continue
        
        video_labels = labels_df[labels_df['video_uid'] == video_uid]
        
        for _, label_row in video_labels.iterrows():
            ts = label_row['timestamp_sec']
            
            # Get IMU window around this timestamp (±1 second)
            window = imu[(imu['timestamp'] >= ts - 1) & (imu['timestamp'] <= ts + 1)]
            
            if len(window) > 0:
                motion_data.append({
                    'action': label_row['action'],
                    'scenario': label_row['scenario'],
                    'accel_mean': window['accel_mag'].mean(),
                    'accel_std': window['accel_mag'].std(),
                    'accel_max': window['accel_mag'].max(),
                    'gyro_mean': window.get('gyro_mag', pd.Series([0])).mean() if 'gyro_mag' in window.columns else 0
                })
    
    if not motion_data:
        print("No motion data collected")
        return None
    
    motion_df = pd.DataFrame(motion_data)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    motion_df.boxplot(column='accel_mean', by='action', ax=ax1)
    ax1.set_title('Acceleration by Action Label', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean Acceleration (m/s²)', fontsize=12)
    ax1.set_xlabel('Action Label', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    plt.suptitle('')
    
    motion_df.boxplot(column='gyro_mean', by='action', ax=ax2)
    ax2.set_title('Gyroscope by Action Label', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Mean Gyroscope (rad/s)', fontsize=12)
    ax2.set_xlabel('Action Label', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig('motion_by_label_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved motion analysis to motion_by_label_analysis.png")
    plt.show()
    
    # Print statistics
    print("\n=== Motion Statistics by Action Label ===")
    stats = motion_df.groupby('action').agg({
        'accel_mean': ['mean', 'std', 'count'],
        'gyro_mean': ['mean', 'std']
    }).round(3)
    print(stats)
    
    return motion_df


def main():
    print("=" * 80)
    print("IMU DATA COVERAGE AND CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Load labels
    print(f"\n1. Loading labels from {LABELS_PATH}...")
    df = pd.read_csv(LABELS_PATH)
    print(f"   Total labeled narrations: {len(df):,}")
    print(f"   Unique videos: {df['video_uid'].nunique():,}")
    
    # Analyze coverage
    print(f"\n2. Analyzing IMU coverage...")
    coverage = analyze_coverage(df, IMU_DIR)
    
    print(f"\n   === IMU COVERAGE ===")
    print(f"   Total labeled videos: {coverage['total_labeled']:,}")
    print(f"   Videos with IMU data: {coverage['have_imu']:,}")
    print(f"   Coverage: {coverage['coverage_pct']:.1f}%")
    print(f"   Missing IMU data: {coverage['missing_imu']:,} videos")
    
    if coverage['missing_imu'] > 0:
        print(f"\n   Sample missing UIDs (first 5):")
        for uid in list(coverage['missing_uids'])[:5]:
            print(f"      - {uid}")
    
    # Visualize a sample video
    if coverage['have_imu'] > 0:
        print(f"\n3. Visualizing sample video with IMU + labels...")
        sample_uid = list(coverage['covered_uids'])[0]
        visualize_imu_with_labels(sample_uid, df, IMU_DIR)
        
        print(f"\n4. Analyzing motion patterns by action label...")
        motion_df = analyze_motion_by_label(df, IMU_DIR, max_videos=20)
        
        if motion_df is not None:
            print(f"\n   Collected {len(motion_df)} motion samples from {motion_df['action'].nunique()} action types")
    else:
        print("\n⚠️  No IMU data available for any labeled videos")
        print("   Please download IMU data from Ego4D.")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
