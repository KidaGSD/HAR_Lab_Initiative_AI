#!/usr/bin/env python3
"""
Simulate AR glasses deployment with mock sensor data streaming.
Tests real-time inference pipeline before deploying to actual hardware.

Usage:
    python scripts/simulate_glasses.py --model deployment/model.onnx --stats deployment/normalization_stats.json
"""

import numpy as np
import argparse
import time
from pathlib import Path
from tqdm import tqdm
import json
import sys
import random

sys.path.append(str(Path(__file__).parent))
from inference_engine import InferenceEngine


class MockSensorStream:
    """Generate mock sensor data stream from preprocessed data"""
    
    def __init__(self, seq_npz_path, fps=50, loop=True):
        """
        Args:
            seq_npz_path: Path to seq.npz file with preprocessed data
            fps: Sampling rate (default 50Hz)
            loop: Whether to loop the data when reaching the end
        """
        self.seq_npz_path = Path(seq_npz_path)
        self.fps = fps
        self.loop = loop
        
        # Load data
        data = np.load(self.seq_npz_path)
        self.imu_windows = data['traj']  # (N_windows, 50, 6)
        
        if 'gaze' in data:
            gaze_data = data['gaze']
            if gaze_data.size > 0:
                self.gaze_windows = gaze_data  # (N_windows, 50, 2)
            else:
                self.gaze_windows = None
        else:
            self.gaze_windows = None
        
        # Flatten windows into samples
        self.imu_samples = self.imu_windows.reshape(-1, 6)  # (N_samples, 6)
        if self.gaze_windows is not None:
            self.gaze_samples = self.gaze_windows.reshape(-1, 2)  # (N_samples, 2)
        else:
            self.gaze_samples = None
        
        self.num_samples = len(self.imu_samples)
        self.current_idx = 0
        
        print(f"✓ Loaded mock sensor data: {self.num_samples} samples ({self.num_samples/self.fps:.1f} seconds)")
        if self.gaze_samples is not None:
            print(f"   Gaze data: available")
        else:
            print(f"   Gaze data: missing (will zero-fill)")
    
    def get_next_sample(self):
        """Get next sensor sample"""
        if self.current_idx >= self.num_samples:
            if self.loop:
                self.current_idx = 0
            else:
                return None, None
        
        imu_sample = self.imu_samples[self.current_idx]
        gaze_sample = self.gaze_samples[self.current_idx] if self.gaze_samples is not None else None
        
        self.current_idx += 1
        return imu_sample, gaze_sample
    
    def reset(self):
        """Reset stream to beginning"""
        self.current_idx = 0


def simulate_real_time_inference(engine, sensor_stream, inference_interval=1.0, duration=None, verbose=True):
    """
    Simulate real-time inference with sensor streaming.
    
    Args:
        engine: InferenceEngine instance
        sensor_stream: MockSensorStream instance
        inference_interval: How often to run inference (seconds)
        duration: Total simulation duration (seconds), None for full stream
        verbose: Whether to print predictions
    
    Returns:
        results: List of inference results
    """
    results = []
    start_time = time.time()
    last_inference_time = 0
    sample_count = 0
    
    # Calculate duration
    if duration is None:
        duration = sensor_stream.num_samples / sensor_stream.fps
    
    print(f"\n🚀 Starting simulation...")
    print(f"   Duration: {duration:.1f} seconds")
    print(f"   Inference interval: {inference_interval:.1f} seconds")
    print(f"   Sampling rate: {sensor_stream.fps} Hz\n")
    
    try:
        while True:
            current_time = time.time() - start_time
            
            # Check if duration exceeded
            if duration is not None and current_time >= duration:
                break
            
            # Get next sample
            imu_sample, gaze_sample = sensor_stream.get_next_sample()
            if imu_sample is None:
                break
            
            # Add sample to engine buffer
            engine.add_sample(imu_sample, gaze_sample)
            sample_count += 1
            
            # Run inference at specified interval
            if current_time - last_inference_time >= inference_interval:
                try:
                    scenario_pred, action_preds, scenario_probs, action_probs_list = engine.infer()
                    
                    # Get top scenario probability
                    top_scenario_prob = max(scenario_probs.values())
                    
                    # Get most common action
                    from collections import Counter
                    action_counts = Counter(action_preds)
                    most_common_action = action_counts.most_common(1)[0][0]
                    most_common_action_count = action_counts.most_common(1)[0][1]
                    
                    result = {
                        'time': current_time,
                        'sample_count': sample_count,
                        'scenario': scenario_pred,
                        'scenario_prob': top_scenario_prob,
                        'scenario_probs': scenario_probs,
                        'most_common_action': most_common_action,
                        'action_distribution': dict(action_counts),
                        'buffer_status': engine.get_buffer_status()
                    }
                    results.append(result)
                    
                    if verbose:
                        print(f"[{current_time:6.1f}s] Scenario: {scenario_pred:20s} ({top_scenario_prob:.2%}) | "
                              f"Action: {most_common_action:20s} ({most_common_action_count}/{len(action_preds)}) | "
                              f"Buffer: {result['buffer_status']['percent_full']:.1f}%")
                    
                    last_inference_time = current_time
                    
                except ValueError as e:
                    # Buffer not full enough yet
                    if verbose:
                        print(f"[{current_time:6.1f}s] Waiting for buffer... ({engine.buffer.size()}/{engine.buffer.max_samples})")
            
            # Simulate real-time sampling rate
            time.sleep(1.0 / sensor_stream.fps)
            
    except KeyboardInterrupt:
        print("\n⚠️  Simulation interrupted by user")
    
    print(f"\n✓ Simulation complete: {len(results)} inferences, {sample_count} samples processed")
    return results


def main():
    parser = argparse.ArgumentParser(description='Simulate AR glasses deployment')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to exported model (ONNX or TorchScript)')
    parser.add_argument('--stats', type=str, required=True,
                        help='Path to normalization_stats.json')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to seq.npz file for simulation (default: use test set)')
    parser.add_argument('--processed-dir', type=str, default='data/processed_ego4d',
                        help='Directory containing processed data')
    parser.add_argument('--scenario-labels', type=str, default='data/labels/scenario_labels.csv',
                        help='Path to scenario labels CSV')
    parser.add_argument('--inference-interval', type=float, default=1.0,
                        help='Inference interval in seconds (default: 1.0)')
    parser.add_argument('--duration', type=float, default=None,
                        help='Simulation duration in seconds (default: full stream)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file to save results')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run inference on')
    parser.add_argument('--no-verbose', action='store_true',
                        help='Disable verbose output')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    print("="*80)
    print("Initializing Inference Engine")
    print("="*80)
    engine = InferenceEngine(
        model_path=args.model,
        stats_path=args.stats,
        use_gaze=None,  # Auto-detect from normalization_stats.json config
        device=args.device
    )
    
    # Load sensor data
    test_uid = None
    if args.data:
        seq_npz_path = Path(args.data)
        # Extract UID from path if possible
        if seq_npz_path.parent.name:
            test_uid = seq_npz_path.parent.name
    else:
        # Use a random test set video
        import pandas as pd
        scenario_df = pd.read_csv(args.scenario_labels)
        test_uids = scenario_df[scenario_df['split'] == 'test']['video_uid'].tolist()
        
        processed_dir = Path(args.processed_dir)
        available_uids = [p.parent.name for p in processed_dir.glob('*/seq.npz')]
        test_uids_available = [u for u in test_uids if u in available_uids]
        
        if len(test_uids_available) == 0:
            print("❌ ERROR: No test videos with processed data found!")
            return
        
        # Randomly select a test video
        test_uid = random.choice(test_uids_available)
        seq_npz_path = processed_dir / test_uid / 'seq.npz'
        print(f"📹 Selected test video (random): {test_uid}")
        print(f"   Total available test videos: {len(test_uids_available)}")
    
    if not seq_npz_path.exists():
        print(f"❌ ERROR: Data file not found: {seq_npz_path}")
        return
    
    # Print video UID prominently
    if test_uid:
        print(f"\n{'='*80}")
        print(f"Video UID: {test_uid}")
        print(f"Data file: {seq_npz_path}")
        print(f"{'='*80}\n")
    
    # Create mock sensor stream
    print("\n" + "="*80)
    print("Loading Mock Sensor Data")
    print("="*80)
    sensor_stream = MockSensorStream(seq_npz_path, fps=50, loop=False)
    
    # Run simulation
    print("\n" + "="*80)
    print("Running Simulation")
    print("="*80)
    results = simulate_real_time_inference(
        engine=engine,
        sensor_stream=sensor_stream,
        inference_interval=args.inference_interval,
        duration=args.duration,
        verbose=not args.no_verbose
    )
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        results_serializable = convert_to_serializable(results)
        
        # Add metadata to results
        output_data = {
            'metadata': {
                'video_uid': test_uid,
                'data_file': str(seq_npz_path),
                'model_path': str(args.model),
                'stats_path': str(args.stats),
                'inference_interval': args.inference_interval,
                'duration': args.duration,
                'total_inferences': len(results)
            },
            'results': results_serializable
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
        if test_uid:
            print(f"   Video UID: {test_uid}")
        print(f"   Total inferences: {len(results)}")
    
    # Print summary
    if results:
        print("\n" + "="*80)
        print("Simulation Summary")
        print("="*80)
        
        scenarios = [r['scenario'] for r in results]
        from collections import Counter
        scenario_counts = Counter(scenarios)
        
        print(f"\nScenario predictions:")
        for scenario, count in scenario_counts.most_common():
            print(f"  {scenario:20s}: {count:3d} ({count/len(results):.1%})")
        
        print(f"\nTotal inferences: {len(results)}")
        print(f"Average scenario confidence: {np.mean([r['scenario_prob'] for r in results]):.2%}")


if __name__ == "__main__":
    main()

