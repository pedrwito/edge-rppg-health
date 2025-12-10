import os
import numpy as np
import pandas as pd

from IppgSignalObtainer import IppgSignalObtainer
from Tools.ParametersCalculator import ParametersCalculator


def calculate_gt_hr_from_rr_file(rr_file_path):
    """
    Calculate overall HR from ground truth RR intervals file.
    
    Args:
        rr_file_path: Path to .rr file (format: time_seconds RR_interval_seconds)
    
    Returns:
        float: Overall heart rate in BPM
    """
    rr_intervals = []
    with open(rr_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    rr_interval_s = float(parts[1])
                    rr_intervals.append(rr_interval_s)
                except ValueError:
                    continue
    
    if len(rr_intervals) == 0:
        return None
    
    # Calculate mean RR interval in seconds, convert to BPM
    mean_rr_s = np.mean(rr_intervals)
    hr_bpm = 60.0 / mean_rr_s
    
    return hr_bpm


def process_video(original_data_dir, participant_id, video_filename=None, method='pos'):
    """
    Process a single video and calculate overall HR error for each ROI.
    Same structure as main.py - calculates HR for all ROIs and stores errors.
    
    Args:
        original_data_dir: Path to "2. ORIGINAL DATA" directory
        participant_id: Participant ID (e.g., 'P1LC1')
        video_filename: Optional video filename (if None, searches for common patterns)
        method: rPPG method to use ('pos', 'chrom', 'ica', 'green')
    
    Returns:
        dict: Results with same structure as main.py
    """
    participant_dir = os.path.join(original_data_dir, participant_id)
    
    if not os.path.exists(participant_dir):
        raise FileNotFoundError(f"Participant directory not found: {participant_dir}")
    
    # Find video file
    if video_filename is None:
        # Try common video filename patterns
        patterns = [
            f"{participant_id}_edited.avi",
            f"{participant_id}_edited.mp4",
            f"{participant_id}-edited.avi",
            f"{participant_id}-edited.mp4",
        ]
        video_path = None
        for pattern in patterns:
            test_path = os.path.join(participant_dir, pattern)
            if os.path.exists(test_path):
                video_path = test_path
                break
        
        if video_path is None:
            # List all video files
            for f in os.listdir(participant_dir):
                if f.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
                    video_path = os.path.join(participant_dir, f)
                    break
        
        if video_path is None:
            raise FileNotFoundError(f"No video file found in {participant_dir}")
    else:
        video_path = os.path.join(participant_dir, video_filename)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Find RR intervals file
    rr_patterns = [
        f"{participant_id}_Mobi_RR-intervals.rr",
        f"{participant_id}_RR-intervals.rr",
    ]
    rr_path = None
    for pattern in rr_patterns:
        test_path = os.path.join(participant_dir, pattern)
        if os.path.exists(test_path):
            rr_path = test_path
            break
    
    if rr_path is None:
        raise FileNotFoundError(f"No RR intervals file found in {participant_dir}")
    
    # Calculate HR from ground truth RR intervals
    hr_gt = calculate_gt_hr_from_rr_file(rr_path)
    
    if hr_gt is None:
        raise ValueError(f"Could not calculate GT HR from {rr_path}")
    
    print(f"Processing {participant_id} (GT HR: {hr_gt:.2f}) ...")
    
    # Extract ROIs once per video (same for all methods) - same as main.py
    fs = 30
    window_length = 60
    start_time = 0
    
    rois = IppgSignalObtainer.extractSeriesRoiRGBFromVideo(
        video_path, fs, window_length=window_length, start_time=start_time,
        forehead=True, cheeks=True, under_nose=False, play_video=False
    )
    
    if not isinstance(rois, dict) or len(rois) == 0:
        raise ValueError(f"No ROI data extracted from {video_path}")
    
    # Process each ROI and calculate HR (same as main.py)
    calc = ParametersCalculator()
    hr_by_roi = {}
    
    for roi_name, channels in rois.items():
        red = channels.get('red', [])
        green = channels.get('green', [])
        blue = channels.get('blue', [])
        
        if len(red) == 0 or len(green) == 0 or len(blue) == 0:
            continue
        
        ippg = IppgSignalObtainer.GetRppGSeriesfromRGBSeries(
            red, green, blue, fs, normalize=False, derivative=False,
            bandpass=True, detrend=True, method=method
        )
        
        if len(ippg) == 0:
            continue
        
        hr = calc.ObtainHeartRate(np.array(ippg), np.array([]), fs, method='two_peaks_periodogram')
        hr_by_roi[roi_name] = float(hr)
    
    if len(hr_by_roi) == 0:
        raise ValueError(f"Could not compute HR for any ROI in {participant_id} with method {method}")
    
    # Find best ROI using the same method (same as main.py)
    best_roi = IppgSignalObtainer.FindBestRoiUsingHEnergyBand(rois, fs, method=method)
    hr_best = hr_by_roi.get(best_roi, None)
    if hr_best is None:
        best_roi = next(iter(hr_by_roi.keys()))
        hr_best = hr_by_roi[best_roi]
    
    # Create result row with same structure as main.py
    result_row = {
        'method': method,
        'file': participant_id,  # Using participant_id as file identifier
        'best_roi': best_roi,
        'hr': hr_best,
        'gt_hr': hr_gt,
    }
    
    # Add error for each ROI (same as main.py)
    for roi_name in sorted(hr_by_roi.keys()):
        result_row[f'error_for_roi_{roi_name}'] = abs(hr_by_roi[roi_name] - hr_gt)
    
    print(f"  {method}: best_roi={best_roi}, hr={hr_best:.2f}, gt={hr_gt:.2f}")
    
    return result_row


def main():
    """
    Main function to process all videos in the "2. ORIGINAL DATA" directory
    Similar to main.py - calculates overall HR for each video and compares with GT
    """
    original_data_dir = os.path.join("googledrive-archive", "2. ORIGINAL DATA")
    
    if not os.path.exists(original_data_dir):
        print(f"Error: Directory not found: {original_data_dir}")
        return
    
    # Get all participant directories
    participant_ids = []
    for item in os.listdir(original_data_dir):
        item_path = os.path.join(original_data_dir, item)
        if os.path.isdir(item_path) and item.startswith('P'):
            participant_ids.append(item)
    
    participant_ids = sorted(participant_ids)
    print(f"Found {len(participant_ids)} participants: {participant_ids}")
    
    # All IPPG methods to compare (same as main.py)
    ippg_methods = ['pos', 'chrom', 'ica', 'green']
    
    results_rows = []
    roi_names_union = set()
    
    # Process each participant
    for participant_id in participant_ids:
        for method in ippg_methods:
            try:
                result_row = process_video(original_data_dir, participant_id, method=method)
                # Track all ROI names seen
                for roi_name in sorted(result_row.keys()):
                    if roi_name.startswith('error_for_roi_'):
                        roi_names_union.add(roi_name.replace('error_for_roi_', ''))
                results_rows.append(result_row)
            except Exception as e:
                print(f"Error processing {participant_id} with {method}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if len(results_rows) == 0:
        print("No results to save.")
        return
    
    # Ensure all error columns exist (same as main.py)
    ordered_error_cols = [f'error_for_roi_{r}' for r in sorted(roi_names_union)]
    for result_row in results_rows:
        for col in ordered_error_cols:
            if col not in result_row:
                result_row[col] = np.nan
    
    # Create DataFrame with same column order as main.py
    out_df = pd.DataFrame(results_rows, columns=['method', 'file', 'best_roi', 'hr', 'gt_hr'] + ordered_error_cols)
    out_path = "results_processed_data_analysis.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")
    
    # Print summary statistics (same format as main.py)
    print("\nMSE and RMSE per Method and ROI (across processed files):")
    for method in ippg_methods:
        method_df = out_df[out_df['method'] == method]
        if len(method_df) == 0:
            continue
        print(f"\nMethod: {method}")
        for roi_name in sorted(roi_names_union):
            col = f'error_for_roi_{roi_name}'
            errs = method_df[col].dropna().to_numpy()
            if errs.size == 0:
                continue
            mse = float(np.mean(errs ** 2))
            rmse = float(np.sqrt(mse))
            print(f"  {roi_name}: MSE={mse:.3f}  RMSE={rmse:.3f}")


if __name__ == "__main__":
    main()
