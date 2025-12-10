import os
import numpy as np
import pandas as pd

from IppgSignalObtainer import IppgSignalObtainer
from Tools.ParametersCalculator import ParametersCalculator
from process_ubfc_dataset import load_ubfc_ground_truth


def calculate_gt_hr_from_ubfc(gt_trace, gt_time, gt_hr, fs=None):
    """
    Calculate overall HR from UBFC ground truth data.
    
    Args:
        gt_trace: PPG signal values
        gt_time: Time values (in milliseconds)
        gt_hr: Heart rate values from GT file (array)
        fs: Sampling frequency (optional, will be calculated if not provided)
    
    Returns:
        float: Overall heart rate in BPM
    """
    # Use the mean of HR values from the GT file
    # The GT file contains HR values for each time point
    if gt_hr is not None and len(gt_hr) > 0:
        # Filter out any invalid values
        valid_hr = gt_hr[np.isfinite(gt_hr)]
        if len(valid_hr) > 0:
            return float(np.mean(valid_hr))
    
    # Fallback: calculate HR from PPG signal using periodogram
    if fs is None:
        # Calculate fs from time array
        if len(gt_time) > 1:
            time_seconds = gt_time / 1000.0  # Convert ms to seconds
            fs = len(gt_trace) / time_seconds[-1] if time_seconds[-1] > 0 else 30
        else:
            fs = 30  # Default
    
    # Calculate HR from signal using periodogram
    calc = ParametersCalculator()
    hr = calc.ObtainHeartRate(gt_trace, np.array([]), fs, method='two_peaks_periodogram')
    return float(hr)


def main():
    base_folder = 'UBFC_DATASET_MERGED'
    fs = 30  # UBFC dataset videos are typically at 30 fps
    window_length = 60  # Process 60 seconds of video
    start_time = 5
    
    # All IPPG methods to compare
    ippg_methods = ['pos', 'chrom', 'ica', 'green']
    
    calc = ParametersCalculator()
    results_rows = []
    roi_names_union = set()
    
    # Process both DATASET_1 and DATASET_2
    for dataset_name in ['DATASET_1', 'DATASET_2']:
        dataset_path = os.path.join(base_folder, dataset_name)
        
        if not os.path.exists(dataset_path):
            print(f"Dataset {dataset_name} not found, skipping...")
            continue
        
        print("=" * 60)
        print(f"Processing {dataset_name}")
        print("=" * 60)
        
        # Get list of subject directories
        try:
            all_entries = os.listdir(dataset_path)
            dirs = [d for d in all_entries if os.path.isdir(os.path.join(dataset_path, d)) 
                    and d not in ['.', '..', 'desktop.ini']]
            dirs.sort()
        except Exception as e:
            print(f"Error listing directories in {dataset_path}: {e}")
            continue
        
        print(f"Found {len(dirs)} subjects to process")
        
        # Process each subject
        for idx, dir_name in enumerate(dirs):
            vid_folder = os.path.join(dataset_path, dir_name)
            video_path = os.path.join(vid_folder, 'vid.avi')
            
            # Check if video exists
            if not os.path.exists(video_path):
                print(f"Warning: Video not found for {dir_name}: {video_path}")
                continue
            
            # Load ground truth
            gt_trace, gt_time, gt_hr, format_used = load_ubfc_ground_truth(vid_folder)
            
            if gt_trace is None:
                print(f"Warning: No ground truth found for {dir_name}, skipping...")
                continue
            
            # Calculate GT HR
            gt_hr_mean = calculate_gt_hr_from_ubfc(gt_trace, gt_time, gt_hr, fs=fs)
            
            if gt_hr_mean is None or np.isnan(gt_hr_mean):
                print(f"Warning: Could not calculate GT HR for {dir_name}, skipping...")
                continue
            
            print(f"\n[{idx+1}/{len(dirs)}] Processing {dir_name} (GT HR: {gt_hr_mean:.2f} bpm)")
            
            # Extract ROIs once per video (same for all methods)
            try:
                rois = IppgSignalObtainer.extractSeriesRoiRGBFromVideo(
                    video_path, fs, window_length=window_length, start_time=start_time,
                    forehead=True, cheeks=True, under_nose=False, play_video=False
                )
            except Exception as e:
                print(f"  Error extracting ROIs: {e}")
                continue
            
            if not isinstance(rois, dict) or len(rois) == 0:
                print(f"  No ROI data extracted for {dir_name}; skipping.")
                continue
            
            # Process each IPPG method
            for method in ippg_methods:
                print(f"  Processing with method: {method}")
                hr_by_roi = {}
                
                for roi_name, channels in rois.items():
                    red = channels.get('red', [])
                    green = channels.get('green', [])
                    blue = channels.get('blue', [])
                    
                    if len(red) == 0 or len(green) == 0 or len(blue) == 0:
                        continue
                    
                    try:
                        ippg = IppgSignalObtainer.GetRppGSeriesfromRGBSeries(
                            red, green, blue, fs, normalize=False, derivative=False,
                            bandpass=True, detrend=True, method=method
                        )
                        
                        if len(ippg) == 0:
                            continue
                        
                        hr = calc.ObtainHeartRate(np.array(ippg), np.array([]), fs, method='two_peaks_periodogram')
                        hr_by_roi[roi_name] = float(hr)
                    except Exception as e:
                        print(f"    Error processing {roi_name} with {method}: {e}")
                        continue
                
                if len(hr_by_roi) == 0:
                    print(f"  Could not compute HR for any ROI in {dir_name} with method {method}; skipping.")
                    continue
                
                # Find best ROI using the same method
                try:
                    best_roi = IppgSignalObtainer.FindBestRoiUsingHEnergyBand(rois, fs, method=method)
                    hr_best = hr_by_roi.get(best_roi, None)
                    if hr_best is None:
                        best_roi = next(iter(hr_by_roi.keys()))
                        hr_best = hr_by_roi[best_roi]
                except Exception as e:
                    print(f"    Error finding best ROI: {e}")
                    best_roi = next(iter(hr_by_roi.keys()))
                    hr_best = hr_by_roi[best_roi]
                
                result_row = {
                    'method': method,
                    'file': dir_name,
                    'dataset': dataset_name,
                    'best_roi': best_roi,
                    'hr': hr_best,
                    'gt_hr': gt_hr_mean,
                }
                
                for roi_name in sorted(hr_by_roi.keys()):
                    roi_names_union.add(roi_name)
                    result_row[f'error_for_roi_{roi_name}'] = abs(hr_by_roi[roi_name] - gt_hr_mean)
                
                results_rows.append(result_row)
                print(f"    {method}: best_roi={best_roi}, hr={hr_best:.2f}, gt={gt_hr_mean:.2f}, error={abs(hr_best - gt_hr_mean):.2f}")
    
    if len(results_rows) == 0:
        print("No results to save.")
        return
    
    # Create DataFrame with all results
    ordered_error_cols = [f'error_for_roi_{r}' for r in sorted(roi_names_union)]
    for result_row in results_rows:
        for col in ordered_error_cols:
            if col not in result_row:
                result_row[col] = np.nan
    
    out_df = pd.DataFrame(results_rows, columns=['method', 'file', 'dataset', 'best_roi', 'hr', 'gt_hr'] + ordered_error_cols)
    out_path = "results_ubfc_comparison.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    print("\nMSE and RMSE per Method and ROI (across all subjects):")
    for method in ippg_methods:
        method_df = out_df[out_df['method'] == method]
        if len(method_df) == 0:
            continue
        print(f"\nMethod: {method}")
        print(f"  Overall MAE: {np.mean(np.abs(method_df['hr'] - method_df['gt_hr'])):.3f} bpm")
        print(f"  Overall RMSE: {np.sqrt(np.mean((method_df['hr'] - method_df['gt_hr'])**2)):.3f} bpm")
        
        for roi_name in sorted(roi_names_union):
            col = f'error_for_roi_{roi_name}'
            errs = method_df[col].dropna().to_numpy()
            if errs.size == 0:
                continue
            mse = float(np.mean(errs ** 2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(errs))
            print(f"  {roi_name}: MAE={mae:.3f} bpm, MSE={mse:.3f}, RMSE={rmse:.3f} bpm")
    
    # Print per-dataset statistics
    print("\n" + "=" * 60)
    print("Per-Dataset Statistics")
    print("=" * 60)
    for dataset_name in ['DATASET_1', 'DATASET_2']:
        dataset_df = out_df[out_df['dataset'] == dataset_name]
        if len(dataset_df) == 0:
            continue
        print(f"\n{dataset_name}:")
        for method in ippg_methods:
            method_df = dataset_df[dataset_df['method'] == method]
            if len(method_df) == 0:
                continue
            mae = np.mean(np.abs(method_df['hr'] - method_df['gt_hr']))
            rmse = np.sqrt(np.mean((method_df['hr'] - method_df['gt_hr'])**2))
            print(f"  {method}: MAE={mae:.3f} bpm, RMSE={rmse:.3f} bpm")


if __name__ == "__main__":
    main()

