import os
import numpy as np
import pandas as pd

from IppgSignalObtainer import IppgSignalObtainer
from Tools.ParametersCalculator import ParametersCalculator


def list_video_files(directory, exts=(".mp4", ".mov", ".avi", ".mkv")):
    files = []
    for name in os.listdir(directory):
        if name.lower().endswith(exts):
            files.append(os.path.join(directory, name))
    print(files)
    return sorted(files)


def main():
    videos_dir = "videos_iPPG_paula"
    excel_gt_path = os.path.join(videos_dir, "Metadata videos iPPG.xlsx")
    fs = 30
    window_length = 60
    start_time = 0

    # All IPPG methods to compare
    ippg_methods = ['pos', 'chrom', 'ica', 'green']

    # Read ground-truth HRs from Excel
    # The Excel has a header row, so we skip it and rename columns properly
    df = pd.read_excel(excel_gt_path, skiprows=1)
    
    # Rename columns based on the actual structure
    df.columns = ['Empty', 'Archivo', 'Celu', 'Promedio', 'Condiciones']
    
    # Drop the empty column if it exists
    if 'Empty' in df.columns:
        df = df.drop(columns=['Empty'])
    
    print(f"Excel columns: {df.columns.tolist()}")
    print(f"Found {len(df)} rows in Excel file")
    
    if "Archivo" not in df.columns or "Promedio" not in df.columns:
        print("Error: Excel file must have 'Archivo' and 'Promedio' columns")
        return

    calc = ParametersCalculator()
    
    results_rows = []
    roi_names_union = set()

    # Iterate through each row in the Excel file
    for idx, row in df.iterrows():
        file_name = str(row["Archivo"]).strip()
        
        # Get ground truth HR
        try:
            gt_hr = float(row["Promedio"]) if not pd.isna(row["Promedio"]) else None
        except Exception:
            gt_hr = None
        
        if not file_name or gt_hr is None:
            print(f"Row {idx}: Skipping (missing filename or HR value)")
            continue
        
        # Try different extensions if the file doesn't have one
        video_path = os.path.join(videos_dir, file_name)
        
        # If file doesn't exist and has no extension, try common video extensions
        if not os.path.exists(video_path) and not any(file_name.lower().endswith(ext) for ext in ['.mp4', '.mov', '.avi', '.mkv']):
            for ext in ['.mp4', '.mov', '.avi', '.mkv']:
                test_path = os.path.join(videos_dir, file_name + ext)
                if os.path.exists(test_path):
                    video_path = test_path
                    file_name = file_name + ext
                    break
        
        # Check if file exists
        if not os.path.exists(video_path):
            print(f"Warning: File not found: {video_path}; skipping.")
            continue
        
        print(f"Processing {file_name} (GT HR: {gt_hr:.2f}) ...")

        # Extract ROIs once per video (same for all methods)
        rois = IppgSignalObtainer.extractSeriesRoiRGBFromVideo(
            video_path, fs, window_length=window_length, start_time=start_time,
            forehead=True, cheeks=True, under_nose=False, play_video=False
        )

        if not isinstance(rois, dict) or len(rois) == 0:
            print(f"No ROI data extracted for {file_name}; skipping.")
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

                ippg = IppgSignalObtainer.GetRppGSeriesfromRGBSeries(
                    red, green, blue, fs, normalize=False, derivative=False,
                    bandpass=True, detrend=True, method=method
                )

                if len(ippg) == 0:
                    continue

                hr = calc.ObtainHeartRate(np.array(ippg), np.array([]), fs, method='two_peaks_periodogram')
                hr_by_roi[roi_name] = float(hr)

            if len(hr_by_roi) == 0:
                print(f"  Could not compute HR for any ROI in {file_name} with method {method}; skipping.")
                continue

            # Find best ROI using the same method
            best_roi = IppgSignalObtainer.FindBestRoiUsingHEnergyBand(rois, fs, method=method)
            hr_best = hr_by_roi.get(best_roi, None)
            if hr_best is None:
                best_roi = next(iter(hr_by_roi.keys()))
                hr_best = hr_by_roi[best_roi]

            result_row = {
                'method': method,
                'file': file_name,
                'best_roi': best_roi,
                'hr': hr_best,
                'gt_hr': gt_hr,
            }

            for roi_name in sorted(hr_by_roi.keys()):
                roi_names_union.add(roi_name)
                result_row[f'error_for_roi_{roi_name}'] = abs(hr_by_roi[roi_name] - gt_hr)

            results_rows.append(result_row)
            print(f"  {method}: best_roi={best_roi}, hr={hr_best:.2f}, gt={gt_hr:.2f}")

    if len(results_rows) == 0:
        print("No results to save.")
        return

    ordered_error_cols = [f'error_for_roi_{r}' for r in sorted(roi_names_union)]
    for result_row in results_rows:
        for col in ordered_error_cols:
            if col not in result_row:
                result_row[col] = np.nan

    out_df = pd.DataFrame(results_rows, columns=['method', 'file', 'best_roi', 'hr', 'gt_hr'] + ordered_error_cols)
    out_path = "results_all_methods_comparison.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")

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

