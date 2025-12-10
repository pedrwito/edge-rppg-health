import os
import numpy as np
import pandas as pd


def load_ubfc_ground_truth(vid_folder):
    """
    Load ground truth data from UBFC dataset folder.
    Supports both DATASET_1 (gtdump.xmp) and DATASET_2 (ground_truth.txt) formats.
    
    Args:
        vid_folder: Path to folder containing ground truth file
        
    Returns:
        tuple: (gt_trace, gt_time, gt_hr, format_used) or (None, None, None, None) if not found
    """
    gt_trace = None
    gt_time = None
    gt_hr = None
    format_used = None
    
    # Try DATASET_1 format first
    gt_filename_dataset1 = os.path.join(vid_folder, 'gtdump.xmp')
    if os.path.exists(gt_filename_dataset1):
        try:
            # MATLAB's csvread is similar to pandas.read_csv
            gt_data = pd.read_csv(gt_filename_dataset1, header=None).values
            gt_trace = gt_data[:, 3]  # 4th column (0-indexed is 3) - PPG signal
            gt_time = gt_data[:, 0]  # 1st column (0-indexed is 0) - time in milliseconds
            gt_hr = gt_data[:, 1]  # 2nd column (0-indexed is 1) - heart rate
            format_used = 'DATASET_1'
        except Exception as e:
            print(f"Error reading {gt_filename_dataset1}: {e}")
            return None, None, None, None
    else:
        # Try DATASET_2 format
        gt_filename_dataset2 = os.path.join(vid_folder, 'ground_truth.txt')
        if os.path.exists(gt_filename_dataset2):
            try:
                # MATLAB's dlmread is similar to numpy.loadtxt or pandas.read_csv with delimiter
                gt_data = np.loadtxt(gt_filename_dataset2)
                gt_trace = gt_data[0, :].T  # 1st row, transpose
                gt_time = gt_data[2, :].T  # 3rd row, transpose
                gt_hr = gt_data[1, :].T  # 2nd row, transpose
                format_used = 'DATASET_2'
            except Exception as e:
                print(f"Error reading {gt_filename_dataset2}: {e}")
                return None, None, None, None
        else:
            return None, None, None, None
    
    return gt_trace, gt_time, gt_hr, format_used


def normalize_signal(signal):
    """
    Normalize signal to zero mean and unit variance.
    
    Args:
        signal: Input signal array
        
    Returns:
        Normalized signal array
    """
    signal = signal - np.mean(signal)
    if np.std(signal) != 0:
        signal = signal / np.std(signal)
    return signal


def save_ground_truth_to_csv(gt_trace, gt_time, gt_hr, output_path, normalize=True):
    """
    Save ground truth data to CSV file.
    
    Args:
        gt_trace: PPG signal values
        gt_time: Time values (in milliseconds)
        gt_hr: Heart rate values
        output_path: Path to save CSV file
        normalize: Whether to normalize the signal before saving
    """
    # Normalize if requested
    if normalize:
        gt_trace_normalized = normalize_signal(gt_trace)
    else:
        gt_trace_normalized = gt_trace
    
    # Create DataFrame
    df = pd.DataFrame({
        'Time[ms]': gt_time,
        'PPG_Signal': gt_trace_normalized,
        'HR[bpm]': gt_hr
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False, sep=';')
    print(f"Saved ground truth to: {output_path}")


def process_ubfc_dataset(root_folder='DATASET_2/', output_dir='ubfc_ground_truth/', normalize=True):
    """
    Process UBFC dataset and save ground truth data to CSV files.
    
    Args:
        root_folder: Root folder containing UBFC dataset subdirectories
        output_dir: Directory to save output CSV files
        normalize: Whether to normalize PPG signals before saving
    """
    print(f"Processing dataset from: {root_folder}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Get list of directories (excluding '.' and '..')
    try:
        all_entries = os.listdir(root_folder)
        dirs = [d for d in all_entries if os.path.isdir(os.path.join(root_folder, d)) 
                and d not in ['.', '..', 'desktop.ini']]
        dirs.sort()  # Sort to ensure consistent order
    except FileNotFoundError:
        print(f"Error: Dataset root folder '{root_folder}' not found.")
        return
    
    if not dirs:
        print(f"No subdirectories found in '{root_folder}'. Please check the dataset structure.")
        return
    
    # Iterate through all directories
    for i, dir_name in enumerate(dirs):
        vid_folder = os.path.join(root_folder, dir_name)
        print(f"\n--- Processing folder {i+1}/{len(dirs)}: {dir_name} ---")
        
        # Load ground truth
        gt_trace, gt_time, gt_hr, format_used = load_ubfc_ground_truth(vid_folder)
        
        if gt_trace is None:
            print(f"Warning: No ground truth file found in {vid_folder} (checked gtdump.xmp and ground_truth.txt)")
            continue
        
        print(f"  Format: {format_used}")
        print(f"  Number of PPG signal values: {len(gt_trace)}")
        
        # Convert time to milliseconds if needed (DATASET_2 might need conversion)
        if format_used == 'DATASET_2':
            # If time is in seconds, convert to milliseconds
            if gt_time[0] < 1000:  # Likely in seconds
                gt_time = gt_time * 1000
        
        print(f"  Length of ground truth signal: {gt_time[-1]:.2f} milliseconds ({gt_time[-1]/1000:.2f} seconds)")
        
        # Save to CSV
        output_filename = f"{dir_name}_ground_truth.csv"
        output_path = os.path.join(output_dir, output_filename)
        save_ground_truth_to_csv(gt_trace, gt_time, gt_hr, output_path, normalize=normalize)
    
    print("\n--- Processing complete! ---")
    print(f"All ground truth files saved to: {output_dir}")


if __name__ == "__main__":
    base_folder = 'UBFC_DATASET_MERGED'
    
    # Process DATASET_1 format
    dataset1_path = os.path.join(base_folder, 'DATASET_1')
    if os.path.exists(dataset1_path):
        print("=" * 60)
        print("Processing DATASET_1")
        print("=" * 60)
        process_ubfc_dataset(dataset1_path, 'ubfc_ground_truth/DATASET_1/', normalize=True)
    
    # Process DATASET_2 format
    dataset2_path = os.path.join(base_folder, 'DATASET_2')
    if os.path.exists(dataset2_path):
        print("\n" + "=" * 60)
        print("Processing DATASET_2")
        print("=" * 60)
        process_ubfc_dataset(dataset2_path, 'ubfc_ground_truth/DATASET_2/', normalize=True)

