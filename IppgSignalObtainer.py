import cv2
import numpy as np
import mediapipe as mp
from scipy import signal
import os

from Methods.pos import POS_WANG
from Methods.green import green
from Methods.ica import ICA
from Methods.chrom import CHROM
import Tools.utils as utils


class IppgSignalObtainer:
    """
    A class for extracting iPPG (imaging photoplethysmography) signals from video files.
    
    This class provides methods to extract RGB time series from facial regions of interest
    and calculate heart rate using various iPPG algorithms.
    """

    @staticmethod
    def extractSeriesRoiRGBFromVideo(video_path, fs, window_length=30, start_time=0, forehead=True, cheeks=False, under_nose=False, play_video=False):
        """
        Extract RGB time series from selected facial regions of interest in a video.
        
        Parameters:
        -----------
        video_path : str
            Path to the video file
        fs : float
            Sampling frequency in Hz
        window_length : float, optional
            Length of the analysis window in seconds (default: 30)
        start_time : float, optional
            Start time for analysis in seconds (default: 0)
        forehead : bool, optional
            Whether to extract RGB series from forehead region (default: True)
        cheeks : bool, optional
            Whether to extract RGB series from cheek regions (default: False)
        under_nose : bool, optional
            Whether to extract RGB series from under nose region (default: False)
        play_video : bool, optional
            If True, shows the video with ROI overlays while processing (default: False)
            
        Returns:
        --------
        dict
            Dictionary containing RGB time series for each selected region.
            Structure: {
                'region_name': {
                    'red': [list of red channel values],
                    'green': [list of green channel values], 
                    'blue': [list of blue channel values]
                }
            }
            Possible region names: 'forehead', 'left_cheek', 'right_cheek', 'under_nose'
            
        Raises:
        -------
        FileNotFoundError
            If the video file doesn't exist
        ValueError
            If parameters are invalid or video cannot be opened
        """
        # Input validation
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if fs <= 0:
            raise ValueError("Sampling frequency must be positive")
        if window_length <= 0:
            raise ValueError("Window length must be positive")
        if start_time < 0:
            raise ValueError("Start time must be non-negative")
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Initialize dictionaries to store time series for each region
        rgb_series = {}
        
        if forehead:
            rgb_series['forehead'] = {'red': [], 'green': [], 'blue': []}
        if cheeks:
            rgb_series['left_cheek'] = {'red': [], 'green': [], 'blue': []}
            rgb_series['right_cheek'] = {'red': [], 'green': [], 'blue': []}
        if under_nose:
            rgb_series['under_nose'] = {'red': [], 'green': [], 'blue': []}

        # Initialize Mediapipe FaceMesh
        
        #Code based on mediapipe doc 
        #https://mediapipe.readthedocs.io/en/latest/solutions/face_mesh.html#:~:text=The%20Face%20Landmark%20Model%20performs,weak%20perspective%20projection%20camera%20model.
        
        mp_face_mesh = mp.solutions.face_mesh
        
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

            # Flag to limit the number of frames
            frame_limit = fs * (start_time + window_length)
            current_frame = 0
            
            while True and current_frame < frame_limit:
                ret, frame = cap.read()
                if not ret:
                    # If the video has reached the end, break out of the loop
                    break

                # Convert the frame to RGB for Mediapipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:

                        
                        # Extract landmarks only for requested regions
                        forehead_landmarks = None
                        left_cheek_landmarks = None
                        right_cheek_landmarks = None
                        under_nose_landmarks = None
                        
                        if forehead:
                            forehead_landmarks = [
                                face_landmarks.landmark[10],  #Top forehead
                                face_landmarks.landmark[105],# Left forehead
                                face_landmarks.landmark[334]# Right forehead
                            ]
                        
                        if cheeks:
                            left_cheek_landmarks = [
                                face_landmarks.landmark[116],  # Left cheek center
                                face_landmarks.landmark[117],  # Left cheek
                                face_landmarks.landmark[118],  # Left cheek
                                face_landmarks.landmark[119],  # Left cheek
                                face_landmarks.landmark[120],  # Left cheek
                                face_landmarks.landmark[121],  # Left cheek
                                face_landmarks.landmark[126],  # Left cheek
                                face_landmarks.landmark[142],  # Left cheek
                            ]
                            
                            right_cheek_landmarks = [
                                face_landmarks.landmark[345],  # Right cheek center
                                face_landmarks.landmark[346],  # Right cheek
                                face_landmarks.landmark[347],  # Right cheek
                                face_landmarks.landmark[348],  # Right cheek
                                face_landmarks.landmark[349],  # Right cheek
                                face_landmarks.landmark[350],  # Right cheek
                                face_landmarks.landmark[355],  # Right cheek
                                face_landmarks.landmark[371],  # Right cheek
                            ]
                        
                        if under_nose:
                            under_nose_landmarks = [
                                face_landmarks.landmark[2],    # Chin tip (upper bound)
                                face_landmarks.landmark[164],  # Under nose center
                                face_landmarks.landmark[165],  # Under nose
                                face_landmarks.landmark[167],  # Under nose
                                face_landmarks.landmark[169],  # Under nose
                                face_landmarks.landmark[170],  # Under nose
                                face_landmarks.landmark[171],  # Under nose
                                face_landmarks.landmark[175],  # Under nose
                            ]
                        
                        #For denormalization of pixel coordinates, we should multiply x coordinate by width and y coordinate by height.
                        
                        # Get frame dimensions
                        h, w, _ = frame.shape
                        
                        def extract_roi_coordinates(landmarks, top_margin=4, bottom_margin=4, left_margin=4, right_margin=4):
                            """Helper to extract ROI coordinates; supports per-side margins.
                            If margin is provided, it overrides top/bottom/left/right equally."""
                            points = np.array([(int(landmark.x * w), int(landmark.y * h)) for landmark in landmarks])

                            
                            # Calculate bounding box with side-specific margins
                            x_min = np.min(points[:, 0]) + left_margin
                            x_max = np.max(points[:, 0]) - right_margin
                            y_min = np.min(points[:, 1]) + top_margin
                            y_max = np.max(points[:, 1]) - bottom_margin
                            
                            # Bounds checking
                            x_min = max(0, x_min)
                            x_max = min(w, x_max)
                            y_min = max(0, y_min)
                            y_max = min(h, y_max)
                            
                            return x_min, x_max, y_min, y_max
                        
                        # Process each selected region
                        rois = {}
                        
                        if forehead and forehead_landmarks is not None: 
                            # Use the same robust bounding-box helper as other regions to handle head tilt/rotation
                            x_min, x_max, y_min, y_max = extract_roi_coordinates(forehead_landmarks, top_margin=4, bottom_margin=6, left_margin=4, right_margin=4)    
                            if y_max > y_min and x_max > x_min:
                                rois['forehead'] = rgb_frame[y_min:y_max, x_min:x_max]
                                if play_video:
                                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                                    cv2.putText(frame, 'forehead', (x_min, max(0, y_min-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                        
                        if cheeks and left_cheek_landmarks is not None and right_cheek_landmarks is not None:
                            # Left cheek ROI
                            left_x_min, left_x_max, left_y_min, left_y_max = extract_roi_coordinates(left_cheek_landmarks, top_margin=4, bottom_margin=4, left_margin=4, right_margin=4)
                            if left_y_max > left_y_min and left_x_max > left_x_min:
                                rois['left_cheek'] = rgb_frame[left_y_min:left_y_max, left_x_min:left_x_max]
                                if play_video:
                                    cv2.rectangle(frame, (left_x_min, left_y_min), (left_x_max, left_y_max), (255, 0, 0), 2)
                                    cv2.putText(frame, 'left_cheek', (left_x_min, max(0, left_y_min-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
                            
                            # Right cheek ROI
                            right_x_min, right_x_max, right_y_min, right_y_max = extract_roi_coordinates(right_cheek_landmarks, top_margin=4, bottom_margin=4, left_margin=4, right_margin=4)
                            if right_y_max > right_y_min and right_x_max > right_x_min:
                                rois['right_cheek'] = rgb_frame[right_y_min:right_y_max, right_x_min:right_x_max]
                                if play_video:
                                    cv2.rectangle(frame, (right_x_min, right_y_min), (right_x_max, right_y_max), (0, 0, 255), 2)
                                    cv2.putText(frame, 'right_cheek', (right_x_min, max(0, right_y_min-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                        
                        if under_nose and under_nose_landmarks is not None:
                            # Under nose ROI
                            nose_x_min, nose_x_max, nose_y_min, nose_y_max = extract_roi_coordinates(under_nose_landmarks, top_margin=4, bottom_margin=4, left_margin=4, right_margin=4)
                            if nose_y_max > nose_y_min and nose_x_max > nose_x_min:
                                rois['under_nose'] = rgb_frame[nose_y_min:nose_y_max, nose_x_min:nose_x_max]
                                if play_video:
                                    cv2.rectangle(frame, (nose_x_min, nose_y_min), (nose_x_max, nose_y_max), (0, 255, 255), 2)
                                    cv2.putText(frame, 'under_nose', (nose_x_min, max(0, nose_y_min-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)

                        
                    current_frame += 1

                    if current_frame > (start_time)*fs and current_frame < fs*(window_length + start_time):
                        # Calculate average intensity for each color channel for each ROI
                        for region_name, roi in rois.items():
                            if roi.size > 0:  # Ensure ROI is not empty
                                red_avg = np.mean(roi[:, :, 0])
                                green_avg = np.mean(roi[:, :, 1])
                                blue_avg = np.mean(roi[:, :, 2])

                                # Append the average intensities to the time series lists
                                rgb_series[region_name]['red'].append(red_avg)
                                rgb_series[region_name]['green'].append(green_avg)
                                rgb_series[region_name]['blue'].append(blue_avg)

                    # If visualization is enabled, show the frame with overlays
                    if play_video:
                        cv2.imshow('ROI Visualization', frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27 or key == ord('q'):
                            # ESC or 'q' pressed: stop early
                            break


            # Release the video capture object
            cap.release()
            if play_video:
                cv2.destroyAllWindows()

        return rgb_series


    @staticmethod
    def GetRppGSeriesfromRGBSeries(red_series, green_series, blue_series, fs, normalize=False, 
                      derivative=False, bandpass=True, detrend=True, method='pos'):

        seriesRGB = np.asarray([red_series, green_series, blue_series])
        # Apply the selected iPPG method
        if method.lower() == 'pos':
            series = POS_WANG(seriesRGB, fs, derivative=derivative, 
                            normalize=normalize, bandpass=bandpass, detrend=detrend)
        elif method.lower() == 'chrom':
            series = CHROM(seriesRGB, fs, normalize=normalize, detrend=detrend, 
                         bandpass=bandpass, derivative=derivative)
        elif method.lower() == 'ica':
            series = ICA(seriesRGB, fs, normalize=normalize, detrending=detrend, 
                       bandpass=bandpass, derivate=derivative)
        elif method.lower() == 'green':
            series = green(green_series, fs, normalize=normalize, detrend=detrend, 
                         bandpass=bandpass, derivative=derivative)

        return series

    @staticmethod
    def FindBestRoiUsingHEnergyBand(series_for_roi, fs, normalize=False, 
                      derivative=False, bandpass=True, detrend=True, method='pos'):
                              
        best_energy = 0
        for roi in series_for_roi:
            roi_rgb = series_for_roi[roi]
            ippg_series = IppgSignalObtainer.GetRppGSeriesfromRGBSeries(roi_rgb['red'], roi_rgb['green'], roi_rgb['blue'], fs, normalize, derivative, bandpass, detrend, method)
            freqs, power = utils.spectogram(ippg_series, fs, lower_band=0.5, upper_band=4)
            #calculate energy in the 0.5-4 Hz band
            energy = np.sum(power) * (freqs[1] - freqs[0]) #area under the curve, easy calculation. Could replace by scipy.trapezoid
            print(f"ROI: {roi} with energy {energy}")
            if energy > best_energy:
                best_energy = energy
                best_roi = roi
                print(f"Best ROI: {roi} with energy {energy}")
                
        return best_roi


    @staticmethod
    def extractFullFaceSkinRGBFromVideo(video_path, fs, window_length=30, start_time=0,
                                        play_video=False, dilation_px=2, skin_filter=True):
        """
        Extract RGB time series from the full facial skin area, automatically excluding
        eyes, eyebrows, lips/mouth, and most hair.

        Parameters:
        - video_path: path to video file
        - fs: sampling frequency (Hz)
        - window_length: seconds to analyze
        - start_time: starting second
        - play_video: if True, show visualization
        - dilation_px: extra pixels to dilate the exclusion regions
        - skin_filter: if True, apply a YCrCb skin-color filter within the face oval

        Returns:
        - dict with one key 'face_skin' holding red/green/blue lists
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if fs <= 0:
            raise ValueError("Sampling frequency must be positive")
        if window_length <= 0:
            raise ValueError("Window length must be positive")
        if start_time < 0:
            raise ValueError("Start time must be non-negative")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        rgb_series = {'face_skin': {'red': [], 'green': [], 'blue': []}}

        # Landmark index sets (MediaPipe FaceMesh topology)
        LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
        LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
        LIPS_OUTER = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,185,
                      40,39,37,0,267,269,270,409,415,310,311,312,13,82,81,42,183,78]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, dilation_px*2+1), max(1, dilation_px*2+1)))

        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

            frame_limit = fs * (start_time + window_length)
            current_frame = 0

            while True and current_frame < frame_limit:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                mask = None
                overlay_poly_pts = []

                if results.multi_face_landmarks:
                    for fl in results.multi_face_landmarks:
                        h, w, _ = frame.shape

                        def pts_from_indices(indices):
                            return np.array([[int(fl.landmark[i].x * w), int(fl.landmark[i].y * h)] for i in indices], dtype=np.int32)

                        # Start with pyVHR-like full-face convex hull mask (more robust than fixed oval)
                        mask = np.zeros((h, w), dtype=np.uint8)
                        all_pts = np.array([[int(fl.landmark[i].x * w), int(fl.landmark[i].y * h)] for i in range(len(fl.landmark))], dtype=np.int32)
                        face_hull = cv2.convexHull(all_pts)
                        cv2.fillConvexPoly(mask, face_hull, 255)
                        overlay_poly_pts.append(face_hull)

                        # Build exclusion mask for eyes, brows, lips
                        exclude = np.zeros((h, w), dtype=np.uint8)
                        # Compute convex hulls of exclusion regions (pyVHR approach)
                        left_eye_poly = cv2.convexHull(pts_from_indices(LEFT_EYE))
                        right_eye_poly = cv2.convexHull(pts_from_indices(RIGHT_EYE))
                        left_brow_poly = cv2.convexHull(pts_from_indices(LEFT_BROW))
                        right_brow_poly = cv2.convexHull(pts_from_indices(RIGHT_BROW))
                        lips_poly = cv2.convexHull(pts_from_indices(LIPS_OUTER))

                        for poly in [left_eye_poly, right_eye_poly, left_brow_poly, right_brow_poly, lips_poly]:
                            cv2.fillConvexPoly(exclude, poly, 255)

                        if dilation_px > 0:
                            exclude = cv2.dilate(exclude, kernel, iterations=1)

                        # Remove excluded regions from face oval
                        mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude))

                        # Optional color filter to suppress hair/non-skin
                        if skin_filter:
                            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                            lower = np.array([0, 133, 77], dtype=np.uint8)
                            upper = np.array([255, 173, 127], dtype=np.uint8)
                            color_mask = cv2.inRange(ycrcb, lower, upper)
                            mask = cv2.bitwise_and(mask, color_mask)

                        # Light cleanup
                        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                current_frame += 1

                if mask is not None and current_frame > (start_time)*fs and current_frame < fs*(window_length + start_time):
                    valid = mask > 0
                    if np.any(valid):
                        r_mean = float(np.mean(rgb_frame[:, :, 0][valid]))
                        g_mean = float(np.mean(rgb_frame[:, :, 1][valid]))
                        b_mean = float(np.mean(rgb_frame[:, :, 2][valid]))
                        rgb_series['face_skin']['red'].append(r_mean)
                        rgb_series['face_skin']['green'].append(g_mean)
                        rgb_series['face_skin']['blue'].append(b_mean)

                if play_video:
                    vis = frame.copy()
                    if mask is not None:
                        # Overlay the exact skin area being used (semi-transparent green)
                        overlay = vis.copy()
                        overlay[mask > 0] = (0, 255, 0)
                        vis = cv2.addWeighted(overlay, 0.3, vis, 0.7, 0)
                        # Draw mask boundary for clarity
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)
                    cv2.imshow('Face Skin ROI', vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord('q'):
                        break

        cap.release()
        if play_video:
            cv2.destroyAllWindows()

        return rgb_series
