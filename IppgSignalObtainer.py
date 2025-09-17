import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
#from Tools.jadeR import jadeR
from scipy import signal
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, median_absolute_error

from Methods.pos import POS_WANG
#from Methods.green import green
#from Methods.ica import ICA
#from Methods.chrom import CHROM
import Tools.utils as utils
from Tools.signalprocesser import signalprocesser

class IppgSignalObtainer:

    @staticmethod
    def extractSeriesRoiRGBFromVideo(video_path, fs, window_lenght = 30, start_time = 0):
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Initialize lists to store time series
        red_series = []
        green_series = []
        blue_series = []

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
            frame_limit = fs*(start_time + window_lenght)
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

                        
                        # Extract forehead landmarks
                        forehead_landmarks = [
                            face_landmarks.landmark[10],  #Top forehead
                            face_landmarks.landmark[105],# Left forehead
                            face_landmarks.landmark[334]# Right forehead
                        ]
                        
                        #For denormalization of pixel coordinates, we should multiply x coordinate by width and y coordinate by height.
                        
                        
                        # Convert forehead landmarks to pixel coordinates
                        h, w, _ = frame.shape
                        
                        forehead_points = np.array([(int(l.x * w), int(l.y * h)) for l in forehead_landmarks])
                        margin = 4
 
                            
                        forehead_top = forehead_points[0][1]- 2*margin# Top of forehead
                        forehead_left = min(forehead_points[1][0], forehead_points[2][0]) + 2*margin# Leftmost point    
                        forehead_right = max(forehead_points[2][0], forehead_points[1][0]) - 2*margin# Rightmost point
                        forehead_bottom = min(forehead_points[1][1], forehead_points[2][1])+ 2*margin  # Bottom of forehead

                        roi = rgb_frame[forehead_top:forehead_bottom, forehead_left:forehead_right]

                        
                    current_frame += 1

                    if current_frame > (start_time)*fs and current_frame < fs*(window_lenght + start_time):
                    # Calculate average intensity for each color channel
                        red_avg = np.mean(roi[:, :, 0])
                        green_avg = np.mean(roi[:, :, 1])
                        blue_avg = np.mean(roi[:, :, 2])

                        # Append the average intensities to the time series lists
                        red_series.append(red_avg)
                        green_series.append(green_avg)
                        blue_series.append(blue_avg)


            # Release the video capture object
            cap.release()

        return red_series, green_series, blue_series

    
    @staticmethod
    def GetHRFromVideo(video_path, fs, start_time = 0, window_lenght = 30, normalize = False, derivative = False, bandpass = True, detrend = True, method = 'pos', calculationMethod = 'freq', uma = False, play_video = False, plot = False):
        
        seriesRGB = IppgSignalObtainer.extractSeriesRoiRGBFromVideo(video_path, fs, start_time= start_time, window_lenght= window_lenght)
        return IppgSignalObtainer.calculate_HR(seriesRGB, fs, derivative, normalize, bandpass, detrend, method, calculationMethod)

    @staticmethod
    def calculate_HR(seriesRGB, fs, derivative, normalize, bandpass, detrend, method, calculationMethod):
        if method == 'pos':
            series = POS_WANG(np.asarray(seriesRGB), fs, derivative= derivative, normalize = normalize, bandpass = bandpass, detrend= detrend)
            
        #Only because there is only 1 method, needs to be changed after    
        else:
            series = POS_WANG(np.asarray(seriesRGB), fs, derivative= derivative, normalize = normalize, bandpass = bandpass, detrend= detrend)
        
        #Get HR from peak frequency in the bands of the HR.
        if calculationMethod == 'freq':
            freqs, fftSeriesICA = utils.FFT(series, fs)
            HR = utils.getPeakFrequencyFFT(freqs, fftSeriesICA) * 60 
            
        #Get HR from peak detection in time domain.    
        elif calculationMethod == 'time':
            lenght = len(series)
            h_max = 1*np.mean(series)
            peaks, _  = signal.find_peaks(series, height = h_max, distance = (fs*0.25)) # 0.25 because we assume max HR of 240 BPM
            HR = len(peaks)/(lenght/fs)*60
        
        return series, HR

    @staticmethod
    def GetRGBSeries(red_series, green_series, blue_series):
        return red_series, green_series, blue_series


    @staticmethod
    def GetHRUsingRoiRGBSeries(red_series, green_series, blue_series, fs, normalize=False, derivative=False, bandpass=True,
                       detrend=True, method='pos', calculationMethod='freq'):

        seriesRGB = IppgSignalObtainer.GetRGBSeries(red_series, green_series, blue_series)
        return IppgSignalObtainer.calculate_HR(seriesRGB, fs, derivative, normalize, bandpass, detrend, method, calculationMethod)
        
       
       

        
        
                        
        

        
        
        
