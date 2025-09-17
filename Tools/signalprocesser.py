import Tools.utils as utils
import numpy as np
import scipy.signal as signal
from scipy import sparse
from scipy.interpolate import interp1d

class signalprocesser():

    def bandpass(series, fs, color = "orange", order=3, lowcut=0.5, highcut=4, plot = False):
        b, a = signal.butter(order, [lowcut, highcut], btype='band', analog=False, fs=fs)
        filtered_series = signal.filtfilt(b, a, series)
        if plot:
            utils.plotSignal(filtered_series,fs, color, "Bandpass Filter")

        return filtered_series

    def derivativeFilter(series, fs,  color = "orange", L = 1, plot = False): #L represents the order of the derivative
        L = 1
        h = np.zeros(2*L + 1)
        h[0] = 1
        h[-1] = -1
        h = h*fs / (2*L)
        filtered_series = np.convolve(series, h, 'same')

        if plot:
          utils.plotSignal(filtered_series,fs, color, "Derivative Filter")

        return filtered_series


    def normalize(series, fs,  color = "orange", plot = False):
        mean = np.mean(series)
        std = np.std(series)
        normalized_series = series - mean
        normalized_series = normalized_series / std

        if plot:
          utils.plotSignal(normalized_series,fs, color, "Normalized signal")

        return normalized_series


    def detrend(series, lambda_value = 100, fs = 30, method = 'mcduff', color = 'orange', plot= False):

        if method == 'mcduff':
            series_length = series.shape[0]
            # observation matrix
            H = np.identity(series_length)
            ones = np.ones(series_length)
            minus_twos = -2 * np.ones(series_length)
            diags_data = np.array([ones, minus_twos, ones])
            diags_index = np.array([0, 1, 2])
            D = sparse.spdiags(diags_data, diags_index,
                        (series_length - 2), series_length).toarray()
            detrended_series = np.dot(
                (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), series)

        if method == 'haan':
            #Use detrendign method proposed by de Haan, G. and Van Leest, A. (2014).
            # Improved motion robustness of remote-PPG by using the blood volume pulse signature, Physiological measurement 35(9): 1913.
            #L is assumed as 1 second
            L = L * fs

            detrended_series = np.zeros(len(signal))

            for i in range(len(signal)):
                if i < L:
                    L_aux = i
                else:
                    L_aux = L
                m = np.sum(signal[i-L_aux:i])
                detrended_series[i]= signal[i] - m / m

        if plot:
            utils.plotSignal(detrended_series, fs, color, "Detrended signal")

        return detrended_series

    def ScaleMinMax(series):
        minVal = min(series)
        maxVal = max(series)

        # Step 2: Use map and lambda to scale each value
        scaledSeries = list(map(lambda x: (x - minVal) / (maxVal - minVal), series))
        return scaledSeries

    def process_serie_IBI_absolute(serie, peaks = [], int_min = 0.24, int_max = 2):
      serie_aux = []
      i_del = []
      if len(peaks) > 0:
        peaks_aux = [peaks[0]]
      n_mini = 0
      for i in range(len(serie)):
          interval = serie[i]
          if len(peaks) > 0:
            peak = peaks[i+1]
          if interval < int_min:
            if 0 < i < (len(serie) - 1):
              n_mini = n_mini + 1
              interval_prev = serie[i-1]
              interval_next = serie[i+1]
              if interval_prev < interval_next:
                if len(serie_aux) > 0:
                  serie_aux[-1] = interval_prev + interval
                  if len(peaks) > 0:
                    peaks_aux[-1] = peak
                else:
                  n_mini = n_mini - 1
                  serie_aux.append(interval_prev + interval)
                  if len(peaks) > 0:
                    peaks_aux.append(peak)
              else:
                  serie[i+1] = interval_next + interval
          elif interval > int_max:
            i_del.append(i-n_mini)
          else:
            serie_aux.append(interval)
            if len(peaks) > 0:
              peaks_aux.append(peak)
      return serie_aux, peaks_aux, i_del


    def process_serie_IBI_relative(serie_, r_peaks, i_del = [], k_min = 1/3, k_max = 3):
      int_med = np.median(serie_)
      serie = serie_.copy()
      serie_final = []
      r_peaks_final = []
      if len(r_peaks) > 0:
        r_peaks_final.append(r_peaks[0])
      n_mini = 0
      for i in range(len(serie)):
        interval = serie[i]
        if len(r_peaks) > 0:
          r_peak = r_peaks[i+1]
        if interval < (k_min*int_med):
          if 0 < i < (len(serie) - 1):
            n_mini = n_mini + 1
            for j in range(len(i_del)):
              if i < i_del[j]:
                i_del[j] = i_del[j] - 1
            interval_prev = serie[i-1]
            interval_next = serie[i+1]
            if interval_prev < interval_next:
              if len(serie_final) > 0:
                  serie_final[-1] = interval_prev + interval
                  if len(r_peaks) > 0:
                    r_peaks_final[-1] = r_peak
              else:
                  n_mini = n_mini - 1
                  serie_final.append(interval_prev + interval)
                  if len(r_peaks) > 0:
                    r_peaks_final.append(r_peak)
            else:
              serie[i+1] = interval_next + interval
        elif interval > (k_max*int_med):
            i_del.append(i-n_mini)
        else:
            serie_final.append(interval)
            if len(r_peaks) > 0:
              r_peaks_final.append(r_peaks[i+1])
      i_del = np.array(i_del)
      return serie_final, r_peaks_final, i_del


    def linear_interpolation(time, values, new_fs):

      # Ensure the time series is sorted by time
      num_additional_samples = int(new_fs*time[-1] - len(values)) + 5
      time = np.array(time)
      values = np.array(values)

      # Create an interpolation function
      interpolation_function = interp1d(time, values, kind='linear', fill_value="extrapolate")

      # Generate new time points that are evenly spaced between the min and max of the original time points
      new_time = np.linspace(time[0], time[-1], num_additional_samples)

      # Combine original and new time points, then sort them
      combined_time = np.unique(np.concatenate((time, new_time)))

      # Interpolate values at the combined time points
      combined_values = interpolation_function(combined_time)

      return combined_time, combined_values

    def cubic_interpolation(time, values, new_fs):

      steps = 1 / new_fs
      f_ippg = interp1d(time, values, kind='cubic')

      # now we can sample from interpolation function
      interpolated_time = np.arange(np.min(time), np.max(time), steps)
      interpolated_values = f_ippg(interpolated_time)

      return interpolated_time, interpolated_values

