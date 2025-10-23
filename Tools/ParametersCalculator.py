import numpy as np
from scipy import signal
from scipy.integrate import trapezoid
from scipy.stats import mode
from Tools.signalprocesser  import SignalProcessor
import Tools.utils as u
import math
import matplotlib.pyplot as plt

class ParametersCalculator:
    """
    A class to calculate various physiological parameters from time series data (rPPG).

    This class provides methods to calculate heart rate, respiratory rate,
    heart rate variability, and stress indices from rPPG signals.
    """

    def GetParameters(self, signal: np.ndarray, fs: int, interpolation: str = "cubic", 
                     fs_interpolation: int = 4) -> dict:
        """
        Calculate physiological parameters from am rPPG signal.

        Args:
            signal (np.ndarray): Input signal time series
            fs (float): Sampling frequency in Hz
            interpolation (str, optional): Interpolation method ("cubic" or "linear"). Defaults to "cubic"
            fs_interpolation (int, optional): Interpolation sampling frequency. Defaults to 4
            plot (bool, optional): Whether to plot intermediate results. Defaults to False

        Returns:
            dict: Dictionary containing calculated parameters:
                - HR: Heart rate in BPM
                - RR: Respiratory rate in breaths/minute
                - HRV parameters (SDNN, RMSSD, etc.)
                - Stress index
        """

        ibi_series, start, peaks = self.GetIBISeries(signal, fs)
        parameters = {}

        #Processing IBI series by removing too short IBI (due to incorrect detections) using absolute threshold and then relative threshhold (based on median IBI)
        ibi_series, peaks_aux, i_del = SignalProcessor.process_serie_IBI_absolute(ibi_series, peaks = peaks, int_min = 0.3, int_max = 2)
        ibi_series, peaks_aux, i_del  = SignalProcessor.process_serie_IBI_relative(ibi_series, peaks_aux, i_del = [], k_min = 1/2.5, k_max = 2.5)

        parameters["HR"] = self.ObtainHeartRate(signal, ibi_series, fs, method = 'two_peaks_periodogram')

        time_signal = np.linspace(0,len(signal)/fs, len(signal))

        #filtered after finding frequency dominating signal---------------------------------------------------
        signal_filt = SignalProcessor.bandpass(signal, fs, lowcut = parameters["HR"]/60 - 0.5 , highcut = parameters["HR"]/60 + 0.5)

        ibi_series_filt, start_filt, peaks_filt = self.GetIBISeries(signal_filt, fs)

        #Processing IBI series by removing too short IBI (due to incorrect detections) using absolute threshold and then relative threshhold (based on median IBI)

        ibi_series_filt, peaks_aux_filt, i_del_filt = SignalProcessor.process_serie_IBI_absolute(ibi_series_filt, peaks = peaks_filt, int_min = 0.3, int_max = 2)
        ibi_series_filt, peaks_aux_filt, i_del_filt  = SignalProcessor.process_serie_IBI_relative(ibi_series_filt, peaks_aux_filt, i_del = [], k_min = 1/2.5, k_max = 2.5)

        def refine_peak_locations(original_signal, peak_indices, window_size=5):
          """
          Refines peak locations found in filtered signal by looking for true peaks in original signal.

          Args:
              original_signal (array): The original unfiltered signal
              peak_indices (array): Indices of peaks found in filtered signal
              window_size (int): Size of window to search around each filtered peak (default=5)

          Returns:
              array: Refined peak indices in the original signal
          """
          refined_peaks = []

          for peak_idx in peak_indices:
              # Define search window around the filtered peak
              start_idx = max(0, peak_idx - window_size)
              end_idx = min(len(original_signal), peak_idx + window_size + 1)

              # Find the maximum point in the original signal within this window
              window = original_signal[start_idx:end_idx]
              max_idx = np.argmax(window)
              refined_peak = start_idx + max_idx

              refined_peaks.append(refined_peak)

          return np.array(refined_peaks)


        peaks_filt_new = refine_peak_locations(signal, peaks_filt, window_size = int(0.2*fs))

        ibi_series_filt_2 = np.diff(time_signal[peaks_filt_new])

        t_ibi_filtered_2 = np.cumsum(ibi_series_filt_2)
        
        #calculation of interpolated IBI series for frequency based parameters of HRV and Respiratory Rate(RR)
        if interpolation == "linear":
          interpolated_time_filtered_2, interpolated_series_filtered_2 = SignalProcessor.linear_interpolation(t_ibi_filtered_2, ibi_series_filt_2, fs_interpolation)


        elif interpolation == "cubic":
          interpolated_time_filtered_2, interpolated_series_filtered_2 = SignalProcessor.cubic_interpolation(t_ibi_filtered_2, ibi_series_filt_2, fs_interpolation)

        parameters["RR"] = self.ObtainRespiratoryRate(interpolated_series_filtered_2, fs = fs_interpolation, method = "freq")

        results_HRV, fxx, pxx = self.HeartRateVariability(ibi_series, interpolated_time_filtered_2, fs = fs_interpolation)

        parameters.update(results_HRV)

        parameters["stress"] = self.GetStress(ibi_series)
        
        return parameters


    def ObtainHeartRate(self, senal: np.ndarray, series: np.ndarray, fs: float, 
                       method: str = 'freq', peaks: np.ndarray = None) -> float:
        """
        Calculate heart rate from a signal using various methods.

        Args:
            senal (np.ndarray): Input signal time series
            series (np.ndarray): IBI series
            fs (float): Sampling frequency in Hz
            method (str, optional): Method to use ('freq', 'time', 'two_peaks_fft', 
                                  'periodogram', 'two_peaks_periodogram'). Defaults to 'freq'
            peaks (np.ndarray, optional): Pre-computed peaks. Defaults to None

        Returns:
            float: Heart rate in beats per minute (BPM)
        """

        if method == 'freq':
            freqs, fftSeries = u.FFT(senal, fs)
            fftSeries = np.abs(fftSeries)**2/len(fftSeries)
            HR = u.getPeakFrequencyFFT(freqs, fftSeries) * 60

        #Get HR from peak detection in time domain.
        elif method == 'time':

            HR = 60/np.median(series)

        elif method == 'two_peaks_fft':

              freqs, fftSeries = u.FFT(senal, fs)
              fftSeries = np.abs(fftSeries)**2/len(fftSeries)

              # Get the corresponding frequencies

              highest_freq, second_highest_freq = 10, 10
              peaks = self.two_highest_peaks_fft(fftSeries, 30, separation = 0.1)

              if len(peaks) == 2:

                highest_freq, second_highest_freq = freqs[peaks]

              if len(peaks) == 2 and highest_freq <1.15 and (highest_freq + 0.1 < second_highest_freq) and fftSeries[peaks[0]] * 0.60 <  fftSeries[peaks[1]]:

                HR = second_highest_freq * 60

              else:
                HR = highest_freq * 60

        elif method == 'periodogram':
            fxx, pxx = u.spectogram(senal, fs)
            HR = u.getPeakFrequencyFFT(fxx, pxx) * 60


        elif method =='two_peaks_periodogram':
            fxx, pxx = u.spectogram(senal, fs)
            
            # Mark the peaks
            highest_freq, second_highest_freq = 10, 10
            peaks = self.two_highest_peaks_fft(pxx, 30, separation=0.1)
            #if len(peaks) > 0:
                #plt.plot(fxx[peaks], pxx[peaks], 'ro', label='Peaks')
            #plt.legend()
            #plt.show()

            if len(peaks) == 2:
              highest_freq, second_highest_freq = fxx[peaks]

            elif len(peaks)==1:
                highest_freq = fxx[peaks[0]]

            if len(peaks) == 2 and highest_freq <1.15 and (highest_freq + 0.1 < second_highest_freq) and pxx[peaks[0]] * 0.60 <  pxx[peaks[1]]:

              HR = second_highest_freq * 60

            else:
                HR = highest_freq * 60

        return HR

    def ObtainRespiratoryRate(self, ibi: np.ndarray, interpolate: str = None, 
                             lowcut: float = 0.15, highcut: float = 0.4, 
                             method: str = "freq", fs: float = 4) -> float:
        """
        Calculate respiratory rate from IBI series.

        Args:
            ibi (np.ndarray): Inter-beat interval series
            interpolate (str, optional): Interpolation method. Defaults to None
            lowcut (float, optional): Lower frequency cutoff in Hz. Defaults to 0.15
            highcut (float, optional): Upper frequency cutoff in Hz. Defaults to 0.4
            method (str, optional): Method to use ("freq" or "time"). Defaults to "freq"
            fs (float, optional): Sampling frequency in Hz. Defaults to 4

        Returns:
                - RR (float): Respiratory rate in breaths/minute
                
        """
        
        time = np.cumsum(ibi)
        if interpolate:

            if interpolate == "linear":
                ibi = SignalProcessor.linear_interpolation(time, ibi, fs)
            elif interpolate == "cubic":
                ibi = SignalProcessor.cubic_interpolation(time, ibi, fs)

        if method == "time":
            time = np.cumsum(ibi)
            filtered = SignalProcessor.bandpass(ibi, fs, lowcut = lowcut, highcut = highcut)
            peaks = self.GetPeaks(filtered, fs = fs, k_h_max_R = 1, distance = 2.15) #distance of max 28 RR
            RR = len(peaks)/time[-1]*60

        elif method == "freq":
            
            fxx, pxx = signal.welch(x=ibi, fs=fs,nperseg=len(ibi),scaling = 'density')
            mask = (fxx > lowcut) & (fxx < highcut)
            
            fxx_acot = fxx[mask]
            pxx_acot = pxx[mask]
            
            RR = u.getPeakFrequencyFFT(fxx_acot, pxx_acot) * 60

        return RR

    def HeartRateVariability(self, serie, interpolated_series, fs = 4):
        mean_serie = np.mean(serie)
        results = {}
        results["Pcv"] = self.__P_cv__(serie)
        results["P_NMASD"] = self.__P_NMASD__(serie)
        results["SDNN"] = np.std(serie)
        results["RMSSD"] =self.__RMSSD__(serie)
        results["dif50"] = self.dif_50ms(serie)
        results["I_shan"] = self.__I_ShEn__(serie_ = serie)
        I = self.__I_SamEn__(serie)
        results["I_CSamEn"] = self.__I_CSamEn__(serie, I)
        #results["p1"] = self.__poincare1__(serie)
        results["p2"] = self.__poincare2__(serie)
        results["p3"] = self.__poincare3__(serie)

        std_0, std_1 = self.__std_y_j__(serie)
        std_ed = self.__std_c__(serie)
        results["std_0_nom"] = std_0/mean_serie
        results["std_ed_norm"] = std_ed/mean_serie

        results_frequency, fxx, pxx = self.__frequency_domain_hrv__(interpolated_series, fs)

        results.update(results_frequency)

        return results, fxx, pxx

    def GetStress(self, rr_intervals: np.ndarray) -> float:
        """
        Calculate Baevsky's Stress Index from RR intervals.

        Args:
            rr_intervals (np.ndarray): Array of RR intervals in seconds

        Returns:
            float: Stress index value (higher values indicate more stress)

        References:
            Baevsky, R. M. (2009). Methodical recommendations use KARDiVAR 
            system for determination of stress level and estimation of body 
            adaptability.
        """
        # Convert RR intervals to a numpy array
        rr_intervals = np.array(rr_intervals)

        # Mode (Mo): Most frequent RR interval (considered in seconds)
        Mo = mode(rr_intervals, keepdims=False).mode

        # Tolerance for mode (±0.05 seconds)
        tolerance = 0.05

        # Mode Amplitude (AMo): Percentage of RR intervals within ±0.05 seconds of the mode
        Mo_count = np.sum((rr_intervals >= Mo - tolerance) & (rr_intervals <= Mo + tolerance))
        AMo = (Mo_count / len(rr_intervals)) * 100  # in percentage

        # Variational Range (VR): Difference between max and min RR intervals (in seconds)
        VR = np.ptp(rr_intervals)  # Peak-to-peak range

        # Calculate Baevsky's Stress Index (SI)
        SI = AMo / (2 * VR * Mo)

        return SI

    def GetPeaks(self, signal_, fs, k_h_max_R = 1, distance = 0.3):
      h_max = k_h_max_R*np.mean(signal_)
      peaks, _  = signal.find_peaks(signal_, height = h_max, distance = round(fs*distance))

      return peaks

    def GetIBISeries(self, signal_, fs):

      peaks = self.GetPeaks(signal_,fs,1)
      serie = []
      start = 0
      t = np.linspace(0, len(signal_)/fs, len(signal_))

      for j in range(len(peaks)):
          i_r = peaks[j]
          t_r = t[i_r]
          if j == 0:
            start = i_r
          else:
            t_r_prev = t[peaks[j-1]]
            serie.append(t_r - t_r_prev)

      return serie, start, peaks
    
    def two_highest_peaks_fft(self, senal, fs, separation):
      peaks = u.GetPeaks(senal, fs, k_h_max_R = 1, separation = separation)
      peaks = peaks[np.argsort(senal[peaks])[::-1]]  # Sort them in descending order of amplitude
      peaks = peaks[:2]

      return peaks

    #-----------------------------CALCULAR INDICES----------------------------------------------------------------------
    # DISPERSIÓN

    def __P_NMASD__(self, serie_):
      serie = np.array(serie_)
      dx = np.diff(serie)
      P = np.sum(np.abs(dx)) / np.mean(serie)
      return P

    def __P_cv__(self, serie_):
        serie = np.array(serie_)
        P = np.std(serie) / np.mean(serie)
        return P

    def __RMSSD__(self,serie):

      rr_diffs = np.diff(serie)
      rr_diffs_squared = rr_diffs ** 2
      sum_rr_diffs_squared = np.sum(rr_diffs_squared)
      mean_squared_diffs = sum_rr_diffs_squared/(len(rr_diffs) - 1)
      rmssd = np.sqrt(mean_squared_diffs)

      return rmssd

    def dif_50ms(self, serie, tol = 0.05):
        contador = 0
        if len(serie) > 0:
            for i in range(0, len(serie) - 1):
                x = serie[i]
                x_next = serie[i+1]
                if (x_next - tol < x < x_next + tol) and (abs(x_next - x) < 3*tol):
                    contador = contador + 1
        return contador / len(serie) * 100


      # ENTROPÍA

    def __I_ShEn__(self, serie_, n_bins = 100000):
        serie = np.array(serie_)
        hist, bin_edges = np.histogram(serie, bins = n_bins)
        N = len(serie)
        I = 0
        for i in range(len(hist)):
            Ni = hist[i]
            p_xi = Ni / N
            if p_xi != 0:
                I -= p_xi*math.log2(p_xi)
        return I

    def __B__(self,serie_, m, r):
        serie = np.array(serie_)
        N = len(serie)
        b = 0
        for i in range(N):
            xi = serie[i:i+m]
            for j in range(N):
                if j != i:
                    xj = serie[j:j+m]
                    norma = np.max(np.abs(xi-xj))
                    b += np.heaviside(r - norma, 1)
        b = b / ((N - m)*(N - m - 1))
        return b

    def __I_SamEn__(self, serie_, m = 1, r = 0.05):
        serie = np.array(serie_)
        B1 = self.__B__(serie, m+1, r)
        B0 = self.__B__(serie, m, r)
        if (B0*B1 == 0):
            I = 1.5
        else:
            I = - np.log(B1 / B0)
        return I

    def __I_CSamEn__(self, serie_, I, r = 0.05):
        if I is not None:
            serie = np.array(serie_)
            IC = I + np.log(2*r) - np.log(np.mean(serie))
        else:
            IC = None
        return IC


    # POINCARÉ

    def __poincare3__(self, serie_, l_bin = 0.125):
        serie = np.array(serie_)
        dx = np.diff(serie)
        N = len(dx)
        poin = np.zeros([2,N])
        poin[0,:] = serie[:-1]
        poin[1,:] = dx

        n_bins = int(2 / l_bin)
        hist, edges = np.histogramdd(poin.T, bins = n_bins, range=[[0,2],[-1,1]])
        total_bins = np.count_nonzero(hist)
        return total_bins

    def __poincare2__(self, serie_, l_bin = 0.125, correct = True):
        serie = np.array(serie_)
        dx = np.diff(serie)
        N = len(dx)
        poin = np.zeros([2,N-1])
        poin[0,:] = dx[:-1]
        poin[1,:] = dx[1:]

        n_bins = int(2 / l_bin)
        hist, edges = np.histogramdd(poin.T, bins = n_bins, range=[[-1,1],[-1,1]])
        total_bins = np.count_nonzero(hist)
        i_bin_central_2 = int(n_bins / 2)
        i_bin_central_1 = i_bin_central_2 - 1
        bins_centrales = list(hist[i_bin_central_1][i_bin_central_1:i_bin_central_2+1]) + list(hist[i_bin_central_2][i_bin_central_1:i_bin_central_2+1])
        total_bins_centrales = np.count_nonzero(bins_centrales)
        corrected_bins = int(total_bins - total_bins_centrales)

        if not correct:
            corrected_bins = total_bins

        return corrected_bins

    def __poincare1__(self, serie_):
        serie = np.array(serie_)
        N = len(serie)
        poin = np.zeros([2,N-1])
        poin[0,:] = serie[:-1]
        poin[1,:] = serie[1:]
        return poin


    def __std_y_j__(self, serie_):
        poin_1 = self.__poincare1__(serie_)      # recibe poincaré 1
        x = np.array([poin_1[1,:], poin_1[0,:]])
        N = len(poin_1) + 1
        A = np.array([[np.sin(np.pi/4), np.cos(np.pi/4)], [np.cos(np.pi/4), -np.sin(np.pi/4)]])
        y = np.matmul(A, x)
        y_mean = np.mean(y[1,:])
        sum_0 = 0
        sum_1 = 0
        for i in range(N-1):
            sum_0 += (y[1,i] - y_mean)**2
            sum_1 += (y[0,i] - y_mean)**2
        std_0 = np.sqrt(sum_0 / (N-1))
        std_1 = np.sqrt(sum_1 / (N-1))
        return std_0, std_1


    def __std_c__(self, serie_):
        poin_2 = self.__poincare1__(serie_)
        N = len(poin_2[0,:]) + 2
        suma = 0
        for i in range(N-2):
            suma += np.sqrt(poin_2[0,i]*2 + poin_2[1,i]*2)
        std = suma / (N-2)
        return std

    def __frequency_domain_hrv__(self, rri, fs=4):
        # Estimate the spectral density using Welch's method
        fxx, pxx = signal.welch(x=rri, fs=fs)

        '''
        Segement found frequencies in the bands
        - Very Low Frequency (VLF): 0-0.04Hz
        - Low Frequency (LF): 0.04-0.15Hz
        - High Frequency (HF): 0.15-0.4Hz
        '''
        cond_vlf = (fxx >= 0) & (fxx < 0.04)
        cond_lf = (fxx >= 0.04) & (fxx < 0.15)
        cond_hf = (fxx >= 0.15) & (fxx < 0.4)

        # calculate power in each band by integrating the spectral density
        vlf = trapezoid(pxx[cond_vlf], fxx[cond_vlf])
        lf = trapezoid(pxx[cond_lf], fxx[cond_lf])
        hf = trapezoid(pxx[cond_hf], fxx[cond_hf])

        # sum these up to get total power
        total_power = vlf + lf + hf

        # find which frequency has the most power in each band
        peak_vlf = fxx[cond_vlf][np.argmax(pxx[cond_vlf])]
        peak_lf = fxx[cond_lf][np.argmax(pxx[cond_lf])]
        peak_hf = fxx[cond_hf][np.argmax(pxx[cond_hf])]

        # fraction of lf and hf
        lf_nu = 100 * lf / (lf + hf)
        hf_nu = 100 * hf / (lf + hf)

        results = {}
        results['Power VLF (ms2)'] = vlf
        results['Power LF (ms2)'] = lf
        results['Power HF (ms2)'] = hf
        results['Power Total (ms2)'] = total_power

        results['LF/HF'] = (lf/hf)
        results['Peak VLF (Hz)'] = peak_vlf
        results['Peak LF (Hz)'] = peak_lf
        results['Peak HF (Hz)'] = peak_hf

        results['Fraction LF (nu)'] = lf_nu
        results['Fraction HF (nu)'] = hf_nu
        return results, fxx, pxx

    def calculate_hr_band_energy(self, signal_data: np.ndarray, fs: float, 
                                  hr_bpm: float = None, band_width: float = 0.2) -> dict:
        """
        Calculate energy in the heart rate frequency band.
        
        This method computes the spectral power/energy within a frequency band
        centered around the heart rate frequency. This is useful for assessing
        signal quality and selecting the best ROI for rPPG analysis.
        
        Args:
            signal_data (np.ndarray): Input signal (rPPG or other physiological signal)
            fs (float): Sampling frequency in Hz
            hr_bpm (float, optional): Heart rate estimate in BPM. If None, it will be
                                     estimated from the signal using the dominant frequency
            band_width (float, optional): Width of the frequency band around HR in Hz 
                                         (default: 0.2 Hz, which corresponds to ±12 BPM)
        
        Returns:
            dict: Dictionary containing:
                - 'hr_band_energy': Energy/power in the HR frequency band
                - 'total_energy': Total energy in the physiological range (0.5-4 Hz)
                - 'hr_band_ratio': Ratio of HR band energy to total energy (0-1)
                - 'hr_frequency_hz': HR frequency in Hz
                - 'hr_bpm': HR in beats per minute
                - 'band_range_hz': Tuple of (lower, upper) frequency bounds
        
        Example:
            >>> calc = ParametersCalculator()
            >>> results = calc.calculate_hr_band_energy(rppg_signal, fs=30, hr_bpm=75)
            >>> print(f"HR band energy ratio: {results['hr_band_ratio']:.3f}")
        """
        # Estimate HR if not provided
        if hr_bpm is None:
            freqs, fft_signal = u.FFT(signal_data, fs)
            hr_frequency = u.getPeakFrequencyFFT(freqs, fft_signal)
            hr_bpm = hr_frequency * 60
        else:
            hr_frequency = hr_bpm / 60
        
        # Calculate power spectral density using Welch's method
        fxx, pxx = signal.welch(x=signal_data, fs=fs, nperseg=min(len(signal_data), 256))
        
        # Define HR frequency band (HR ± band_width)
        lowcut = max(0.5, hr_frequency - band_width)  # Ensure we stay in physiological range
        highcut = min(4.0, hr_frequency + band_width)  # Max 240 BPM
        
        # Define condition for HR band
        cond_hr_band = (fxx >= lowcut) & (fxx < highcut)
        
        # Define condition for total physiological range (0.5-4 Hz = 30-240 BPM)
        cond_total = (fxx >= 0.5) & (fxx < 4.0)
        
        # Calculate energy by integrating the power spectral density
        hr_band_energy = trapezoid(pxx[cond_hr_band], fxx[cond_hr_band])
        total_energy = trapezoid(pxx[cond_total], fxx[cond_total])
        
        # Calculate ratio (how concentrated is the energy in the HR band)
        hr_band_ratio = hr_band_energy / total_energy if total_energy > 0 else 0
        
        results = {
            'hr_band_energy': hr_band_energy,
            'total_energy': total_energy,
            'hr_band_ratio': hr_band_ratio,
            'hr_frequency_hz': hr_frequency,
            'hr_bpm': hr_bpm,
            'band_range_hz': (lowcut, highcut)
        }
        
        return results
