from Tools.signalprocesser import signalprocesser
import numpy as np
from scipy import signal
from scipy.integrate import trapezoid
from scipy.stats import mode
import math
import Tools.utils as utils

class ParametersCalculator():


    def getParameters(self, signal, fs, interpolation = "cubic", fs_interpolation = 4):

        ibi_series, _, peaks = self.getIBISeries(signal, fs)
        parameters = {}

        

        #Processing IBI series by removing too short IBI (due to incorrect detections) using absolute threshold and then relative threshhold (based on median IBI)
        ibi_series, peaks_aux, i_del = signalprocesser.process_serie_IBI_absolute(ibi_series, peaks = peaks, int_min = 0.24, int_max = 2)
        ibi_series, peaks_aux, i_del  = signalprocesser.process_serie_IBI_relative(ibi_series, peaks_aux, i_del = [], k_min = 1/2.5, k_max = 2.5)
        t_ibi= np.cumsum(ibi_series)


        parameters["HR"] = self.obtainHeartRate(signal, ibi_series, fs, method = 'freq')
        #TODO add ectopic beat removal using the classifier developed for the previous work

        #calculation of interpolated IBI series for frequency based parameters of HRV and Respiratory Rate(RR)
        if interpolation == "linear":
          interpolated_time, interpolated_series = signalprocesser.linear_interpolation(t_ibi, ibi_series, fs_interpolation)

        elif interpolation == "cubic":
          interpolated_time, interpolated_series = signalprocesser.cubic_interpolation(t_ibi, ibi_series, fs_interpolation)

        parameters["RR"] = self.obtainRespiratoryRate(interpolated_series, fs = fs_interpolation, method = "freq")

        results_HRV, fxx, pxx = self.heartRateVariability(ibi_series, interpolated_series, fs = fs_interpolation)

        parameters.update(results_HRV)

        parameters["stress"] = self.getStress(ibi_series)

        return parameters

    def obtainHeartRate(self, signal, series, fs, method = 'frequency'):

        if method == 'freq':
            freqs, fftSeriesICA = utils.FFT(signal, fs)
            HR = utils.getPeakFrequencyFFT(freqs, fftSeriesICA) * 60

        #Get HR from peak detection in time domain.
        elif method == 'time':
            lenght = len(signal)
            HR = len(series)/(lenght/fs)*60

        return HR

    def obtainRespiratoryRate(self, ibi, interpolate = None, lowcut = 0.15, highcut = 0.5, method = "freq", fs = 4):

      if interpolate:
        t_ibi = np.cumsum(ibi)
        if interpolate == "linear":
          ibi = signalprocesser.linear_interpolation(t_ibi, ibi, fs)
        elif interpolate == "cubic":
          ibi = signalprocesser.cubic_interpolation(t_ibi, ibi, fs)

      time = np.cumsum(ibi)
      filtered = signalprocesser.bandpass(ibi, fs, lowcut = lowcut, highcut = highcut)
      
      if method == "time":
        peaks = self.getPeaks(filtered, fs = fs, k_h_max_R = 1, distance = 2.15) #distance of max 28 RR
        RR = len(peaks)/time[-1]*60

      elif method == "freq":
            freqs, fftSeries= utils.FFT(ibi, fs)
            mask = (freqs > 0.15) & (freqs < 0.4)
            

            filtered_freqs = freqs[mask]
            filtered_fftSeries = fftSeries[mask]
            RR = utils.getPeakFrequencyFFT(filtered_freqs, filtered_fftSeries) * 60

      return RR

    def heartRateVariability(self, serie, interpolated_series, fs = 4):
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
    """
    # Uses Baevsky's Stress Index
    def getStress(self, series):

        # Convert RR intervals to numpy array for easier calculations
        series = np.array(series)

        # Calculate the mode (most frequent interval)
        Mo = mode(series)[0]

        # Calculate AMo (percentage of intervals close to the mode)
        range_around_mo = 50/1000  # Define a range around Mo, for example, +/- 50 ms
        AMo = np.sum((series > Mo - range_around_mo) & (series < Mo + range_around_mo)) / len(series) * 100

        # Calculate MxDMn (difference between max and min RR intervals)
        MxDMn = np.max(series) - np.min(series)

        # Calculate Baevsky's Stress Index
        stress_index = (AMo * 100) / (2 * Mo * MxDMn)

        return stress_index

    """

    def getStress(self, rr_intervals):
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

    def IBISGroundTruthPPG():
        pass

    def RRGroundTruthECG():
        pass

    def getPeaks(self, signal_, fs, k_h_max_R = 1, distance = 0.24):
      h_max = k_h_max_R*np.mean(signal_)
      peaks, _  = signal.find_peaks(signal_, height = h_max, distance = round(fs*distance))

      return peaks

    def getIBISeries(self, signal_, fs):

      peaks = self.getPeaks(signal_,fs,k_h_max_R = 1, distance = 0.3)
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

        if correct == False:
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
