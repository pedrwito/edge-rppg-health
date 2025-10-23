from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
from typing import Tuple, List
from numpy.typing import NDArray

def plotSignal(series: NDArray, fs: int, color: str, title: str) -> None:
    """
    Plot a time series signal.

    Args:
        series (numpy.ndarray): Input signal array
        fs (int): Sampling frequency in Hz
        color (str): Color of the plot line
        title (str): Title of the plot
    """
    x = np.linspace(0, len(series)/fs, len(series))
    plt.plot(x, series, color=color)
    plt.title(title)
    plt.grid(True)
    plt.show()

def FFT(series: NDArray, fs: int, plot: bool = False) -> Tuple[NDArray, NDArray]:
    """
    Compute the Fast Fourier Transform of a signal up to 4Hz.

    Args:
        series (numpy.ndarray): Input signal array
        fs (int): Sampling frequency in Hz
        plot (bool, optional): If True, plots the FFT and spectrogram. Defaults to False.

    Returns:
        tuple: (frequencies array, FFT coefficients array)
    """
    fft_series = fft.fft(series)
    freqs = fft.fftfreq(len(fft_series), 1/fs)  # Frequency values

    # Keep only first half
    freqs = freqs[0:int(len(freqs)/2)]
    fft_series = fft_series[0:int(len(fft_series)/2)]

    # keep only the frequencies until 4Hz
    indices_until_4Hz = np.where(freqs <= 4)[0]
    freqs = freqs[indices_until_4Hz]
    fft_series = fft_series[indices_until_4Hz]

    if plot:
      plt.plot(freqs, np.abs(fft_series))
      plt.title(f"FFT {getPeakFrequencyFFT(freqs, fft_series)}")
      plt.grid(True)
      plt.show()

      spectogram_manual = np.abs(fft_series)**2/len(fft_series)
      plt.plot(freqs, spectogram_manual)
      plt.title(f"spectogram {getPeakFrequencyFFT(freqs,spectogram_manual)}")
      plt.grid(True)
      plt.show()

    return freqs, fft_series

def getPeakFrequencyFFT(freqs: NDArray, fft_series: NDArray) -> float:
    """
    Find the peak frequency in an FFT spectrum.

    Args:
        freqs (numpy.ndarray): Array of frequency values
        fft_series (numpy.ndarray): FFT coefficients array

    Returns:
        float: Peak frequency value
    """
    peak_index = np.argmax(np.abs(fft_series))
    return freqs[peak_index]

def GetPeaks(signal_: NDArray, fs: int, k_h_max_R: float, separation: float = 0.24) -> NDArray:
    """
    Find peaks in a signal using height and distance criteria.

    Args:
        signal_ (numpy.ndarray): Input signal array
        fs (int): Sampling frequency in Hz
        k_h_max_R (float): Height threshold multiplier relative to signal mean
        separation (float, optional): Minimum distance between peaks in seconds. Defaults to 0.24.

    Returns:
        numpy.ndarray: Array of peak indices
    """
    h_max = k_h_max_R*np.mean(signal_)
    peaks, _  = signal.find_peaks(signal_, height = h_max, distance = round(fs*separation))

    return peaks

def GetIBISeries(signal_: NDArray, fs: int, peaks: NDArray) -> Tuple[List[float], int, NDArray]:
    """
    Calculate Inter-Beat Intervals (IBI) series from peak locations.

    Args:
        signal_ (numpy.ndarray): Input signal array
        fs (int): Sampling frequency in Hz
        peaks (numpy.ndarray): Array of peak indices

    Returns:
        tuple: (IBI series array, start index, peaks array)
    """
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

#See if an IBI is too long or too short to be real using absolute threshholds and correct it

def process_serie_IBI_1(serie: List[float], peaks: List[int] = [], int_min: float = 0.24, int_max: float = 2) -> Tuple[List[float], List[int], List[int]]:
    """
    Process IBI series using absolute thresholds to correct intervals.

    Args:
        serie (list): IBI series array
        peaks (list, optional): Peak indices. Defaults to empty list.
        int_min (float, optional): Minimum acceptable interval in seconds. Defaults to 0.24.
        int_max (float, optional): Maximum acceptable interval in seconds. Defaults to 2.

    Returns:
        tuple: (processed IBI series, processed peaks array, indices of deleted intervals)
    """
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


def process_serie_IBI_2(serie_: List[float], r_peaks: List[int], i_del: List[int] = [], k_min: float = 1/3, k_max: float = 3) -> Tuple[List[float], List[int], NDArray]:
    """
    Process IBI series using relative thresholds based on median interval.

    Args:
        serie_ (list): IBI series array
        r_peaks (list): Peak indices
        i_del (list, optional): Indices to delete. Defaults to empty list.
        k_min (float, optional): Minimum threshold multiplier. Defaults to 1/3.
        k_max (float, optional): Maximum threshold multiplier. Defaults to 3.

    Returns:
        tuple: (processed IBI series, processed peaks array, indices of deleted intervals)
    """
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

def spectogram(series: NDArray, fs: int, lower_band: float = 0.5, upper_band: float = 4, plot: bool = False) -> Tuple[NDArray, NDArray]:
    """
    Compute the power spectral density using Welch's method up to 4Hz.

    Args:
        series (numpy.ndarray): Input signal array
        fs (int): Sampling frequency in Hz
        plot (bool, optional): If True, plots the spectrogram. Defaults to False.

    Returns:
        tuple: (frequencies array, power spectral density array)
    """
    fxx, pxx = signal.welch(x=series, fs=fs)
    indices_band = np.where((fxx >= lower_band) & (fxx < upper_band))[0]
    freqs = fxx[indices_band]
    fft_series = pxx[indices_band]
    if plot:
      plt.plot(freqs,fft_series)
      plt.xlim((0,4))
      plt.grid()
      plt.show()
    return freqs, fft_series

def energy(freqs, power) -> float:
  energy = np.sum(power) * (freqs[1] - freqs[0])
  return energy