import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
from scipy import signal

def plotSignal(series, fs, color, title):
    x = np.linspace(0, len(series)/fs, len(series))
    plt.plot(x, series, color=color)
    plt.title(title)
    plt.grid(True)
    plt.show()

def FFT(series, fs):

    fft_series = fft.fft(series)
    freqs = fft.fftfreq(len(fft_series), 1/fs)  # Frequency values

    # Keep only first half
    freqs = freqs[0:int(len(freqs)/2)]
    fft_series = fft_series[0:int(len(fft_series)/2)]

    # keep only the frequencies until 4Hz
    indices_until_4hz = np.nonzero(freqs <= 4)[0]
    freqs = freqs[indices_until_4hz]
    fft_series = fft_series[indices_until_4hz]
    #print(IppgSignalObtainer.__getPeakFrequency__(freqs, fft_series)* 60)

    return freqs, fft_series

def getPeakFrequencyFFT(freqs, fft_series):
    peak_index = np.argmax(np.abs(fft_series))
    return freqs[peak_index]

def getPeaks(signal_, fs, k_h_max_R):
    h_max = k_h_max_R*np.mean(signal_)
    peaks, _  = signal.find_peaks(signal_, height = h_max, distance = round(fs*0.24))

    return peaks

def getIBISeries(signal_, fs, peaks):

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

def process_serie_IBI_1(serie, peaks = [], int_min = 0.24, int_max = 2):
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


def process_serie_RR_2(serie_, r_peaks, i_del = [], k_min = 1/3, k_max = 3):
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
