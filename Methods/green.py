import numpy as np
from Tools.signalprocesser import SignalProcessor

def green(series, fs, plot_steps = False, normalize = True, detrend = True, bandpass = True, derivative = True):

    color = 'green'
    processedSeries = np.array(series)
    if normalize:
        processedSeries = SignalProcessor.normalize(processedSeries, fs, plot = plot_steps, color = color)

    if detrend:
        processedSeries = SignalProcessor.detrend(processedSeries, fs, plot = plot_steps)

    if bandpass:
        processedSeries = SignalProcessor.bandpass(processedSeries, fs, plot = plot_steps, color = color)

    if derivative:
        processedSeries = SignalProcessor.derivativeFilter(processedSeries, fs, plot = plot_steps, color = color)


    return processedSeries