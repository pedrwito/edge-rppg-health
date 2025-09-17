import numpy as np
from Tools.signalprocesser import SignalProcesser

def green(series, fs, plot_steps = False, normalize = True, detrend = True, bandpass = True, derivative = True):

    color = 'green'
    processedSeries = np.array(series)
    if normalize:
        processedSeries = SignalProcesser.normalize(processedSeries, fs, plot = plot_steps, color = color)

    if detrend:
        processedSeries = SignalProcesser.detrend(processedSeries, fs, plot = plot_steps)

    if bandpass:
        processedSeries = SignalProcesser.bandpass(processedSeries, fs, plot = plot_steps, color = color)

    if derivative:
        processedSeries = SignalProcesser.derivativeFilter(processedSeries, fs, plot = plot_steps, color = color)


    return processedSeries