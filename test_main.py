from IppgSignalObtainer import IppgSignalObtainer
import matplotlib.pyplot as plt
import numpy as np
from Tools.ParametersCalculator import ParametersCalculator
calc = ParametersCalculator()
path = "videos_iPPG_paula/20241206_104904.mp4" 
rois = IppgSignalObtainer.extractSeriesRoiRGBFromVideo(path, 30, 30, 0, True, True, False, True)
print(rois.keys())
for roi in rois:
    print(f"{roi} ========================================================================")
    red = rois[roi]['red']
    green = rois[roi]['green']
    blue = rois[roi]['blue']
    ippg = IppgSignalObtainer.GetRppGSeriesfromRGBSeries(red, green, blue, 30, normalize=False, derivative=False, bandpass=False, detrend=True, method='pos')
    plt.plot(ippg)
    plt.show()
    hr = calc.ObtainHeartRate(np.array(ippg), [], fs = 30, method='two_peaks_fft')
    print(f"HR {roi}: {hr}")

best_roi = IppgSignalObtainer.FindBestRoiUsingHEnergyBand(rois, 30)
print(f"Best ROI: {best_roi}")
#IppgSignalObtainer.extractFullFaceSkinRGBFromVideo(path, 30, 30, 0, True, 2, False)

