from IppgSignalObtainer import IppgSignalObtainer

rois_pedro_video_30fps_lossless = IppgSignalObtainer.extractSeriesRoiRGBFromVideo(
    "video_lossless_20251210_193429_54bpm.mkv", 60,
    window_length=60, start_time=5,
    forehead=True, cheeks=True, under_nose=False, chin=True, full_face=False,
    play_video=True
)