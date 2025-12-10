import subprocess
import time
import csv
from datetime import datetime, UTC
from typing import Optional


def record_brio_lossless_fps(
    fps: int,
    device_name: str = "Logitech BRIO",
    video_out: Optional[str] = None,
    timestamp_out: Optional[str] = None,
    input_mode: str = "auto",
):
    """
    Records lossless video from a Logitech Brio using FFV1.
    - If fps == 30  -> uses 1920x1080
    - If fps == 60  -> uses 1280x720

    If no output filenames are provided, they are auto-generated using
    a UTC timestamp (e.g., video_lossless_YYYYMMDD_HHMMSS.mkv).

    input_mode:
        - "auto": use uncompressed YUY2 for 30 fps; use MJPEG for 60 fps
        - "raw":  force uncompressed YUY2 (will fail for 60 fps on BRIO)
        - "mjpeg": force MJPEG from camera (needed for 60 fps on BRIO)
    """

    if fps == 30:
        width, height = 1920, 1080
    elif fps == 60:
        width, height = 1280, 720
    else:
        raise ValueError("Only 30 or 60 FPS are supported.")

    print(f"Using resolution: {width}x{height} @ {fps} FPS")

    # Auto-generate output filenames if not provided
    if not video_out or not timestamp_out:
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        if not video_out:
            video_out = f"video_lossless_{ts}.mkv"
        if not timestamp_out:
            timestamp_out = f"timestamps_{ts}.csv"

    # Decide input transport from camera (DirectShow)
    # For BRIO, 60 fps requires MJPEG; 30 fps supports uncompressed YUY2.
    resolved_mode = input_mode.lower()
    if resolved_mode not in ("auto", "raw", "mjpeg"):
        raise ValueError('input_mode must be one of: "auto", "raw", "mjpeg"')
    if resolved_mode == "auto":
        resolved_mode = "mjpeg" if fps >= 60 else "raw"

    # Build ffmpeg command
    ffmpeg_cmd = ["ffmpeg", "-y", "-f", "dshow", "-rtbufsize", "256M"]

    if resolved_mode == "raw":
        # Uncompressed transport from camera (limited to 30 fps at 1080p on BRIO)
        ffmpeg_cmd += [
            "-pixel_format", "yuyv422",
            "-video_size", f"{width}x{height}",
            "-framerate", str(fps),
            "-i", f"video={device_name}",
        ]
    else:
        # MJPEG transport from camera (allows 60 fps on BRIO)
        ffmpeg_cmd += [
            "-vcodec", "mjpeg",
            "-video_size", f"{width}x{height}",
            "-framerate", str(fps),
            "-i", f"video={device_name}",
        ]

    ffmpeg_cmd += [
        "-vsync", "0",
        "-c:v", "ffv1",
        "-level", "3",
        "-pix_fmt", "rgb24",
        video_out,
    ]

    print("Starting lossless recording...")
    print("Press CTRL+C to stop.\n")

    process = None
    try:
        # Start ffmpeg capture
        process = subprocess.Popen(ffmpeg_cmd)

        frame_idx = 0

        with open(timestamp_out, "w", newline="") as f:
            writer = csv.writer(f)
            # Write header row for downstream parsing
            writer.writerow(["frame_index", "timestamp_unix", "timestamp_iso"])

            try:
                while True:
                    now = time.time()
                    # Higher precision ISO-8601 timestamp in UTC for better sync accuracy
                    iso_time = datetime.now(UTC).isoformat(timespec="microseconds")
                    writer.writerow([frame_idx, now, iso_time])
                    frame_idx += 1
                    # Pace the loop to approximately the target frame rate
                    time.sleep(1.0 / fps)

            except KeyboardInterrupt:
                print("\nStopping recording...")

    finally:
        # Ensure ffmpeg is stopped even if an error or interrupt occurs
        if process is not None:
            process.terminate()
            process.wait()

    print("Recording saved to:", video_out)
    print("Timestamps saved to:", timestamp_out)


# -------------------------
# Discovery/inspection helpers (Windows DirectShow)
# -------------------------
def list_dshow_devices() -> None:
    """
    Print DirectShow capture devices as seen by ffmpeg on Windows.
    ffmpeg prints this information to stderr; we forward it.
    """
    cmd = ["ffmpeg", "-hide_banner", "-f", "dshow", "-list_devices", "true", "-i", "dummy"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = (result.stderr or b"").decode(errors="ignore")
    print(output.strip())


def print_dshow_camera_modes(device_name: str = "Logitech Brio") -> None:
    """
    Print supported capture modes (resolution, pixel format, fps) for the device.
    Uses ffmpeg's dshow -list_options. Output depends on the driver.
    """
    cmd = ["ffmpeg", "-hide_banner", "-f", "dshow", "-list_options", "true", "-i", f"video={device_name}"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = (result.stderr or b"").decode(errors="ignore")
    print(output.strip())


def show_camera_property_dialog(device_name: str = "Logitech Brio", seconds: int = 3) -> None:
    """
    Open the Windows camera property dialog for the device via ffmpeg dshow.
    This brings up the driver UI; it does not print values to the console.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-f", "dshow",
        "-show_video_device_dialog", "true",
        "-i", f"video={device_name}",
        "-t", str(seconds),
        "-f", "null", "-",
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


try:
    import cv2  # Optional: used to print basic camera properties if available
except Exception:
    cv2 = None  # OpenCV not installed; helpers will inform the user


def print_camera_basic_settings_opencv(device_index: int = 0) -> None:
    """
    Print a small set of camera properties via OpenCV if available.
    This relies on the driver exposing controls through DirectShow/OpenCV.
    Not all cameras or drivers will report values here.
    """
    if cv2 is None:
        print("OpenCV not available. Install `opencv-python` to use this helper.")
        return

    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Could not open camera at index {device_index} via OpenCV (CAP_DSHOW).")
        return
    try:
        props = {
            "FRAME_WIDTH": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "FRAME_HEIGHT": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "FPS": cap.get(cv2.CAP_PROP_FPS),
            "BRIGHTNESS": cap.get(cv2.CAP_PROP_BRIGHTNESS),
            "CONTRAST": cap.get(cv2.CAP_PROP_CONTRAST),
            "SATURATION": cap.get(cv2.CAP_PROP_SATURATION),
            "HUE": cap.get(cv2.CAP_PROP_HUE),
            "GAIN": cap.get(cv2.CAP_PROP_GAIN),
            "EXPOSURE": cap.get(cv2.CAP_PROP_EXPOSURE),
            # White balance values may be readonly or -1 depending on driver
            "WHITE_BALANCE_BLUE_U": cap.get(getattr(cv2, "CAP_PROP_WHITE_BALANCE_BLUE_U", 17)),
        }
        for k, v in props.items():
            print(f"{k}: {v}")
    finally:
        cap.release()


# =========================
# EXAMPLE USAGE
# =========================
if __name__ == "__main__":

    # Device/mode discovery:
    list_dshow_devices()
    print_dshow_camera_modes(device_name="Logitech BRIO")

    # Open driver settings UI (does not print values):
    show_camera_property_dialog(device_name="Logitech BRIO", seconds=3)

    # Best-effort basic settings via OpenCV (if installed):
    # print_camera_basic_settings_opencv(device_index=0)
    
    #  Choose ONE:
    #record_brio_lossless_fps(fps=30)   # → 1080p30 lossless
    
    #record_brio_lossless_fps(fps=30, input_mode="raw", video_out ="pedro_video_30fps_lossless.mkv") 
    record_brio_lossless_fps(fps=30, input_mode="mjpeg", video_out ="pedro_video_30fps_mjpeg.mkv")
    #record_brio_lossless_fps(fps=60) # → 720p60 lossless



