import os
from pathlib import Path
import subprocess
from typing import List
import cv2
import numpy as np
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import logging
from utils.audio_text_utils import extract_audio
logging.basicConfig(level=logging.INFO)

class ZoomEffect:
    def __init__(
        self,
        start_time: float,
        end_time: float,
        zoom_in_duration: float,
        scale: float,
        zoom_out_duration: float = 0,
        lag_time=None,
    ):
        self.start_time = start_time
        self.end_time = end_time
        if lag_time is not None:
            self.lag_time = lag_time
        else:
            self.lag_time = end_time - start_time - zoom_in_duration - zoom_out_duration
        self.zoom_in_duration = zoom_in_duration
        self.zoom_out_duration = zoom_out_duration
        self.scale = scale
        self.total_duration = zoom_in_duration + zoom_out_duration + self.lag_time

    def linear(self, t):
        return t

    def ease_in(self, t):
        return t ** 2

    def ease_out(self, t):
        return 1 - (1 - t) ** 2

    def ease_in_out(self, t):
        return 3 * t**2 - 2 * t**3

    def apply_easing(self, easing_function_name, progress):
        # Map easing function names to instance methods
        easing_functions = {
            'linear': self.linear,
            'ease_in': self.ease_in,
            'ease_out': self.ease_out,
            'ease_in_out': self.ease_in_out,
        }
        
        easing_function = easing_functions.get(easing_function_name)
        if easing_function is None:
            raise ValueError(f"Easing function '{easing_function_name}' not found.")
        return easing_function(progress)

    def get_scale_at_time(self, current_time: float, easing_function_name='ease_in_out') -> float:
        time_in_effect = current_time - self.start_time
        if 0 <= time_in_effect <= self.zoom_in_duration:
            progress = time_in_effect / self.zoom_in_duration
            eased_progress = self.apply_easing(easing_function_name, progress)
            return 1.0 + (self.scale - 1.0) * eased_progress
        elif 0 <= time_in_effect - self.zoom_in_duration <= self.zoom_out_duration:
            time_in_zoom_out = time_in_effect - self.zoom_in_duration
            progress = time_in_zoom_out / self.zoom_out_duration
            eased_progress = self.apply_easing(easing_function_name, progress)
            return self.scale - (self.scale - 1.0) * eased_progress
        return 1.0

    def get_scale_at_time_with_lag(self, current_time: float, scale=None, easing_function_name='ease_in_out') -> float:
        if scale is None:
            scale = self.scale

        time_in_effect = current_time - self.start_time
        if 0 <= time_in_effect <= self.zoom_in_duration:
            progress = time_in_effect / self.zoom_in_duration
            eased_progress = self.apply_easing(easing_function_name, progress)
            return 1.0 + (scale - 1.0) * eased_progress
        elif (
            self.zoom_in_duration
            <= time_in_effect
            <= self.lag_time + self.zoom_in_duration
        ):
            return scale
        elif (
            0
            <= time_in_effect - self.zoom_in_duration - self.lag_time
            <= self.zoom_out_duration
        ):
            time_in_zoom_out = time_in_effect - self.zoom_in_duration - self.lag_time
            progress = time_in_zoom_out / self.zoom_out_duration
            eased_progress = self.apply_easing(easing_function_name, progress)
            return scale - (scale - 1.0) * eased_progress
        return 1.0



def apply_zoom(
    frame: np.ndarray, scale: float, center_x: int = None, center_y: int = None
) -> np.ndarray:
    if scale == 1.0:
        return frame
    height, width = frame.shape[:2]
    if center_x is None and center_y is None:
        center_x, center_y = width / 2, height / 2 - (height / 4)
    M = np.float32(
        [[scale, 0, center_x * (1 - scale)], [0, scale, center_y * (1 - scale)]]
    )
    return cv2.warpAffine(
        frame,
        M,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def process_frames_worker(frame_queue, out, zoom_scales, processed_centers):
    while True:
        try:
            frame_data = frame_queue.get()
            if frame_data is None:
                frame_queue.task_done()
                break  # Exit loop when sentinel is received

            frame_count, frame = frame_data
            current_scale = zoom_scales[frame_count]

            if current_scale != 1.0:
                frame = apply_zoom(frame, current_scale, processed_centers[frame_count][0], center_y=processed_centers[frame_count][1])

            out.write(frame)
            frame_queue.task_done()

        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            frame_queue.task_done()  # Ensure task_done is called even in case of error



def process_video(video_path: str, zoom_scales, processed_centers) -> str:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        temp_video = video_path + "_temp.mp4"
        transcode_command = [
            "ffmpeg", "-i", video_path,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-y",
            temp_video
        ]
        subprocess.run(transcode_command)
        video_path = temp_video
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_dir = Path("temp_output")
    temp_dir.mkdir(exist_ok=True, parents=True)
    temp_video = str(temp_dir / "temp_video.mp4")
    temp_audio = str(temp_dir / "temp_audio.aac")
    final_output = str(temp_dir / f"output_{Path(video_path).stem}.mp4")

    status_text = st.empty()
    status_text.text("Extracting audio...")
    try:
        extract_audio(video_path, temp_audio)
    except RuntimeError as e:
        logging.error("An error occurred during audio extraction: %s", e)
        return

    out = cv2.VideoWriter(
        temp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not out.isOpened():
        raise RuntimeError("Failed to initialize video writer")

    # zoom_scales = get_initial_zoom_scales(total_frames, fps, zoom_effects)
    progress_bar = st.progress(0)
    # output_queue = Queue(maxsize=total_frames)
    # frame_queue = Queue(maxsize=400)

    # with ThreadPoolExecutor(max_workers=16) as executor:
    #     executor.submit(process_bounding_boxes, frame_queue, output_queue, zoom_scales)
    #     frame_count = 0
    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             break

    #         frame_queue.put((frame_count, frame))
    #         frame_count += 1

    #         if frame_count % (total_frames // 20) == 0:
    #             progress_bar.progress(frame_count / total_frames)
    #             status_text.text(f"Processing frame {frame_count}/{total_frames}")

    #     cap.release()
    #     # Signal that no more frames will be added
    #     frame_queue.put(None)
    #     # Wait for the processing to complete
    #     frame_queue.join()

    # zoom_scales, processed_centers = process_scales_centers_after_extracting_boundaries(output_queue, zoom_scales, zoom_effects, total_frames, fps)
    # cap = cv2.VideoCapture(video_path)
    frame_queue = Queue(maxsize=400)
    with ThreadPoolExecutor(max_workers=16) as executor:
        executor.submit(process_frames_worker, frame_queue, out, zoom_scales, processed_centers)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_queue.put((frame_count, frame))
            frame_count += 1

            if frame_count % (total_frames // 20) == 0:
                progress_bar.progress(frame_count / total_frames)
                status_text.text(f"Processing frame {frame_count}/{total_frames}")

        cap.release()

        # Signal that no more frames will be added
        frame_queue.put(None)

        # Wait for the processing to complete
        frame_queue.join()
        out.release()

        logging.info("Beginning audio-video combination.")
        status_text.text("Combining video with audio...")

        ffmpeg_command = [
                "ffmpeg",
                "-i", temp_video,        # Input video
                "-i", temp_audio,        # Input audio
                "-c:v", "libx264",       # Video codec (H.264)
                "-crf", "28",            # Increase CRF for lower quality and smaller size
                "-b:v", "1000k",         # Set video bitrate (1 Mbps)
                "-vf", "scale=640:360",  # Resize video to 640x360
                "-c:a", "aac",           # Use AAC for audio
                "-b:a", "128k",          # Set audio bitrate to 128 kbps
                "-shortest",             # Use the shortest input stream (video or audio)
                "-movflags", "+faststart",  # Enable fast start for streaming
                "-y",                    # Overwrite output file if it exists
                final_output
            ]
        result = subprocess.run(
            ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        if result.returncode != 0:
            logging.error("FFmpeg error during combination: %s", result.stderr.decode())
            raise RuntimeError("Failed to combine video and audio")

        logging.info("Combination completed successfully. Cleaning up temporary files.")
        if os.path.exists(temp_video):
            os.remove(temp_video)
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

        status_text.text("Processing complete!")
        return final_output
