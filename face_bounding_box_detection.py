import mediapipe as mp
import cv2
import streamlit as st
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import logging

logging.basicConfig(level=logging.INFO)

HEIGHT_PADDING = 0.4
WIDTH_PADDING = 0.3


def expand_bounding_box(
    bbox, frame_height, frame_width, height_padding=0.2, width_padding=0.3
):
    """
    Expands the bounding box to include the full head (top of the forehead to the chin).

    Args:
        bbox (tuple): The bounding box for the face (x, y, w, h).
        frame_height (int): Height of the video frame.
        frame_width (int): Width of the video frame.
        padding (float): Fraction of padding to add around the face (default: 0.2).

    Returns:
        tuple: Expanded bounding box (x, y, w, h).
    """
    x, y, w, h = bbox
    expanded_h = int(
        h * (1 + height_padding)
    )  # Increase height by 20% or any desired value
    expanded_y = max(
        0, y - int(height_padding * h)
    )  # Ensure it doesn't go out of the top of the frame

    expanded_w = int(
        w * (1 + width_padding)
    )  # Increase height by 20% or any desired value
    expanded_x = max(0, x - int(width_padding * w))
    expanded_h = min(expanded_h, frame_height - expanded_y)
    expanded_w = min(expanded_w, frame_width - expanded_x)

    return (expanded_x, expanded_y, expanded_w, expanded_h)

def get_bounding_box(frame):
    if frame is None:
        return None
        
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    width = frame.shape[1]
    height = frame.shape[0]
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = (
                int(bboxC.xmin * width),
                int(bboxC.ymin * height),
                int(bboxC.width * width),
                int(bboxC.height * height),
            )
            return (x, y, w, h)
    return None

def process_frame_batch(frames_batch):
    """Process a batch of frames and return their bounding boxes."""
    results = {}
    for frame_count, frame in frames_batch:
        try:
            bbox = get_bounding_box(frame)
            results[frame_count] = bbox
        except Exception as e:
            logging.error(f"Error processing frame {frame_count}: {e}")
            results[frame_count] = None
    return results

def get_bounding_box_coordinates(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize results dictionary with None for all frames
    bounding_box_coordinates = {i: None for i in range(total_frames)}
    
    # Read frames in batches
    batch_size = 32  # Process frames in small batches
    current_batch = []
    frame_count = 0
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_batch.append((frame_count, frame))
            frame_count += 1
            
            # Process batch when it reaches batch_size or on last frames
            if len(current_batch) >= batch_size or frame_count == total_frames:
                # Submit batch for processing
                future = executor.submit(process_frame_batch, current_batch)
                
                # Get results and update coordinates
                try:
                    batch_results = future.result(timeout=30)  # 30 second timeout per batch
                    bounding_box_coordinates.update(batch_results)
                except Exception as e:
                    logging.error(f"Error processing batch: {e}")
                    # Reprocess failed frames individually
                    for fc, fr in current_batch:
                        try:
                            bbox = get_bounding_box(fr)
                            bounding_box_coordinates[fc] = bbox
                        except Exception as e2:
                            logging.error(f"Error reprocessing frame {fc}: {e2}")
                
                # Update progress
                progress = min(0.99, frame_count / total_frames)
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames}")
                
                # Clear batch
                current_batch = []
    
    cap.release()
    
    # Verify all frames were processed
    processed_frames = sum(1 for v in bounding_box_coordinates.values() if v is not None)
    faces_detected = sum(1 for v in bounding_box_coordinates.values() if v is not None)
    
    progress_bar.progress(1.0)
    status_text.text(f"Completed processing {total_frames} frames")
    
    st.info(f"Processed all {total_frames} frames")
    st.info(f"Frames with faces detected: {faces_detected}/{total_frames}")
    
    # Final verification
    missing_frames = set(range(total_frames)) - set(bounding_box_coordinates.keys())
    if missing_frames:
        st.warning(f"Missing {len(missing_frames)} frames out of {total_frames}")
        
    return bounding_box_coordinates