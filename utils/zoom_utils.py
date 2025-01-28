from face_bounding_box_detection import expand_bounding_box
import logging
import math
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
from statistics import mean


logging.basicConfig(level=logging.INFO)
HEIGHT_PADDING = 0.3
WIDTH_PADDING = 0.3

def process_scales_centers_after_extracting_boundaries(
    out_queue, 
    zoom_scales, 
    zoom_effects, 
    total_frames,
    fps,
    ease_name
):

    # Get the processed scales and centers
    processed_scales = {}
    processed_centers = {}
    while not out_queue.empty():
        frame_count, processed_scale, center_x, center_y = out_queue.get()
        processed_scales[frame_count] = processed_scale
        processed_centers[frame_count] = (center_x, center_y)
    not_none_processed_centers = [p for _, p in processed_centers.items() if p[0]!=None and p[1]!=None]
    zoom_center = None
    if len(not_none_processed_centers)/len(processed_centers) > 0.4:
        first_items, second_items = zip(*not_none_processed_centers)
        mean_first = mean(first_items)
        mean_second = mean(second_items)
        zoom_center = (mean_first, mean_second)
        
    # get the minimum scale for holding in that position
    min_zoom_keys_per_effect = {}
    for i, effect in enumerate(zoom_effects):
        start_frame_init = int(effect.start_time * fps)
        start_frame = start_frame_init + int(effect.zoom_in_duration * fps)
        end_frame = min(total_frames, start_frame_init + int(effect.total_duration * fps))
        values = [(key, processed_scales[key]) for key in range(start_frame, end_frame)]
        min_zoom_key, min_zoom_scale = sorted(values, key=lambda x: x[1])[len(values) // 2]
        for frame_num in range(start_frame, end_frame):
            zoom_scales[frame_num] = min_zoom_scale
            if zoom_center is not None:
                processed_centers[frame_num] = processed_centers[min_zoom_key]
            else:
                processed_centers[frame_num] = processed_centers[zoom_center]
        min_zoom_keys_per_effect[i] = min_zoom_key
        effect.scale = min_zoom_scale
        
    for i, effect in enumerate(zoom_effects):
        start_frame = int(effect.start_time * fps)
        end_frame = start_frame + int(effect.zoom_in_duration * fps)#min(total_frames, start_frame + int(effect.zoom_in_duration * fps))
        for frame_num in range(start_frame, end_frame):
            current_time = frame_num / fps
            zoom_scales[frame_num] = effect.get_scale_at_time_with_lag(current_time, easing_function_name = ease_name)
            processed_centers[frame_num] = processed_centers[
                min_zoom_keys_per_effect[i]
            ]

    return zoom_scales, processed_centers


def get_initial_zoom_scales(total_frames, fps, zoom_effects, ease_button_state):
    zoom_scales = [1.0] * total_frames
    for effect in zoom_effects:
        start_frame = int(effect.start_time * fps)
        end_frame = min(total_frames, start_frame + int(effect.total_duration * fps))
        for frame_num in range(start_frame, end_frame):
            current_time = frame_num / fps
            zoom_scales[frame_num] = effect.get_scale_at_time_with_lag(current_time, easing_function_name=ease_button_state)

    return zoom_scales


def process_zoom_scales_centers_after_extracting_boundaries(
   zoom_effects,
   total_frames, 
   fps,
   height,
   width,
   bounding_box_data,
   ease_button_state
):
    
    
    zoom_scales = get_initial_zoom_scales(total_frames, fps, zoom_effects, ease_button_state)
    output_queue = Queue(maxsize=total_frames * 2)
    frame_queue = Queue(maxsize=400)
    status_text = st.empty()
    progress_bar = st.progress(0)

    with ThreadPoolExecutor(max_workers=16) as executor:
        executor.submit(process_scales, frame_queue, output_queue, zoom_scales, height, width)
        frame_count = 0
        while frame_count < total_frames:
            frame_queue.put((frame_count, bounding_box_data[frame_count]))
            frame_count += 1

            if frame_count % (total_frames // 20) == 0:
                progress_bar.progress(frame_count / total_frames)
                status_text.text(f"Processing frame {frame_count}/{total_frames}")

        # Signal that no more frames will be added
        frame_queue.put(None)
        # Wait for the processing to complete
        frame_queue.join()
    with st.spinner("Process scale centers after bounding box detection ..."):
        zoom_scales, processed_centers = process_scales_centers_after_extracting_boundaries(output_queue, zoom_scales, zoom_effects, total_frames, fps, ease_button_state)
    return zoom_scales, processed_centers
    

def process_scales(frame_queue, output_queue, zoom_scales, height, width):
    while True:
        try:
            frame_data = frame_queue.get()
            if frame_data is None:
                frame_queue.task_done()
                break  # Exit loop when sentinel is received

            frame_count, bounding_box_data = frame_data
            current_scale = zoom_scales[frame_count]
            refined_scale, center_x, center_y = get_expanded_frame_scales(bounding_box_data, height, width)
            if current_scale != 1.0:
                if refined_scale is not None:
                    output_queue.put(
                        (
                            frame_count,
                            min(current_scale, math.ceil(refined_scale * 10) / 10),
                            center_x,
                            center_y,
                        )
                    )
                else:
                    output_queue.put((frame_count, current_scale, center_x, center_y))
            else:
                output_queue.put((frame_count, current_scale, center_x, center_y))
            frame_queue.task_done()

        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            frame_queue.task_done()
            
        
def get_expanded_frame_scales(bounding_box_data, height, width):
    """
    Draw the bounding box on the frame.

    Args:
        frame (numpy.array): The frame to draw the bounding box on.
        bounding_box (tuple): The bounding box to draw (x, y, w, h).

    Returns:
        numpy.array: The frame with the bounding box drawn on it.
    """
    if bounding_box_data is not None:
        x, y, w, h = bounding_box_data
        x_new, y_new, w_new, h_new = expand_bounding_box(
            (x, y, w, h), height, width, height_padding=HEIGHT_PADDING, width_padding=WIDTH_PADDING
        )
        center_x, center_y = x_new + w_new // 2, y_new + h_new // 2
        scale_top = center_y / (center_y - y_new) if center_y != y_new else float('inf')
        scale_bottom = (height - center_y) / ((y_new + h_new) - center_y) if (y_new + h_new) != center_y else float('inf')
        scale_left = center_x / (center_x - x_new) if center_x != x_new else float('inf')
        scale_right = (width - center_x) / ((x_new + w_new) - center_x) if (x_new + w_new) != center_x else float('inf')
        
        min_scale = min(scale_top, scale_bottom, scale_left, scale_right)
        return min_scale, center_x, center_y
    else:    
        return None, width // 2, height // 2


#def get_scales_after_expanding_bounding_boxes(



    # # Get the processed scales and centers
    # processed_scales = {}
    # processed_centers = {}
    # while not out_queue.empty():
    #     frame_count, processed_scale, center_x, center_y = out_queue.get()
    #     processed_scales[frame_count] = processed_scale
    #     processed_centers[frame_count] = (center_x, center_y)

    # # get the minimum scale for holding in that position
    # min_zoom_keys_per_effect = {}
    # for i, effect in enumerate(zoom_effects):
    #     start_frame = int((effect.start_time + effect.zoom_in_duration)* fps)
    #     end_frame = min(
    #         total_frames,
    #         start_frame + int((effect.total_duration - effect.zoom_in_duration) * fps),
    #     )
    #     values = [(key, processed_scales[key]) for key in range(start_frame, end_frame)]
    #     min_zoom_key, min_zoom_scale = min(values, key=lambda x: x[1])
    #     for frame_num in range(start_frame, end_frame):
    #         zoom_scales[frame_num] = min_zoom_scale
    #         processed_centers[frame_num] = processed_centers[min_zoom_key]
    #     min_zoom_keys_per_effect[i] = min_zoom_key
    #     effect.scale = min_zoom_scale

    # for i, effect in enumerate(zoom_effects):
    #     start_frame = int(effect.start_time * fps)
    #     end_frame = min(total_frames, start_frame + int(effect.zoom_in_duration * fps))
    #     for frame_num in range(start_frame, end_frame):
    #         current_time = frame_num / fps
    #         zoom_scales[frame_num] = effect.get_scale_at_time_with_lag(current_time)
    #         processed_centers[frame_num] = processed_centers[
    #             min_zoom_keys_per_effect[i]
    #         ]

    # return zoom_scales, processed_centers