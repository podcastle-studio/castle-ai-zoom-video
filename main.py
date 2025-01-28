import streamlit as st
import cv2
import re
import torchaudio as ta
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from glob import glob
import streamlit.components.v1 as components
import base64
import os
import gc
import hashlib
from pathlib import Path
from asr import get_client_settings, transcribe_audio
from utils.audio_text_utils import (
    save_audio_from_video,
    check_ffmpeg,
    split_sentences_by_seconds,
    get_word_indices,
    split_words_by_duration,
    add_silence_duration,
    split_and_save_audio,
    construct_new_sentences,
    add_sentences_to_file,
    prediction_checks
)
from utils.emphasess_utils import save_emphasis_predictions
import json
from predictor import ClaudeAdapter, GPTAdapter
from zoom_effect import ZoomEffect, process_video
from dotenv import load_dotenv, find_dotenv
import warnings
from utils.zoom_utils import process_zoom_scales_centers_after_extracting_boundaries
from utils.scene_utils import read_scenes_with_seconds, find_broll_boundaries
from face_bounding_box_detection import get_bounding_box_coordinates
import time
import threading

_ = load_dotenv(find_dotenv())


# Ignore warnings
st.set_option("deprecation.showfileUploaderEncoding", False)
warnings.filterwarnings("ignore")

SPLIT_SENTENCE_BY_DURATION = 120  * 5
ZOOM_DURATION = 0.7

def cleanup_threads():
    current_thread = threading.current_thread()
    for thread in threading.enumerate():
        if thread != current_thread and not thread.daemon:
            try:
                thread.join(timeout=1)
            except TimeoutError:
                pass


# Set up logging configuration
def get_zooms(
    preds, sntnces_splitted_by_duration, splittedwords, slow=False, jumpcut=False
):
    zoom_effects = []
    for i, pred in enumerate(preds):
        prediction = pred["zoom_events"]
        for p in prediction:
            sentence_num = p["sentence_number"]
            text_applied = p["text_applied"]
            # reason = p['reason']
            zoom_in_scale = p["zoom_in_scale"]
            zoom_out_duration = p["zoom_out_duration"]
            st_idx, end_idx = get_word_indices(
                sntnces_splitted_by_duration[i][sentence_num - 1], text_applied
            )
            if st_idx == -1 or end_idx == -1:
                continue
            start_time = splittedwords[i][sentence_num - 1][st_idx][1]
            end_time = splittedwords[i][sentence_num - 1][end_idx][2]
            if slow and not jumpcut:
                zoom_effects.append(
                    ZoomEffect(
                        start_time,
                        end_time,
                        end_time - start_time,
                        zoom_in_scale,
                        zoom_out_duration=zoom_out_duration,
                        lag_time=0,
                    )
                )
            elif not slow and not jumpcut:
                zoom_effects.append(
                    ZoomEffect(start_time, end_time, 1, zoom_in_scale, 1)
                )
            elif slow and jumpcut:
                zoom_effects.append(
                    ZoomEffect(
                        start_time,
                        end_time,
                        end_time - start_time,
                        zoom_in_scale,
                        0,
                        lag_time=0,
                    )
                )

    return zoom_effects


def get_zooms_claude(
    preds,
    sntnces_splitted_by_duration,
    splittedwords,
    zoom_in_duration=None,
    slow=False,
    jumpcut=False,
    hold=False,
):
    zoom_effects = []
    for i, pred in enumerate(preds):
        prediction = pred.get(list(pred.keys())[0], [])
        for p in prediction:
            sentence_num = p["sentence_number"]
            text_applied = p["zoom_in_phrase"]
            # reason = p['reason']
            zoom_in_scale = 1.3
            zoom_duration = zoom_in_duration
            transition_sentence_num = p["transition_sentence_number"]
            transition_sentence_word = p["transition_sentence_word"]

            # zoom_out_duration = p['zoom_out_duration']
            st_idx, end_idx = get_word_indices(
                sntnces_splitted_by_duration[i][sentence_num], text_applied
            )
            st_idx_cut, end_idx_cut = get_word_indices(
                sntnces_splitted_by_duration[i][transition_sentence_num],
                transition_sentence_word,
            )

            if st_idx == -1 or end_idx == -1:
                continue
            if st_idx_cut == -1 or end_idx_cut == -1:
                continue
            start_time = splittedwords[i][sentence_num][st_idx][1]
            end_time = splittedwords[i][transition_sentence_num][st_idx_cut][1]

            if not slow and jumpcut and hold:
                if end_time-start_time < zoom_duration:
                    continue
                zoom_effects.append(
                    ZoomEffect(start_time, end_time, zoom_duration, zoom_in_scale, 0)
                )
            elif not slow and jumpcut and not hold:
                zoom_effects.append(
                    ZoomEffect(
                        start_time,
                        end_time,
                        zoom_duration,
                        zoom_in_scale,
                        0,
                        lag_time=0,
                    )
                )

    return zoom_effects


def main():
    st.title("Video Zoom In Editor")

    # Check for FFMPEG
    if not check_ffmpeg():
        st.error(
            """
        FFMPEG is not installed. Please install it:
        - Windows: Download from https://ffmpeg.org/download.html
        - Mac: brew install ffmpeg
        - Linux: sudo apt-get install ffmpeg
        """
        )
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a video file", accept_multiple_files=False, type=["mp4", "avi", "mov"]
    )

    if uploaded_file:
        # Create temporary directory for uploads
        temp_dir = Path("./uploaded_files/recordings/video_recordings")
        temp_dir.mkdir(exist_ok=True)

        # Save uploaded file
        video_path = str(temp_dir / uploaded_file.name)
        cap = cv2.VideoCapture(video_path)
        st.session_state.fps = cap.get(cv2.CAP_PROP_FPS)
        st.session_state.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        st.session_state.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        st.session_state.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        # Display original video
        st.subheader("Original Video")
        st.video(video_path)

        # Transcribe audio and save transcription in a file
        if uploaded_file or not st.session_state.get("data_uploaded", False):

            st.session_state.data_uploaded = True

            st.session_state.interview_id = hashlib.md5(
                f"{uploaded_file.name}".encode("utf-8")
            ).hexdigest()

            save_audio_from_video(video_path, video_path.replace(".mp4", ".mp3"))

            st.session_state.audio_file = video_path.replace(".mp4", ".mp3")
            st.session_state.video_file = video_path
            st.subheader("Transcribe Audio")

            # transcribe audios
            if "asr_client_settings" not in st.session_state:
                st.session_state["asr_client_settings"] = get_client_settings()

            output_file_path_sentence = st.session_state.audio_file.replace(
                "recordings", "transcriptions"
            ).replace(".mp3", "_trancriptions_with_align_sentence.txt")
            os.makedirs(os.path.dirname(output_file_path_sentence), exist_ok=True)

            output_file_path_words = st.session_state.audio_file.replace(
                "recordings", "transcriptions"
            ).replace(".mp3", "_trancriptions_with_align_words.json")
            (
                st.session_state.interview_to_transcription_meta_sentence,
                st.session_state.interview_to_transcription_meta_words,
            ) = transcribe_audio(
                st.session_state.audio_file,
                output_file_path_sentence,
                output_file_path_words,
                st.session_state["asr_client_settings"],
            )

        # split audios by sentence to predict
        
        
        # sentence_times = []
        # for sentence in st.session_state.interview_to_transcription_meta_sentence:
        #     pattern = r"\|([0-9.]+)\|([0-9.]+)"

        #     # Find matches
        #     match = re.search(pattern, sentence)

        #     if match:
        #         start_time = float(match.group(1))
        #         end_time = float(match.group(2))
        #         sentence_times.append((start_time, end_time))
        #     else:
        #         raise Exception("Sentence must have start and end times")
        
        
        splitted_audio_dir = f"./uploaded_files/recordings/splitted_audios/{video_path.split('/')[-1].split('.')[0]}"
        os.makedirs(splitted_audio_dir, exist_ok=True)

        with st.spinner("Split and save the audio..."):
            split_and_save_audio(
                st.session_state.audio_file, output_file_path_sentence, splitted_audio_dir
            )
        # cleanup_threads() 
        
        # gc.collect()  # Force garbage collection

        # # Give system a moment to fully release resources
        # time.sleep(1)
        # Run emphasis model and change the sentence txt file
        files = glob(f"{splitted_audio_dir}/*.mp3")
        splitted_audio_txt_dir = f"./uploaded_files/emphasis_detection/{video_path.split('/')[-1].split('.')[0]}"
        
        with st.spinner("Detection of emphasized phrases..."):
            save_emphasis_predictions(files, splitted_audio_txt_dir)
        
        # cleanup_threads() 
        # gc.collect()  # Force garbage collection

        # Give system a moment to fully release resources
        # time.sleep(1)
        
        audio_basename = os.path.basename(st.session_state.audio_file)
        # add silence duration to the words
        # with open(st.session_state.interview_to_transcription_meta_words) as f:
        word_data = (
            st.session_state.interview_to_transcription_meta_words
        )  # json.load(f)
        
        # 
        add_silence_duration(word_data)
        
        # add B-roll detection to the video
        scenes_splitted_video_path = f'./uploaded_files/recordings/scene_splitted_videos/{video_path.split("/")[-1].split(".")[0]}/'
        os.makedirs(scenes_splitted_video_path, exist_ok=True)
        
        
        with st.spinner("Detection of separate scenes ..."):
            if not os.path.exists(scenes_splitted_video_path + f'{video_path.split("/")[-1].split(".")[0]}-Scenes.csv'):
                os.system(f'python -m scenedetect -i "{video_path}" --backend pyav detect-content list-scenes -o "{scenes_splitted_video_path}" split-video -o "{scenes_splitted_video_path}"')

        # cleanup_threads() 
        # gc.collect()  # Force garbage collection

        # Give system a moment to fully release resources
        # time.sleep(1)
        
        bounding_boxes_path =  './uploaded_files/recordings/face_detected_bounding_boxes/'
        os.makedirs(bounding_boxes_path, exist_ok=True)
        if os.path.exists(f'{bounding_boxes_path}/{video_path.split("/")[-1].split(".")[0]}.json'):
            with open(f'{bounding_boxes_path}/{video_path.split("/")[-1].split(".")[0]}.json') as f:
                st.session_state.bounding_box_coordinates = {int(key): tuple(value) if value is not None else None for key, value in json.load(f).items()}
                st.session_state.total_frames  = len(st.session_state.bounding_box_coordinates)
        else:  
            if not st.session_state.get("face_detected", False):
                with st.spinner("Face Detection for bounding boxes ..."):
                    st.session_state.face_detected = True
                    st.session_state.bounding_box_coordinates = get_bounding_box_coordinates(video_path)
                    st.session_state.total_frames  = len(st.session_state.bounding_box_coordinates)
                    json_compatible_bounding_box_data = {str(key): (list(value) if value is not None else None) for key, value in st.session_state.bounding_box_coordinates.items()}
                    with open(f'{bounding_boxes_path}/{video_path.split("/")[-1].split(".")[0]}.json', 'w') as json_file:
                        json.dump(json_compatible_bounding_box_data, json_file, indent=4)

        scene_path_json = f"./uploaded_files/recordings/scene_splitted_videos/{video_path.split('/')[-1].split('.')[0]}.json"
        if not os.path.exists(scene_path_json):
            scenes_data = read_scenes_with_seconds(scenes_splitted_video_path + f'{video_path.split("/")[-1].split(".")[0]}-Scenes.csv')
            os.makedirs(os.path.dirname(scene_path_json), exist_ok=True)
            with open(scene_path_json, "w+") as f:
                json.dump(scenes_data, f)
        else:
            with open(scene_path_json) as f:
                scenes_data = json.load(f)
                
        for i, scene in enumerate(scenes_data):
            scene_count_detected = 0
            start_frame = int(scene['start_time'] * st.session_state.fps)
            end_frame = int(scene['end_time'] * st.session_state.fps)
            for frame_num in range(start_frame, end_frame):
                if frame_num not in st.session_state.bounding_box_coordinates.keys():
                    st.session_state.bounding_box_coordinates[frame_num] = None
                if st.session_state.bounding_box_coordinates[frame_num] is not None:
                    scene_count_detected += 1
            if end_frame - start_frame == 0:
                scenes_data[i]['Broll'] = False
            elif scene_count_detected/(end_frame - start_frame) > 0.7:
                scenes_data[i]['Broll'] = False
            else:
                scenes_data[i]['Broll']= True

        # Change the sentence capitalization
        emphasis_files = glob(f"{splitted_audio_txt_dir}/*.txt")
        sentence_info_path_updated = output_file_path_sentence.replace(
            ".txt", "_updated.txt"
        )
        
        broll_boundaries = find_broll_boundaries(word_data, scenes_data)

        st.session_state.new_sentences = construct_new_sentences(
            emphasis_files,
            audio_basename,
            word_data,
            broll_boundaries, 
            sentence_info_path_updated,
            output_file_path_sentence,
        )
        numbered_txt_file = sentence_info_path_updated.replace(
            "_updated.txt", "_updated_numbered.txt"
        )
        if not os.path.exists(numbered_txt_file):
            for i, sent in enumerate(st.session_state.new_sentences):
                add_sentences_to_file(f"{i}. {sent}", numbered_txt_file)
        
        st.session_state.sentences_splitted_by_duration = (
                split_sentences_by_seconds(st.session_state.new_sentences, SPLIT_SENTENCE_BY_DURATION)
            )
        st.session_state.splitted_words = split_words_by_duration(
            word_data,
            [len(sen) for sen in st.session_state.sentences_splitted_by_duration],
        )
        splitted_sentences = [
            [f"{i}. {sentence}" for i, sentence in enumerate(sentences, start=0)]
            for sentences in st.session_state.sentences_splitted_by_duration
        ]
        
        if "button_clicked" not in st.session_state:
            st.session_state.button_clicked = None
        if "predictions" not in st.session_state:
            st.session_state.predictions = None
        if "zoom_effects" not in st.session_state:
            st.session_state.zoom_effects = None
        if "output_path" not in st.session_state:
            st.session_state.output_path = None
            
        col1, col2 = st.columns(2)
        #ChatGPT predictions
        with col1:
            if st.button("GPT Predictions"):
                # sentences_splitted_by_duration = split_sentences_by_seconds(st.session_state.interview_to_transcription_meta_sentence, SPLIT_SENTENCE_BY_DURATION )
                # splitted_words = split_words_by_duration(st.session_state.interview_to_transcription_meta_words, [len(sen) for sen in sentences_splitted_by_duration] )
                # splitted_sentences = [[f"{i}. {sentence}" for i, sentence in enumerate(sentences, start=1)] for sentences in sentences_splitted_by_duration]

                predictor = GPTAdapter(model_name="gpt-4o",api_key=os.getenv("OPENAI_API_KEY"))

                os.makedirs("gpt_results", exist_ok=True)

                # Construct the JSON file path
                audio_file_name = st.session_state.audio_file.split('/')[-1].split('.')[0]
                json_file_path = f"gpt_results/{audio_file_name}.json"

                # Check if predictions already exist
                if not os.path.exists(json_file_path):
                    # Generate predictions using GPT predictor
                    st.session_state.predictions = predictor.get_predictions(splitted_sentences)
                    
                    # out_message = prediction_checks(st.session_state.predictions,
                    #                       st.session_state.sentences_splitted_by_duration, 
                    #                       st.session_state.splitted_words,
                    #                       broll_boundaries,
                    #                       int(st.session_state.total_frames/st.session_state.fps/60),
                    #                       word_data
                    #                       )
                    # if out_message:
                    #     st.write(out_message)
                    #     st.session_state.predictions = predictor.get_predictions(
                    #     splitted_sentences,
                    #     num_inputs=len(splitted_sentences),
                    #     prev_preds=st.session_state.predictions, 
                    #     out_message = out_message
                    #         )
                            
                    # Save predictions to a JSON file
                    with open(json_file_path, "w") as f:
                        json.dump(st.session_state.predictions, f)
                    st.success(f"Predictions saved to {json_file_path}")
                else:
                    # Load existing predictions
                    with open(json_file_path, "r") as f:
                        st.session_state.predictions = json.load(f)
                    # out_message = prediction_checks(st.session_state.predictions,
                    #                       st.session_state.sentences_splitted_by_duration, 
                    #                       st.session_state.splitted_words, 
                    #                       broll_boundaries,
                    #                       int(st.session_state.total_frames/st.session_state.fps/60),
                    #                       word_data
                    #                       )
                    # st.info(f"Loaded existing predictions from {json_file_path}")
                    # if out_message:
                    #     st.write(out_message)
                        # st.session_state.predictions = predictor.get_predictions(
                        #                                 splitted_sentences,
                        #                                 num_inputs=len(splitted_sentences), 
                        #                                 prev_preds=st.session_state.predictions,
                        #                                 out_message = out_message
                        #     )
                    #     with open(json_file_path, "w") as f:
                    #         json.dump(st.session_state.predictions, f)
                    

        
        # st.write(st.session_state)
        with col2:
            if st.button("Claude Predictions"):
                st.session_state.button_clicked = "claude_predictions"
                predictor = ClaudeAdapter(
                    model_name="claude-3-5-sonnet-20241022",
                    api_key=os.getenv("ANTHROPIC_API_KEY"),
                )
                claude_pred_dir = "claude_results_2_shot_prompt"
                os.makedirs(claude_pred_dir, exist_ok=True)
                # st.session_state.sentences_splitted_by_duration = (
                #     split_sentences_by_seconds(st.session_state.new_sentences, SPLIT_SENTENCE_BY_DURATION)
                # )
                # st.session_state.splitted_words = split_words_by_duration(
                #     word_data,
                #     [len(sen) for sen in st.session_state.sentences_splitted_by_duration],
                # )
                # splitted_sentences = [
                #     [f"{i}. {sentence}" for i, sentence in enumerate(sentences, start=1)]
                #     for sentences in st.session_state.sentences_splitted_by_duration
                # ]

                # os.makedirs("claude_results", exist_ok=True)  
                # st.session_state.sentences_splitted_by_duration = (
                #     split_sentences_by_seconds(st.session_state.new_sentences, SPLIT_SENTENCE_BY_DURATION)
                # )
                # st.session_state.splitted_words = split_words_by_duration(
                #     word_data,
                #     [len(sen) for sen in st.session_state.sentences_splitted_by_duration],
                # )
                # splitted_sentences = [
                #     [f"{i}. {sentence}" for i, sentence in enumerate(sentences, start=1)]
                #     for sentences in st.session_state.sentences_splitted_by_duration
                # ]

                    # Construct the JSON file path
                audio_file_name = st.session_state.audio_file.split("/")[-1].split(".")[0]
                json_file_path = f"{claude_pred_dir}/{audio_file_name}.json"
                if not os.path.exists(json_file_path):
                    with st.spinner("Predicting zoom points..."):
                        st.session_state.predictions = predictor.get_predictions(
                            splitted_sentences, num_inputs=len(splitted_sentences)
                        )
                        out_message = prediction_checks(st.session_state.predictions,
                                          st.session_state.sentences_splitted_by_duration, 
                                          st.session_state.splitted_words,
                                          broll_boundaries,
                                          int(st.session_state.total_frames/st.session_state.fps/60),
                                          word_data
                                          )
                        # if out_message:
                        #     st.write(out_message)
                        #     st.session_state.predictions = predictor.get_predictions(
                        #     splitted_sentences,
                        #     num_inputs=len(splitted_sentences),
                        #     prev_preds=st.session_state.predictions, 
                        #     out_message = out_message
                        #         )
                            
                        with open(json_file_path, "w") as f:
                            json.dump(st.session_state.predictions, f)
                    st.success(f"Predictions saved to {json_file_path}")
                else:
                    with open(json_file_path, "r") as f:
                        st.session_state.predictions = json.load(f)
                    st.info(f"Loaded existing predictions from {json_file_path}")
                    
                    # out_message = prediction_checks(st.session_state.predictions,
                    #                       st.session_state.sentences_splitted_by_duration, 
                    #                       st.session_state.splitted_words, 
                    #                       broll_boundaries,
                    #                       int(st.session_state.total_frames/st.session_state.fps/60),
                    #                       word_data
                    #                       )
                    # if out_message:
                    #     st.write(out_message)
                    #     st.session_state.predictions = predictor.get_predictions(
                    #                                     splitted_sentences,
                    #                                     num_inputs=len(splitted_sentences), 
                    #                                     prev_preds=st.session_state.predictions,
                    #                                     out_message = out_message
                    #         )
                    #     with open(json_file_path, "w") as f:
                    #         json.dump(st.session_state.predictions, f)
            # st.session_state.predictions = predictions

        if st.session_state.predictions:

            [col1] = st.columns(1)

            # with col1:
            #     if st.button("Fast Zoom In-Cut"):
            #         st.session_state.button_clicked = "fast_zoom_cut"
            with col1:
                if st.button("Fast Zoom In-Hold-Cut"):
                    st.session_state.button_clicked = "fast_zoom_hold_cut"
                    
        # Handle the action after the button click
        # if st.session_state.button_clicked == "fast_zoom_cut":
        #     st.write("Slow Zoom In-Cut clicked!")
        #     try:
        #         with st.spinner("Processing video..."):
        #             st.session_state.zoom_effects = get_zooms_claude(st.session_state.predictions, st.session_state.sentences_splitted_by_duration, st.session_state.splitted_words, slow=False, jumpcut=True, hold=False)
        #             st.session_state.output_path = process_video(video_path, st.session_state.zoom_effects)
        #             st.session_state.button_clicked = None

        #     except Exception as e:
        #         st.error(f"An error occurred during processing: {str(e)}")
        if st.session_state.button_clicked == "fast_zoom_hold_cut":
            st.write("Fast Zoom In-Hold-Cut clicked!")
            
            # Initialize button_clicked_ease in session state if it doesn't exist
            if 'button_clicked_ease' not in st.session_state:
                st.session_state.button_clicked_ease = "linear"
            
            # Create columns for easing function buttons
            [linear_, ease_in_, ease_out_, ease_in_out_] = st.columns(4)
            
            # Easing function buttons
            with linear_:
                if st.button("Linear", key="linear_btn"):
                    st.session_state.button_clicked_ease = "linear"
            with ease_in_:
                if st.button("Ease In", key="ease_in_btn"):
                    st.session_state.button_clicked_ease = "ease_in"
            with ease_out_:
                if st.button("Ease Out", key="ease_out_btn"):
                    st.session_state.button_clicked_ease = "ease_out"
            with ease_in_out_:
                if st.button("Ease In-Out", key="ease_in_out_btn"):
                    st.session_state.button_clicked_ease = "ease_in_out"
            
            # Display currently selected easing function
            st.write(f"Selected easing function: {st.session_state.button_clicked_ease}")
            
            # Process button to start video processing
            if st.button("Process Video", key="process_video_btn"):
                try:
                    with st.spinner("Getting zooms from claude..."):
                        zoom_effects = get_zooms_claude(
                            st.session_state.predictions,
                            st.session_state.sentences_splitted_by_duration,
                            st.session_state.splitted_words,
                            zoom_in_duration=ZOOM_DURATION,
                            slow=False,
                            jumpcut=True,
                            hold=True,
                        )
                        st.session_state.zoom_in_times = []
                        for effect in zoom_effects:
                            st.session_state.zoom_in_times.append(
                                f"{int(effect.start_time//60)}m{int(effect.start_time%60)}s"
                            )
                    
                    with st.spinner("Process zooms after face detection ..."):
                        zoom_scales, processed_centers = process_zoom_scales_centers_after_extracting_boundaries(
                            zoom_effects,
                            st.session_state.total_frames,
                            st.session_state.fps,
                            st.session_state.height,
                            st.session_state.width,
                            st.session_state.bounding_box_coordinates,
                            st.session_state.button_clicked_ease
                        )
                    
                    with st.spinner("Process video..."):
                        st.session_state.output_path = process_video(
                            video_path, zoom_scales, processed_centers
                        )
                            
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
            

        
        if st.session_state.zoom_effects:
            zoom_in_times = []
            for effect in st.session_state.zoom_effects:
                zoom_in_times.append(
                    f"{int(effect.start_time//60)}m{int(effect.start_time%60)}s"
                )

        if st.session_state.output_path:
            
            # zoom_in_times = []
            # for effect in zoom_effects:
            #     zoom_in_times.append(
            #         f"{int(effect.start_time//60)}m{int(effect.start_time%60)}s"
            #     )
            
            # Your time selection logic
            selected_time = st.selectbox(
                "Select a Zoom-in Start Time (or leave empty to play as it is):",
                options=["Play as it is"] + st.session_state.zoom_in_times,
            )

            # Convert selected time to seconds
            selected_seconds = None
            if selected_time != "Play as it is":
                minutes, seconds = map(int, selected_time[:-1].split("m"))
                selected_seconds = minutes * 60 + seconds

            # Read video file as bytes
            with open(st.session_state.output_path, "rb") as video_file:
                video_bytes = video_file.read()

            # Encode to base64
            video_b64 = base64.b64encode(video_bytes).decode()

            # Create custom HTML component with seek functionality
            html_code = f"""
                <div style="width: 100%; height: 100%;">
                    <video id="videoPlayer" width="100%" height="100%" controls>
                        <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    <script>
                        var video = document.getElementById('videoPlayer');
                        video.addEventListener('loadedmetadata', function() {{
                            {"video.currentTime = " + str(selected_seconds) + ";" if selected_seconds is not None else ""}
                        }});
                    </script>
                </div>
            """
            components.html(html_code, height=400)


if __name__ == "__main__":
    main()