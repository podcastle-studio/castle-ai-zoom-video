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
import hashlib
from pathlib import Path
from utils import save_audio_from_video, check_ffmpeg, split_sentences_by_seconds, get_word_indices, split_words_by_duration, add_silence_duration, split_and_save_audio, save_emphasis_predictions, construct_new_sentences, add_sentences_to_file   
from asr import get_client_settings, transcribe_audio
import json
from predictor import ClaudeAdapter
from zoom_effect import ZoomEffect, process_video
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

SPLIT_SENTENCE_BY_DURATION = 240 * 7

# Set up logging configuration
def  get_zooms(preds, sntnces_splitted_by_duration, splittedwords, slow=False, jumpcut=False):
    zoom_effects = []
    for i, pred in enumerate(preds):
        prediction = pred["zoom_events"]
        for p in prediction:
            sentence_num = p['sentence_number']
            text_applied = p['text_applied']
            #reason = p['reason']
            zoom_in_scale = p['zoom_in_scale']
            zoom_out_duration = p['zoom_out_duration']
            st_idx, end_idx = get_word_indices(sntnces_splitted_by_duration[i][sentence_num-1], text_applied)
            if st_idx == -1 or end_idx == -1:
                continue
            start_time = splittedwords[i][sentence_num-1][st_idx][1]
            end_time = splittedwords[i][sentence_num-1][end_idx][2]
            if slow and not jumpcut:
                zoom_effects.append(ZoomEffect(start_time, end_time, end_time-start_time, zoom_in_scale, zoom_out_duration=zoom_out_duration, lag_time=0))
            elif not slow and not jumpcut:
                zoom_effects.append(ZoomEffect(start_time, end_time, 1, zoom_in_scale, 1))   
            elif slow and jumpcut:
                zoom_effects.append(ZoomEffect(start_time, end_time, end_time-start_time, zoom_in_scale, 0, lag_time=0))

    return zoom_effects


def  get_zooms_claude(preds, sntnces_splitted_by_duration, splittedwords, slow=False, jumpcut=False, hold=False):
    zoom_effects = []
    for i, pred in enumerate(preds):
        prediction = pred.get(list(pred.keys())[0], [])
        for p in prediction:
            sentence_num = p['sentence_number']
            text_applied = p['zoom_in_phrase']
            #reason = p['reason']
            zoom_in_scale = 1.3
            transition_sentence_num = p['transition_sentence_number']
            transition_sentence_word = p['transition_sentence_word']

            # zoom_out_duration = p['zoom_out_duration']
            st_idx, end_idx = get_word_indices(sntnces_splitted_by_duration[i][sentence_num-1], text_applied)
            st_idx_cut, end_idx_cut = get_word_indices(sntnces_splitted_by_duration[i][transition_sentence_num-1], transition_sentence_word)

            if st_idx == -1 or end_idx == -1:
                continue
            if st_idx_cut == -1 or end_idx_cut == -1:
                continue
            start_time = splittedwords[i][sentence_num-1][st_idx][1]
            end_time = splittedwords[i][transition_sentence_num-1][st_idx_cut][1]
            
            if not slow and jumpcut and hold:
                zoom_effects.append(ZoomEffect(start_time, end_time, 1, zoom_in_scale, 0))   
            elif not slow and jumpcut and not hold:
                zoom_effects.append(ZoomEffect(start_time, end_time, 1, zoom_in_scale, 0, lag_time=0))

    return zoom_effects


def main():
    st.title("Video Zoom In Editor")
    
    # Check for FFMPEG
    if not check_ffmpeg():
        st.error("""
        FFMPEG is not installed. Please install it:
        - Windows: Download from https://ffmpeg.org/download.html
        - Mac: brew install ffmpeg
        - Linux: sudo apt-get install ffmpeg
        """)
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", 
                                     accept_multiple_files=False,
                                     type=['mp4', 'avi', 'mov'])
    
    if uploaded_file:
        # Create temporary directory for uploads
        temp_dir = Path("./uploaded_files/recordings/video_recordings")
        temp_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        video_path = str(temp_dir / uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Display original video
        st.subheader("Original Video")
        st.video(video_path)

        # Transcribe audio and save transcription in a file
        if uploaded_file or not st.session_state.get('data_uploaded', False):

            st.session_state.data_uploaded = True

            st.session_state.interview_id = hashlib.md5(
                f'{uploaded_file.name}'.encode('utf-8')
            ).hexdigest()
            
            save_audio_from_video(video_path,
                                  video_path.replace('.mp4', '.mp3'))

            # Clear output
            # st.write("Before clearing session state ", st.session_state)
            # clear_cache(st.session_state)
            # st.write("After clearing session state ", st.session_state)

            st.session_state.audio_file = video_path.replace('.mp4', '.mp3')
            st.session_state.video_file = video_path
            st.subheader("Transcribe Audio")

            #transcribe audios
            if 'asr_client_settings' not in st.session_state:
                st.session_state['asr_client_settings'] = get_client_settings()
            
            output_file_path_sentence = st.session_state.audio_file \
                .replace('recordings', 'transcriptions') \
                .replace('.mp3', '_trancriptions_with_align_sentence.txt')
            os.makedirs(os.path.dirname(output_file_path_sentence), exist_ok=True)

            output_file_path_words = st.session_state.audio_file \
                .replace('recordings', 'transcriptions') \
                .replace('.mp3', '_trancriptions_with_align_words.json')
            st.session_state.interview_to_transcription_meta_sentence, st.session_state.interview_to_transcription_meta_words = transcribe_audio( 
            st.session_state.audio_file,
            output_file_path_sentence,
            output_file_path_words,
            st.session_state['asr_client_settings']
        )
            
        #split audios by sentence to predict
        splitted_audio_dir = f"./uploaded_files/recordings/splitted_audios/{video_path.split('/')[-1].split('.')[0]}"
        os.makedirs(splitted_audio_dir, exist_ok=True)

        split_and_save_audio(st.session_state.audio_file, output_file_path_sentence, splitted_audio_dir)

        # Run emphasis model and change the sentence txt file
        files = glob(f"{splitted_audio_dir}/*.mp3")
        splitted_audio_txt_dir = f"./uploaded_files/emphasis_detection/{video_path.split('/')[-1].split('.')[0]}"

        save_emphasis_predictions(files,
                                  splitted_audio_txt_dir)
        audio_basename = os.path.basename(st.session_state.audio_file)
        # add silence duration to the words
        #with open(st.session_state.interview_to_transcription_meta_words) as f:
        word_data = st.session_state.interview_to_transcription_meta_words#json.load(f)
        add_silence_duration(word_data)
        
        #Change the sentence capitalization
        files = glob(f"{splitted_audio_txt_dir}/*.txt")
        sentence_info_path_updated = output_file_path_sentence.replace('.txt', '_updated.txt')

        new_sentences = construct_new_sentences(files, audio_basename, word_data, sentence_info_path_updated, output_file_path_sentence) 
        numbered_txt_file = sentence_info_path_updated.replace('_updated.txt', '_updated_numbered.txt')
        if not os.path.exists(numbered_txt_file):
            for i, sent in enumerate(new_sentences):
                add_sentences_to_file(f"{i}. {sent}", numbered_txt_file)


        #ChatGPT predictions
        # if st.button("GPT Predictions"):
        #     print("No any gpt predictions")
        #     sentences_splitted_by_duration = split_sentences_by_seconds(st.session_state.interview_to_transcription_meta_sentence, SPLIT_SENTENCE_BY_DURATION )
        #     splitted_words = split_words_by_duration(st.session_state.interview_to_transcription_meta_words, [len(sen) for sen in sentences_splitted_by_duration] )
        #     splitted_sentences = [[f"{i}. {sentence}" for i, sentence in enumerate(sentences, start=1)] for sentences in sentences_splitted_by_duration]

        #     predictor = GPTAdapter(model="gpt-4o", 
        #                            api_key=os.get_env("OPENAI_API_KEY"))
        
        #     os.makedirs("gpt_results", exist_ok=True)
        #     zoom_effects = []

        #     # Construct the JSON file path
        #     audio_file_name = st.session_state.audio_file.split('/')[-1].split('.')[0]
        #     json_file_path = f"gpt_results/{audio_file_name}.json"

        #     # Check if predictions already exist
        #     if not os.path.exists(json_file_path):
        #         # Generate predictions using GPT predictor
        #         predictions = predictor.get_predictions(splitted_sentences)
        #         # Save predictions to a JSON file
        #         with open(json_file_path, "w") as f:
        #             json.dump(predictions, f)
        #         st.success(f"Predictions saved to {json_file_path}")
        #     else:
        #         # Load existing predictions
        #         with open(json_file_path, "r") as f:
        #             predictions = json.load(f)
        #         st.info(f"Loaded existing predictions from {json_file_path}")
                        


                # zoom_effects.append(ZoomEffect(start_time, (end_time-start_time), zoom_in_scale))

                #Zoom effects controls
                # st.subheader("Add Zoom In Effects")
                
                # zoom_effects = []
                # num_effects = st.number_input("Number of zoom in points", min_value=1, max_value=5, value=1)
                
                # for i in range(num_effects):
                #     st.write(f"Zoom In Effect {i+1}")
                #     col1, col2, col3 = st.columns(3)
                    
                #     with col1:
                #         start_time = st.number_input(f"Start Time (ms) #{i+1}", 
                #                                 min_value=0, 
                #                                 value=1000*i)
                #     with col2:
                #         duration = st.number_input(f"Duration (ms) #{i+1}", 
                #                                 min_value=100, 
                #                                 value=1000)
                #     with col3:
                #         scale = st.slider(f"Final Zoom Scale #{i+1}", 
                #                         min_value=1.0, 
                #                         max_value=3.0, 
                #                         value=1.5, 
                #                         step=0.1)
                        
                #     zoom_effects.append(ZoomEffect(start_time, duration, scale))
        
            # Process button
        # st.write(st.session_state)
        if "button_clicked" not in st.session_state:
            st.session_state.button_clicked = None
        
        if "predictions" not in st.session_state:
            st.session_state.predictions = None
        if "zoom_effects" not in st.session_state:
            st.session_state.zoom_effects = None
        if "output_path" not in st.session_state:
            st.session_state.output_path = None
        # st.write(st.session_state)


        if st.button("Claude Predictions"):
            st.session_state.button_clicked = "claude_predictions"
            predictor = ClaudeAdapter(model_name="claude-3-5-sonnet-20241022", 
                                    api_key=os.getenv('ANTHROPIC_API_KEY'))

            os.makedirs("claude_results", exist_ok=True)
            st.session_state.sentences_splitted_by_duration = split_sentences_by_seconds(new_sentences, SPLIT_SENTENCE_BY_DURATION)
            st.session_state.splitted_words = split_words_by_duration(word_data, [len(sen) for sen in st.session_state.sentences_splitted_by_duration])
            splitted_sentences = [[f"{i}. {sentence}" for i, sentence in enumerate(sentences, start=1)] for sentences in st.session_state.sentences_splitted_by_duration]

            # Construct the JSON file path
            audio_file_name = st.session_state.audio_file.split('/')[-1].split('.')[0]
            json_file_path = f"claude_results/{audio_file_name}.json"
            if not os.path.exists(json_file_path):
                st.session_state.predictions = predictor.get_predictions(splitted_sentences, num = len(splitted_sentences))
                with open(json_file_path, "w") as f:
                    json.dump(st.session_state.predictions, f)
                st.success(f"Predictions saved to {json_file_path}")
            else:
                with open(json_file_path, "r") as f:
                    st.session_state.predictions = json.load(f)                    
                st.info(f"Loaded existing predictions from {json_file_path}")
            # st.session_state.predictions = predictions

        output_path = None
        if st.session_state.predictions:

            [col1]= st.columns(1)

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
            try:
                with st.spinner("Processing video..."):
                    st.session_state.zoom_effects = get_zooms_claude(st.session_state.predictions, st.session_state.sentences_splitted_by_duration, st.session_state.splitted_words, slow=False, jumpcut=True, hold=True)
                    st.session_state.output_path = process_video(video_path, st.session_state.zoom_effects)
                    st.session_state.button_clicked = None
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")

        
        if st.session_state.zoom_effects:
            zoom_in_times = []
            for effect in st.session_state.zoom_effects:
                zoom_in_times.append(f"{int(effect.start_time//60)}m{int(effect.start_time%60)}s")
        
        if st.session_state.output_path:
            # Your time selection logic
            selected_time = st.selectbox(
                "Select a Zoom-in Start Time (or leave empty to play as it is):",
                options=["Play as it is"] + zoom_in_times
            )

            # Convert selected time to seconds
            selected_seconds = None
            if selected_time != "Play as it is":
                minutes, seconds = map(int, selected_time[:-1].split("m"))
                selected_seconds = minutes * 60 + seconds

            # Read video file as bytes
            with open(st.session_state.output_path, 'rb') as video_file:
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
            
            # Render the component
            components.html(html_code, height=400)
               
if __name__ == "__main__":
    main()
