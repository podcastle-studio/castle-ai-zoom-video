import os
import concurrent.futures
from emphassess.src.emphasis_classifier.utils.infer_utils import (
    infer_audio,
    Wav2Vec2ForAudioFrameClassification,
)
import torch
import time

def process_file(f, splitted_audio_txt_dir, model, device=None):
    # Get predictions and emphasis boundaries for the audio file
    pred, emph_boundaries = infer_audio(f, model, device)
    # print("Emphasis boundaries (in seconds): ", emph_boundaries)
    
    # Get the base name of the audio file
    output_filename = f.split("/")[-1].split(".")[0] + ".txt"
    # Write the results to the output text file
    with open(os.path.join(splitted_audio_txt_dir, output_filename), "w") as f_out:
        print(os.path.join(splitted_audio_txt_dir, output_filename))
        for start, end in emph_boundaries:
            f_out.write(f"{start:.2f}-{end:.2f}\n")

    print(
        f"Emphasized intervals saved to {os.path.join(splitted_audio_txt_dir, output_filename)}"
    )

def save_emphasis_predictions(files, splitted_audio_txt_dir, device=None):
    if not os.path.exists(splitted_audio_txt_dir):
    # or (
    #     os.path.exists(splitted_audio_txt_dir)
    #     and len(os.listdir(splitted_audio_txt_dir)) != len(files):
        print(len(files))
        
        if device is None:
            #device="cpu"
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        # Load the model
        model = Wav2Vec2ForAudioFrameClassification.from_pretrained(
            "emphassess/src/emphasis_classifier/checkpoints/"
        ).to(device)

        # Create output directory if it does not exist
        os.makedirs(splitted_audio_txt_dir, exist_ok=True)
        
        # for f in files:
        #     process_file(f, splitted_audio_txt_dir, model, device)

        # Use ThreadPoolExecutor to process the files concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Submit tasks to the thread pool
            futures = [
                executor.submit(process_file, f, splitted_audio_txt_dir, model, device) 
                for f in files
            ]
            
            # Wait for all futures to complete
            for future in concurrent.futures.as_completed(futures):
                pass  # We don't need the result here, just wait for all tasks to finish