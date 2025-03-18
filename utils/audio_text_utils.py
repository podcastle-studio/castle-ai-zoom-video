from moviepy.editor import VideoFileClip
from typing import Dict, List
import subprocess
import librosa
import logging
import os
import re
import torchaudio as ta
from tqdm import tqdm
from glob import glob
from span_api_splitter.utils import get_split_info
from span_api_splitter import config
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler("demo_logs.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def has_one_second_intersection(interval1, interval2):
    # Unpack the tuples
    start1, end1 = interval1
    start2, end2 = interval2
    
    # Find intersection
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    
    # Check if intersection is at least 1 second
    return intersection_end - intersection_start >= 1

def find_intersecting_intervals(broll_boundaries, other_times):
    intersecting_times = []
    
    for time_interval in other_times:
        # Check if this interval intersects with any broll
        for broll in broll_boundaries:
            if has_one_second_intersection(broll, time_interval):
                intersecting_times.append(time_interval)
                break  # Once we find one intersection, we can move to next interval
                
    return intersecting_times


def construct_output_message(number_is_not_qualified,pred_length, duration, not_found_start, not_found_transition, shorts, intersected_indices):
    messages = []
#     if number_is_not_qualified:
#         messages.append(
#             f"ERROR: Invalid number of zoom-in moments. Found {pred_length} zoom-ins but video duration requires exactly "
#             f"{int(duration)} zoom-ins. Please ensure there are approximately {int(duration)} zoom-ins, if possible, considering the B-roll segments"
# )
    if shorts:
        messages.append(
            f"ERROR: Insufficient spacing for zoom-ins at indices {shorts}. The gap between zoom-in end and jump cut must "
            f"be at least 3 seconds (considering zoom-in duration of 1 second). Please adjust these transitions to maintain "
            f"proper timing and pacing."
        )

    if not_found_start:
        messages.append(
            f"ERROR: Invalid zoom-in phrases at indices {not_found_start}. The specified zoom-in phrases were not found in "
            f"their respective sentences. Please verify that the zoom-in phrases exactly match the transcript text."
        )
        
    if not_found_transition:
        messages.append(
            f"ERROR: Invalid transition points at indices {not_found_transition}. The specified transition phrases were not "
            f"found in their respective sentences. Please ensure transition points exactly match the transcript text."
        )
    
    if intersected_indices:
         messages.append(
            f"ERROR: B-roll conflicts detected at indices {intersected_indices}. Zoom-ins and transitions must not overlap "
            f"with any B-roll segments. Please relocate these zoom-ins to non-B-roll portions of the video."
        )
    if messages:
        messages.append("Do not apologize, just give me the correct version with the defined structure")
    
        return "\n\n".join(messages)
    else:
        return []
     

        
def prediction_checks(preds, sntnces_splitted_by_duration, splitted_words, broll_boundaries, duration, word_data):
    not_found_start = []
    not_found_transition = []
    shorts = []
    predictions_times = []
    broll_boundary_times = []
    number_is_not_qualified = False
    for i, pred in enumerate(preds):

        prediction = pred.get(list(pred.keys())[0], [])
        if len(prediction) < duration * 0.75:
            number_is_not_qualified = True
        else:
            return []
        broll_boundary_times.extend([(word_data[end_sent_idx][end_word_idx][2], word_data[st_sent_idx][st_word_idx][2]) for (st_sent_idx, st_word_idx, end_sent_idx, end_word_idx) in broll_boundaries])

            
        for j, p in enumerate(prediction):
            sentence_num = p["sentence_number"]
            text_applied = p["zoom_in_phrase"]
            # reason = p['reason']
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
            # check for not found
            if st_idx == -1 or end_idx == -1:
                not_found_start.append((i+1, j+1))
            if st_idx_cut == -1 or end_idx_cut == -1:
                not_found_transition.append((i + 1, j + 1))
            
            
            start_time = splitted_words[i][sentence_num][st_idx][1]
            end_time = splitted_words[i][transition_sentence_num][st_idx_cut][1]
            predictions_times.append((start_time, end_time))
            if end_time - start_time < 4:
                shorts.append((i+1, j + 1))
             
    intersected_indexes = find_intersecting_intervals(broll_boundary_times, predictions_times)
    out_message = construct_output_message(number_is_not_qualified, len(prediction), duration, not_found_start, not_found_transition, shorts, intersected_indexes )
    return out_message

def split_words_by_duration(words: List, sentences_lengths: List):
    splitted_words = []
    for i, length in enumerate(sentences_lengths):
        splitted_words.append(words[:length])
        words = words[length:]
    return splitted_words


def save_audio_from_video(video_path: str, output_path: str):
    """
    Save audio from video

    :param video_path: str: Video path
    :param output_path: str: Output path
    """
    if os.path.exists(output_path):
        logger.info(f"Audio file already exists: {output_path}")
        return
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_path)


def clear_cache(session_state: Dict):
    """
    Clear the session state cache.
    """
    keys = list(key for key in session_state.keys() if key != "interview_id")
    for key in keys:
        session_state.pop(key)


def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False


def extract_sentences_with_durations_with_chunks(
    transcript_metadata: dict, chunk_start: float = None
) -> List[str]:
    """
    Extracts sentences with their start and end times from the transcript metadata.

    Args:
        transcript_metadata (dict): The transcript metadata.

    Returns:
        list: Sentence concated with it's start and end times.
    """
    sentences_with_durations = []
    transcripts_with_alignments = transcript_metadata.get("results", [])
    sentence = ""
    sentence_with_words = []
    for transcript in transcripts_with_alignments:
        word = transcript["alternatives"][0]["content"]

        if sentence == "":
            sentence_with_words.append([])
            start_time_sentence = transcript["start_time"]
            start_time = transcript["start_time"]
            end_time = transcript["end_time"]
            sentence += word + " "
            sentence_with_words[-1].append(
                [word, start_time + chunk_start, end_time + chunk_start]
            )
        elif transcript.get("is_eos", False):  # End of sentence
            start_time = transcript["start_time"]
            end_time = transcript["end_time"]
            sentence = sentence.strip() + word + " "
            sentences_with_durations.append(
                {
                    "sentence": sentence,
                    "start_time": start_time_sentence + chunk_start,
                    "end_time": end_time + chunk_start,
                }
            )
            # sentence_with_words[-1].append([word, start_time, end_time])
            sentence = ""
        elif transcript["type"] == "punctuation":
            sentence = sentence.strip() + word + " "
        else:
            start_time = transcript["start_time"]
            end_time = transcript["end_time"]
            sentence_with_words[-1].append(
                [word, start_time + chunk_start, end_time + chunk_start]
            )
            sentence += word + " "

    return sentences_with_durations, sentence_with_words


def split_sentences_by_seconds(sentences_metadata: dict, seconds: int) -> List[str]:
    """
    Extracts sentences with their start and end times from the transcript metadata.

    Args:
        transcript_metadata (dict): The transcript metadata.

    Returns:
        list: Sentence concated with it's start and end times.
    """
    vrk = 0
    sentences = [[]]
    for sentence in sentences_metadata:
        match = re.search(r"\|(\d+\.\d+)$", sentence)
        if match:
            extracted_number = float(match.group(1))
            sentences[-1].append(sentence)
            if extracted_number // seconds > vrk:
                vrk += 1
                sentences.append([])
        else:
            print("No number found after sentence")

    return sentences


def get_word_indices(full_text, target_text):
    # Clean and split texts
    full_text = re.sub(r'\[\d+(\.\d+)?s\]|\[|\]|#', '', full_text)
    full_words = full_text.strip().split()
    full_words = full_words[:-1]
    target_words = target_text.strip().split()

    if len(target_words) <= 2:
        required_matches = len(target_words)
        
    elif len(target_words) > 2 and len(target_words) <= 4:
        required_matches = len(target_words) -1
    else:
        required_matches = len(target_words) -2

    start_idx = -1
    for i in range(len(full_words) - len(target_words) + 1):
        # Slice of words to compare
        current_slice = full_words[i : i + len(target_words)]

        # Count matching words
        matches = sum(1 for x, y in zip(current_slice, target_words) if x == y)

        # Check if the match count meets or exceeds the required threshold
        if matches >= required_matches:
            start_idx = i
            break
            # print("Match found at position:", i)

    if start_idx == -1:
        print("Target text not found in full text")
        return -1, -1

    end_idx = start_idx + len(target_words) - 1
    return start_idx, end_idx


def add_silence_duration(word_data):
    start = None
    for i, d in enumerate(word_data):
        for j, w in enumerate(d):
            if j == 0 and start == None:
                start = 0
            else:
                start = d[j - 1][2]
            if w[1] - start > 0.03:
                w.append(w[1] - start)
            else:
                w.append(0)
        start = w[2]


@dataclass
class Segment:
    start: float
    end: float
    index: int

def parse_segments(output_file_path_sentence) -> List[Segment]:
    """Parse and sort all segments from the file."""
    segments = []
    with open(output_file_path_sentence, "r") as file:
        for i, line in enumerate(file):
            line_stripped = line.strip()
            start = float(line_stripped.split("|")[-2])
            end = float(line_stripped.split("|")[-1])
            segments.append(Segment(start, end, i))
    
    # Sort segments by start time for efficient processing
    return sorted(segments, key=lambda x: x.start)

def process_segment_chunk(args):
    """Process a group of segments that can be loaded together."""
    audio_path, segment_group, sr, splitted_audio_dir = args
    
    # Calculate the range needed for this group
    start_time = segment_group[0].start
    end_time = segment_group[-1].end
    
    # Load only the required portion of audio
    frame_offset = int(start_time * sr)
    num_frames = int((end_time - start_time) * sr)
    
    try:
        audio_chunk, sr = ta.load(
            audio_path,
            frame_offset=frame_offset,
            num_frames=num_frames
        )
        
        # Process each segment in the loaded chunk
        for segment in segment_group:
            # Calculate relative positions within the loaded chunk
            rel_start = int((segment.start - start_time) * sr)
            rel_end = int((segment.end - start_time) * sr)
            
            # Extract and save segment
            segment_audio = audio_chunk[:, rel_start:rel_end]
            
            # Create output filename
            base_name = os.path.splitext(os.path.basename(audio_path))
            output_filename = os.path.join(
                splitted_audio_dir,
                f"{base_name[0]}_{segment.index}{base_name[1]}"
            )
            
            # Save with larger buffer for speed
            ta.save(output_filename, segment_audio, sr, buffer_size=16384)
            
    except Exception as e:
        print(f"Error processing segments from {start_time:.2f}s to {end_time:.2f}s: {str(e)}")

def group_segments(segments: List[Segment], max_chunk_duration: float = 30.0) -> List[List[Segment]]:
    """Group segments into chunks that can be processed together."""
    groups = []
    current_group = []
    
    for segment in segments:
        if not current_group:
            current_group.append(segment)
        else:
            # Check if adding this segment would exceed max chunk duration
            if segment.end - current_group[0].start <= max_chunk_duration:
                current_group.append(segment)
            else:
                groups.append(current_group)
                current_group = [segment]
    
    if current_group:
        groups.append(current_group)
    
    return groups

def split_and_save_audio(audio_path: str, output_file_path_sentence: str, 
                        splitted_audio_dir: str, max_workers: int = 10):
    """
    Memory-efficient audio splitting for long files.
    """
    if not os.path.exists(splitted_audio_dir):
        os.makedirs(splitted_audio_dir)
    
    if not os.listdir(splitted_audio_dir):
        # Get sample rate once
        sr = librosa.get_samplerate(audio_path)
        
        # Parse all segments
        print("Parsing segments...")
        segments = parse_segments(output_file_path_sentence)
        
        # Group segments for efficient processing
        print("Grouping segments...")
        segment_groups = group_segments(segments)
        
        # Prepare arguments for processing
        args = [
            (audio_path, group, sr, splitted_audio_dir)
            for group in segment_groups
        ]
        
        # Process groups in parallel
        print(f"Processing {len(segment_groups)} chunks with {max_workers} workers...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(
                executor.map(process_segment_chunk, args),
                total=len(args),
                desc="Processing audio chunks"
            ))
            
        print("Audio splitting complete!")
        
    else:
        print(f"Directory {splitted_audio_dir} is not empty. Skipping processing.")

# def process_line(args):
#     """Helper function to process a single line for splitting and saving."""
#     audio_path, line, i, sr, splitted_audio_dir = args
#     line_stripped = line.strip()
#     start = float(line_stripped.split("|")[-2])
#     end = float(line_stripped.split("|")[-1])
#     y, sr = ta.load(
#         audio_path,
#         frame_offset=int(start * sr),
#         num_frames=int((end - start) * sr),
#     )
#     output_filename = (
#         f"{splitted_audio_dir}/{audio_path.split('/')[-1].split('.')[0]}_{i}."
#         f"{audio_path.split('/')[-1].split('.')[1]}"
#     )
#     ta.save(output_filename, y, sr)

# def split_and_save_audio(audio_path, output_file_path_sentence, splitted_audio_dir, max_workers=10):
#     """
#     Splits an audio file into segments defined in a sentence file and saves the segments.
    
#     Parameters:
#     - audio_path: str, path to the audio file.
#     - output_file_path_sentence: str, path to the file containing sentence information.
#     - splitted_audio_dir: str, directory to save the split audio files.
#     - max_workers: int, number of threads for parallel processing.
#     """
#     sr = librosa.get_samplerate(audio_path)
    
#     if not os.listdir(splitted_audio_dir):  # Process only if the directory is empty
#         with open(output_file_path_sentence, "r") as file:
#             lines = file.readlines()
        
#         # Prepare arguments for each line
#         args = [
#             (audio_path, line, i, sr, splitted_audio_dir)
#             for i, line in enumerate(lines)
#         ]

#         # Parallel processing
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             list(tqdm(executor.map(process_line, args), total=len(args)))




def construct_new_sentences(
                            emphasized_files,
                            audio_basename, 
                            word_data,
                            broll_boundaries,
                            sentence_info_path_updated,
                            sentence_path
):
    pattern = re.compile(rf'{audio_basename.split(".")[0]}_(\d+)\.txt$')
    # Sort the file paths based on the extracted numeric part
    try:
        emphasized_files = sorted(emphasized_files, key=lambda x: int(pattern.search(x).group(1)))
    except:
        print("One emphasized file")
    with open(sentence_path, "r") as file:
        sentence_content = file.read()
        
    sentences = sentence_content.split("\n")[:-1]
    new_sentences = []
    start_sent_indexes = set()
    end_sentence_indexes = set()

    for broll_boundary in broll_boundaries:
        start_sent_idx, _, end_sent_idx, _ = broll_boundary
        start_sent_indexes.add(start_sent_idx)
        end_sentence_indexes.add(end_sent_idx)
    
    for i, f in tqdm(enumerate(emphasized_files)):
        with open(f, "r") as file:
            content = file.read()
        times = content.split("\n")
        for t in times[:-1]:
            st, end = t.split("-")
            start_time = float(st) + word_data[i][0][1]
            end_time = float(end) + word_data[i][0][1]
            if end_time - start_time >= 0.18:
                best_match = find_overlapping_words(word_data[i], start_time, end_time)
                for ind in best_match:
                    word_data[i][ind][0] = word_data[i][ind][0].upper()
                # if new_sentence is None:
                #     new_sentence = capitalize_word_by_index(sentences[i], best_match)
                # else:
                #     new_sentence = capitalize_word_by_index(new_sentence, best_match)
        
        
        fomatted_words = format_word_with_silence(word_data[i])
        for j, wrd_info in enumerate(word_data[i]):
            wrd_info[0] = fomatted_words[j]
        
        # if i in start_sent_indexes:
        #     word_indices_start = [broll[1] for broll in broll_boundaries if broll[0] == i]
        #     word_data[i] = format_word_with_opening_paranthesis(word_data[i], word_indices_start)
        # if i in end_sentence_indexes:
        #     word_indices_end = [broll[3] for broll in broll_boundaries if broll[2] == i]
        #     word_data[i] = format_word_with_closing_paranthesis(word_data[i], word_indices_end)
        
        
    
    for broll_boundary in broll_boundaries:
        start_sent_idx, start_word_index, end_sent_idx, end_word_index = broll_boundary
        
        if end_sent_idx - start_sent_idx > 0:
            for sent in range(start_sent_idx, end_sent_idx + 1):
                if sent == start_sent_idx:
                    word_data[start_sent_idx] = format_word_with_opening_vandakanish(word_data[start_sent_idx], [start_word_index])
                    word_data[start_sent_idx] = format_word_with_closing_vandakanish(word_data[start_sent_idx], [len(word_data[start_sent_idx])-1])
                elif sent == end_sent_idx:
                    word_data[sent] = format_word_with_opening_vandakanish(word_data[sent], [0])
                    word_data[sent] = format_word_with_closing_vandakanish(word_data[sent], [end_word_index])
                else:
                    word_data[sent] = format_word_with_opening_vandakanish(word_data[sent], [0])
                    word_data[sent] = format_word_with_closing_vandakanish(word_data[sent], [len(word_data[sent])-1])
        else:
            word_data[start_sent_idx] = format_word_with_opening_vandakanish(word_data[start_sent_idx], [start_word_index])
            word_data[end_sent_idx] = format_word_with_closing_vandakanish(word_data[end_sent_idx], [end_word_index])

    for i, f in tqdm(enumerate(emphasized_files)):
        new_s = new_sentence_from_words([word[0] for word in word_data[i]], sentences[i])
        add_sentences_to_file(new_s, sentence_info_path_updated)
        new_sentences.append(new_s + "\n")
    
    return new_sentences


def find_overlapping_words(word_times, start_time, end_time, overlap_threshold=0.5):
    """
    Find indices of words that overlap with a given time range by at least the specified threshold.

    Args:
        word_times: List of [word, start, end] lists
        start_time: Start time of the range to check
        end_time: End time of the range to check
        overlap_threshold: Minimum overlap ratio required (default: 0.5)

    Returns:
        List of indices where words meet the overlap threshold
    """
    overlapping_indices = []

    for i, (word, word_start, word_end, _) in enumerate(word_times):
        # Calculate word duration and overlap duration
        word_duration = word_end - word_start

        overlap_start = max(start_time, word_start)
        overlap_end = min(end_time, word_end)

        if overlap_end > overlap_start:  # There is some overlap
            overlap_duration = overlap_end - overlap_start
            overlap_ratio = overlap_duration / word_duration

            if overlap_ratio >= overlap_threshold:
                overlapping_indices.append(i)

    return overlapping_indices


def capitalize_word_by_index(sentence, indices):
    """
    Capitalizes words at specified indices in a sentence.

    Args:
        sentence: String containing the sentence and timing info
        indices: List of indices of words to capitalize

    Returns:
        Modified sentence with specified words capitalized
    """
    # Split the sentence and timing info
    text, start_time, end_time = sentence.rsplit("|", 2)

    # Split into words while preserving punctuation
    words = text.split()

    # Capitalize words at specified indices
    for idx in indices:
        if 0 <= idx < len(words):
            # Handle punctuation
            word = words[idx]
            punctuation = ""
            while word and word[-1] in ".,!?":
                punctuation = word[-1] + punctuation
                word = word[:-1]

            # Capitalize and reconstruct with punctuation
            words[idx] = word.upper() + punctuation

    # Reconstruct the sentence with timing info
    return f"{' '.join(words)}|{start_time}|{end_time}"


def format_word_with_silence(word_timings):
    """
    Format words data with silence indicators based on word timings.

    Args:
        word_timings: List of [word, start, end, silence] lists
        sentence: Original sentence with timing info

    Returns:
        Formatted sentence with silence indicators
    """
    # Split into words while preserving punctuation
    formatted_words = []

    for word_info in word_timings:
        word, _, _, silence = word_info

        # Add silence indicator if there's silence
        if silence > 0:
            formatted_words.append(f"[{silence:.2f}s] {word}")
        else:
            formatted_words.append(word)
            
    return formatted_words

def transform_text(input_str, sign, end=True):
    if " " in input_str:
        # Split by space and insert '[' before the second part
        parts = input_str.split(" ", 1)
        if not end:
            return f"{parts[0]} {sign}{parts[1]}"
        else:
            return f"{parts[0]} {parts[1]}{sign}"
    else:
        # If no space, just add '[' at the beginning
        if end:
            return f"{input_str}{sign}"
        else:
            return f"{sign}{input_str}"


def format_word_with_opening_vandakanish(word_timings, indices):
    """
    Format a sentence with silence indicators based on word timings.

    Args:
        word_timings: List of [word, start, end, silence] lists

    """
    # Split into words while preserving punctuation
    for ind in indices:
        word_timings[ind][0] = transform_text(word_timings[ind][0], "# ", end=False)
    return word_timings

def format_word_with_closing_vandakanish(word_timings, indices):
    """
    Format a sentence with closing paranthesis based on word timings.

    Args:
        word_timings: List of [word, start, end, silence] lists

    """
    # Split into words while preserving punctuation
    for ind in indices:
        word_timings[ind][0] = transform_text(word_timings[ind][0], " #")
    return word_timings

def new_sentence_from_words(words_data, sentence):
    # Reconstruct the sentence
    # Get the start and end times from the original sentence
    _, start_time, end_time = sentence.rsplit("|", 2)

    # Join words with spaces and add timing info
    formatted_sentence = " ".join(words_data) + f" |{start_time}|{end_time}"

    # Add punctuation
    formatted_sentence = formatted_sentence.replace(" ,", ",")
    formatted_sentence = formatted_sentence.replace(" .", ".")

    return formatted_sentence


def add_sentences_to_file(sentence, filename):
    """
    Appends new sentences to a file, ensuring each sentence is on a new line.

    Args:
        sentences (list): List of sentences to add.
        filename (str): Path to the text file.
    """
    # sentence = add_numbering_to_sentences(sentence, i)
    
    with open(filename, "a") as file:  # Open the file in append mode
        file.write(sentence + "\n")  # Write each sentence followed by a newline


def split_audio_into_chunks(audio_path: str) -> List[str]:
    """
    Split audio into chunks and return chunk paths.
    """
    outputDir = ".cache/tmp/"
    os.makedirs(outputDir, exist_ok=True)
    payload = {"fileUrl": audio_path, "resultId": "3c5c778553a73441ac9d57622ed4442a"}

    split_info = get_split_info(payload, config)
    logger.info(f"Split info: {split_info}")

    if not split_info:  # No splitting points
        # Convert the audio to wav
        command = f"ffmpeg -i {audio_path} -ac 1 {outputDir}/part_000000000.wav"
        os.system(command)
        return [f"{outputDir}/part_000000000.wav"]

    segments = ",".join([str(x) for x in split_info["splittingPoints"]])

    command = (
        "ffmpeg -i "
        + f'"{audio_path}"'
        + " "
        + "-f segment -segment_times "
        + segments
        + " -reset_timestamps 1 -map 0:a -c:a pcm_s16le "
        + outputDir
        + "/part_%09d.wav"
    )

    os.system(command)

    splitted_audio_files = sorted(
        glob(outputDir + "/part*.wav"),
        key=lambda x: int(os.path.basename(x)[: -len(".wav")].split("_")[-1]),
    )
    return splitted_audio_files


def extract_audio(input_video: str, output_audio: str):
    command = ["ffmpeg", "-i", input_video, "-vn", "-acodec", "aac", "-y", output_audio]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logging.error("Audio extraction failed: %s", result.stderr.decode())
        raise RuntimeError("Failed to extract audio")