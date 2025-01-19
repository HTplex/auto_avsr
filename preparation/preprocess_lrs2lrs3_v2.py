import os
import ffmpeg
import math
import pickle
import shutil
import warnings
from tqdm import tqdm

# Import your own modules 
# pip install --upgrade av==9.2.0
from data.data_module import AVSRDataLoader
from transforms import TextTransform
from utils import save_vid_aud_txt, split_file

warnings.filterwarnings("ignore")

def load_landmarks(landmark_path):
    """
    Helper function to load landmarks if provided.
    Returns:
        landmarks or None
    """
    if landmark_path and os.path.exists(landmark_path):
        with open(landmark_path, "rb") as fp:
            return pickle.load(fp)
    return None

def create_directories(export_path):
    """
    Helper to create directories for videos and text files in the export path.
    """
    video_dir = os.path.join(export_path, "videos")
    text_dir = os.path.join(export_path, "texts")
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)
    return video_dir, text_dir

def split_video_audio(video_data, audio_data, seg_duration, fps, sr):
    """
    (Optional) Helper to split video/audio into chunks of seg_duration seconds.
    If you have precise alignments for text, you can adapt that logic here.
    
    This returns a list of (video_chunk, audio_chunk, index) tuples.
    """
    seg_frames = seg_duration * fps
    seg_samples = seg_duration * sr

    total_frames = len(video_data)
    total_samples = audio_data.size(1) if audio_data.ndim > 1 else len(audio_data)

    # Number of segments needed to cover all frames
    num_segs = math.ceil(total_frames / seg_frames)

    segments = []
    for i in range(num_segs):
        start_frame = i * seg_frames
        end_frame   = min((i+1) * seg_frames, total_frames)

        start_sample = i * seg_samples
        end_sample   = min((i+1) * seg_samples, total_samples)

        vid_chunk = video_data[start_frame:end_frame]
        aud_chunk = audio_data[:, start_sample:end_sample]

        segments.append((vid_chunk, aud_chunk, i))
    return segments

def process_videos(
    meta,
    export_path,
    seg_duration=16,
    fps=25,
    sr=16000,
    combine_av=False,
    split_long_videos=False
):
    """
    Processes a list of video entries and saves them in export_path.

    meta: list of dictionaries, each with keys:
        - 'video_path': path to the video file (required)
        - 'text': text for the entire video (required if you want transcript)
        - 'landmark_path': path to landmark file (optional)
    export_path: directory where outputs will be saved
    seg_duration: how many seconds each segment can be (default=16s)
    fps: frames per second for the video
    sr: sample rate for audio
    combine_av: whether to combine audio and video into a single MP4 (with audio track)
    split_long_videos: if True, splits the video/audio into multiple segments if they exceed seg_duration
    """
    # Instantiate your data loaders as needed
    vid_dataloader = AVSRDataLoader(modality="video", detector="retinaface", convert_gray=False, gpu_type="cuda")
    aud_dataloader = AVSRDataLoader(modality="audio")

    # Instantiate your text transform
    text_transform = TextTransform()

    # Create directories for saving outputs
    video_dir, text_dir = create_directories(export_path)

    # CSV-like manifest (optional): track everything you save
    manifest_path = os.path.join(export_path, "metadata.csv")
    f = open(manifest_path, "w")
    # Write a header (optional)
    f.write("video_filename,frames,token_ids\n")

    for item in tqdm(meta, desc="Processing videos"):
        video_path = item["video_path"]
        text_line  = item.get("text", "")
        landmark_path = item.get("landmark_path", None)

        # Load landmarks if given
        landmarks = load_landmarks(landmark_path)

        # Load video data / audio data
        try:
            video_data = vid_dataloader.load_data(video_path, landmarks)
            audio_data = aud_dataloader.load_data(video_path)
        except (UnboundLocalError, TypeError, OverflowError, AssertionError) as e:
            print(f"Skipping {video_path} due to load error: {e}")
            continue

        # If no data, skip
        if video_data is None or audio_data is None:
            continue

        # Decide how to handle splitting
        if split_long_videos:
            segments = split_video_audio(video_data, audio_data, seg_duration, fps, sr)
        else:
            segments = [(video_data, audio_data, None)]

        # Process each segment
        for seg_idx, (vid_chunk, aud_chunk, idx) in enumerate(segments):
            if vid_chunk is None or aud_chunk is None:
                continue
            if len(vid_chunk) == 0 or aud_chunk.size(1) == 0:
                continue

            # Build filenames
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            if idx is not None:
                # e.g. <originalname>_00.mp4, ...
                vid_filename = f"{base_name}_{idx:02d}.mp4"
                aud_filename = f"{base_name}_{idx:02d}.wav"
                txt_filename = f"{base_name}_{idx:02d}.txt"
            else:
                vid_filename = f"{base_name}.mp4"
                aud_filename = f"{base_name}.wav"
                txt_filename = f"{base_name}.txt"

            out_vid_path = os.path.join(video_dir, vid_filename)
            out_aud_path = os.path.join(video_dir, aud_filename)
            out_txt_path = os.path.join(text_dir, txt_filename)

            # Save video/audio/text
            save_vid_aud_txt(
                out_vid_path,
                out_aud_path,
                out_txt_path,
                vid_chunk,
                aud_chunk,
                text_line,
                video_fps=fps,
                audio_sample_rate=sr
            )

            # Optionally combine audio and video
            if combine_av:
                in_vid = ffmpeg.input(out_vid_path)
                in_aud = ffmpeg.input(out_aud_path)
                out = ffmpeg.output(
                    in_vid["v"],
                    in_aud["a"],
                    out_vid_path[:-4] + ".av.mp4",
                    vcodec="copy",
                    acodec="aac",
                    strict="experimental",
                    loglevel="panic",
                )
                out.run()
                # Move the .av.mp4 to replace the original .mp4
                shutil.move(out_vid_path[:-4] + ".av.mp4", out_vid_path)

            # Prepare data for CSV logging
            frames_count = vid_chunk.shape[0]
            token_ids = text_transform.tokenize(text_line)
            token_str = " ".join(str(t.item()) for t in token_ids)

            # Write to the manifest file
            f.write(f"{vid_filename},{frames_count},{token_str}\n")

    f.close()
    print(f"Finished processing. Manifest written to {manifest_path}")