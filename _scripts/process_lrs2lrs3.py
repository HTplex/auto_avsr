"""
Script for preprocessing LRS2 or LRS3 datasets.

Usage (Example):
---------------
    python preprocess_avsr.py --data-dir /path/to/LRS2/main \
                              --root-dir /path/to/output \
                              --subset train \
                              --dataset lrs2 \
                              --gpu_type cuda \
                              --detector retinaface \
                              --landmarks-dir /path/to/landmarks \
                              --seg-duration 16 \
                              --combine-av true \
                              --groups 4 \
                              --job-index 0
"""

import argparse
import glob
import math
import os
import pickle
import shutil
import warnings

import ffmpeg
from tqdm import tqdm

from data.data_module import AVSRDataLoader
from transforms import TextTransform
from utils import save_vid_aud_txt, split_file

warnings.filterwarnings("ignore")


def parse_args():
    """
    Parse command-line arguments and return them as an argparse Namespace.

    Returns
    -------
    argparse.Namespace
        An object containing all command-line arguments as attributes.

    Examples
    --------
    >>> args = parse_args()
    >>> print(args.data_dir, args.root_dir)
    /path/to/LRS2 /path/to/output
    """
    parser = argparse.ArgumentParser(description="LRS2/LRS3 Preprocessing")

    # Required arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the original dataset (LRS2 or LRS3)."
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Root directory for preprocessed outputs."
    )
    parser.add_argument(
        "--subset",
        type=str,
        required=True,
        help="Subset of the dataset to process (train/val/test)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset (lrs2 or lrs3)."
    )
    parser.add_argument(
        "--gpu_type",
        type=str,
        required=True,
        default="cuda",
        help="GPU type: 'cuda' or 'mps' (for Apple Silicon)."
    )

    # Optional arguments
    parser.add_argument(
        "--detector",
        type=str,
        default="retinaface",
        help="Type of face detector used by AVSRDataLoader."
    )
    parser.add_argument(
        "--landmarks-dir",
        type=str,
        default=None,
        help="Path to directory containing .pkl landmark files."
    )
    parser.add_argument(
        "--seg-duration",
        type=int,
        default=16,
        help="Max duration (seconds) for each segment."
    )
    parser.add_argument(
        "--combine-av",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="If True, merges audio and video into a single MP4 file."
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=1,
        help="Number of parallel groups for processing."
    )
    parser.add_argument(
        "--job-index",
        type=int,
        default=0,
        help="Index of the current job (for parallel processing)."
    )

    return parser.parse_args()


def gather_filenames(dataset, subset, data_dir):
    """
    Collect a list of .mp4 video file paths based on dataset (LRS2 or LRS3)
    and subset (train, val, or test).

    Parameters
    ----------
    dataset : str
        Name of the dataset, e.g., 'lrs2' or 'lrs3'.
    subset : str
        Which subset of the dataset to process, e.g., 'train', 'val', or 'test'.
    data_dir : str
        Path to the root directory of the dataset. For example:
        - LRS3 might have {data_dir}/trainval, {data_dir}/pretrain, {data_dir}/test
        - LRS2 might have {data_dir}/main, {data_dir}/pretrain

    Returns
    -------
    list of str
        A sorted list of full paths to .mp4 files that match the given dataset/subset.

    Raises
    ------
    NotImplementedError
        If the specified subset or dataset is not recognized/implemented.

    Examples
    --------
    # For LRS3, 'train' subset
    >>> gather_filenames('lrs3', 'train', '/path/to/LRS3')
    ['/path/to/LRS3/trainval/video1.mp4',
     '/path/to/LRS3/trainval/video2.mp4',
     '/path/to/LRS3/pretrain/videoX.mp4',
     ...]

    # For LRS2, 'val' subset
    >>> gather_filenames('lrs2', 'val', '/path/to/LRS2')
    ['/path/to/LRS2/main/VIDEO_01.mp4',
     '/path/to/LRS2/main/VIDEO_02.mp4',
     ...]
    """
    dataset_lower = dataset.lower()

    if dataset_lower == "lrs3":
        if subset == "test":
            # Collect all .mp4 files in data_dir/test/**/*
            return glob.glob(
                os.path.join(data_dir, subset, "**", "*.mp4"),
                recursive=True
            )
        elif subset == "train":
            # LRS3 "train" data is from "trainval" + "pretrain"
            trainval_files = glob.glob(
                os.path.join(data_dir, "trainval", "**", "*.mp4"),
                recursive=True
            )
            pretrain_files = glob.glob(
                os.path.join(data_dir, "pretrain", "**", "*.mp4"),
                recursive=True
            )
            all_files = trainval_files + pretrain_files
            all_files.sort()
            return all_files
        else:
            raise NotImplementedError(f"{subset} subset not implemented for LRS3.")

    elif dataset_lower == "lrs2":
        # LRS2 uses text files (train.txt, val.txt, test.txt) in the parent folder
        subset_txt = os.path.join(os.path.dirname(data_dir), f"{subset}.txt")

        if subset in ["val", "test"]:
            # E.g. lines in val.txt => main/<filename>
            with open(subset_txt, "r") as ftxt:
                lines = ftxt.read().splitlines()
            return [
                os.path.join(data_dir, "main", line.split()[0] + ".mp4")
                for line in lines
            ]

        elif subset == "train":
            # LRS2 train => 'main' + 'pretrain'
            train_txt = os.path.join(os.path.dirname(data_dir), "train.txt")
            pretrain_txt = os.path.join(os.path.dirname(data_dir), "pretrain.txt")

            with open(train_txt, "r") as ftxt:
                main_lines = ftxt.read().splitlines()
            with open(pretrain_txt, "r") as ftxt:
                pretrain_lines = ftxt.read().splitlines()

            main_files = [
                os.path.join(data_dir, "main", line.split()[0] + ".mp4")
                for line in main_lines
            ]
            pretrain_files = [
                os.path.join(data_dir, "pretrain", line.split()[0] + ".mp4")
                for line in pretrain_lines
            ]

            all_files = main_files + pretrain_files
            all_files.sort()
            return all_files

        else:
            raise NotImplementedError(f"{subset} subset not implemented for LRS2.")

    else:
        raise NotImplementedError(f"Dataset '{dataset}' not recognized.")


def split_workload(filenames, groups, job_index):
    """
    Split a list of filenames for distributed or parallel processing.

    Parameters
    ----------
    filenames : list of str
        List of video paths to be processed.
    groups : int
        Total number of parallel groups to split across.
    job_index : int
        The index (0-based) of the current job/group.

    Returns
    -------
    list of str
        The sub-list of filenames that this job_index should process.

    Examples
    --------
    >>> files = ["f1.mp4", "f2.mp4", "f3.mp4", "f4.mp4"]
    >>> split_workload(files, groups=2, job_index=0)
    ['f1.mp4', 'f2.mp4']
    >>> split_workload(files, groups=2, job_index=1)
    ['f3.mp4', 'f4.mp4']
    """
    if groups <= 1:
        return filenames

    unit = math.ceil(len(filenames) / groups)
    start_idx = job_index * unit
    end_idx = (job_index + 1) * unit
    return filenames[start_idx:end_idx]


def load_landmarks_if_available(data_filename, landmarks_dir, data_dir):
    """
    Load face landmarks (.pkl) if a landmarks directory is specified.

    Parameters
    ----------
    data_filename : str
        The path to the video file (e.g., /path/to/data_dir/xyz.mp4).
    landmarks_dir : str
        The directory containing the .pkl files with landmarks, or None.
    data_dir : str
        The original data directory prefix. This is replaced by landmarks_dir
        in order to form the .pkl path.

    Returns
    -------
    object or None
        The loaded landmark object (usually a dict) if found, otherwise None.

    Examples
    --------
    Suppose data_filename = "/data_dir/videos/clip123.mp4", 
    landmarks_dir = "/landmarks_dir/videos", 
    then we try to load "/landmarks_dir/videos/clip123.pkl"

    >>> load_landmarks_if_available("/data/clip123.mp4", "/landmarks", "/data")
    {'landmark_points': [...], ...}
    """
    if not landmarks_dir:
        return None
    # Replace the data_dir portion with landmarks_dir
    # Then change '.mp4' extension to '.pkl'
    landmarks_filename = data_filename.replace(data_dir, landmarks_dir)[:-4] + ".pkl"

    with open(landmarks_filename, "rb") as fin:
        return pickle.load(fin)


def process_segment(vid_data, aud_data, content,
                    out_vid, out_aud, out_txt,
                    combine_av, text_transform,
                    dataset_root, dataset_name, f,
                    video_fps=25, audio_rate=16000):
    """
    Save a (video_data, audio_data) segment, optionally combine them,
    and write metadata to the open label file handle.

    Parameters
    ----------
    vid_data : torch.Tensor or numpy.ndarray
        Video frames for the segment (shape [N, H, W] or [N, C, H, W]).
    aud_data : torch.Tensor or numpy.ndarray
        Audio samples for the segment (shape [channels, samples]).
    content : str
        The text transcript corresponding to this segment.
    out_vid : str
        Destination path for the output video (.mp4).
    out_aud : str
        Destination path for the output audio (.wav).
    out_txt : str
        Destination path for the text transcript (.txt).
    combine_av : bool
        If True, merges audio track into the MP4 file (removing the separate .wav).
    text_transform : TextTransform
        A transformation object used for tokenizing text.
    dataset_root : str
        The root directory for the entire preprocessed dataset (used for relative paths).
    dataset_name : str
        E.g., 'lrs2' or 'lrs3'. Used in label entries.
    f : file handle
        An open file handle for writing CSV-style metadata (labels).
    video_fps : int, optional
        Frames per second for the output video. Default is 25.
    audio_rate : int, optional
        Audio sample rate for the output .wav. Default is 16000.

    Returns
    -------
    None

    Examples
    --------
    >>> process_segment(
    ...     vid_data=some_video_tensor, 
    ...     aud_data=some_audio_tensor,
    ...     content="HELLO WORLD",
    ...     out_vid="/output/video.mp4",
    ...     out_aud="/output/audio.wav",
    ...     out_txt="/output/segment.txt",
    ...     combine_av=True,
    ...     text_transform=TextTransform(),
    ...     dataset_root="/output/lrs2",
    ...     dataset_name="lrs2",
    ...     f=file_handle
    ... )
    This will:
      - Save video.mp4, audio.wav, segment.txt
      - If combine_av=True, merge audio.wav into video.mp4
      - Write a line in the label file with tokenized transcript.
    """
    # Basic checks
    if vid_data is None or aud_data is None:
        return
    if vid_data.shape[0] == 0 or aud_data.shape[1] == 0:
        return

    # Save the raw data (video + audio + transcript)
    save_vid_aud_txt(
        out_vid,
        out_aud,
        out_txt,
        vid_data,
        aud_data,
        content,
        video_fps=video_fps,
        audio_sample_rate=audio_rate,
    )

    # Optionally combine audio track into the MP4
    if combine_av:
        in_vid = ffmpeg.input(out_vid)
        in_aud = ffmpeg.input(out_aud)

        merged_out = ffmpeg.output(
            in_vid["v"],
            in_aud["a"],
            out_vid[:-4] + ".av.mp4",
            vcodec="copy",
            acodec="aac",
            strict="experimental",
            loglevel="panic",
        )
        merged_out.run()
        # Replace the original MP4 with the merged one
        os.remove(out_aud)
        shutil.move(out_vid[:-4] + ".av.mp4", out_vid)

    # Write metadata to label file: dataset_name, relative_path, num_frames, token_ids
    basename = os.path.relpath(out_vid, start=dataset_root)
    token_ids = text_transform.tokenize(content)
    token_id_str = " ".join(str(t.item()) for t in token_ids)
    if token_id_str:
        num_frames = vid_data.shape[0]
        f.write(f"{dataset_name},{basename},{num_frames},{token_id_str}\n")


def main():
    """
    Main function orchestrating the preprocessing pipeline:
      1. Parse command-line arguments.
      2. Initialize data loaders.
      3. Gather and split .mp4 files to process.
      4. For each file:
         - Load landmarks (optional)
         - Load video & audio data
         - Depending on directory structure, either:
           (a) process entire video as one segment, or
           (b) split into multiple segments with `split_file(...)`
         - Save outputs (.mp4, .wav, .txt) and record metadata
      5. Close label file and finish.

    Examples
    --------
    >>> # Basic usage from command line
    >>> # python preprocess_avsr.py --data-dir /path/to/LRS2/main --root-dir /output --subset train --dataset lrs2 --gpu_type cuda
    """
    # ----------------------------------------------------
    # 1. Parse args & initial checks
    # ----------------------------------------------------
    args = parse_args()

    # Validate GPU type
    if args.gpu_type not in ["cuda", "mps"]:
        raise ValueError('gpu_type must be either "cuda" or "mps".')

    # Normalize data-dir path (remove trailing slash, etc.)
    args.data_dir = os.path.normpath(args.data_dir)

    # Initialize text transformation and data loaders
    text_transform = TextTransform()
    vid_loader = AVSRDataLoader(
        modality="video",
        detector=args.detector,
        convert_gray=False,
        gpu_type=args.gpu_type,
    )
    aud_loader = AVSRDataLoader(modality="audio")

    # Calculate maximum frames & samples for each segment
    seg_vid_len = args.seg_duration * 25      # 25 fps => # frames
    seg_aud_len = args.seg_duration * 16000   # 16 kHz => # audio samples

    # ----------------------------------------------------
    # 2. Prepare output directories & label file
    # ----------------------------------------------------
    if args.groups <= 1:
        csv_label = f"{args.dataset}_{args.subset}_transcript_lengths_seg{args.seg_duration}s.csv"
    else:
        csv_label = (
            f"{args.dataset}_{args.subset}_transcript_lengths_seg{args.seg_duration}s."
            f"{args.groups}.{args.job_index}.csv"
        )

    label_filename = os.path.join(args.root_dir, "labels", csv_label)
    os.makedirs(os.path.dirname(label_filename), exist_ok=True)
    print(f"Directory {os.path.dirname(label_filename)} created")

    f_label = open(label_filename, "w")

    # Output subdirectories for video and text
    dst_vid_dir = os.path.join(
        args.root_dir, args.dataset, f"{args.dataset}_video_seg{args.seg_duration}s"
    )
    dst_txt_dir = os.path.join(
        args.root_dir, args.dataset, f"{args.dataset}_text_seg{args.seg_duration}s"
    )

    # ----------------------------------------------------
    # 3. Gather and split the list of .mp4 files
    # ----------------------------------------------------
    filenames_all = gather_filenames(args.dataset, args.subset, args.data_dir)
    filenames_job = split_workload(filenames_all, args.groups, args.job_index)

    # ----------------------------------------------------
    # 4. Main loop: process each .mp4 file
    # ----------------------------------------------------
    for mp4_file in tqdm(filenames_job):
        # 4.1 Load facial landmarks if provided
        landmarks = load_landmarks_if_available(mp4_file, args.landmarks_dir, args.data_dir)

        # 4.2 Load video & audio data from the file
        try:
            video_data = vid_loader.load_data(mp4_file, landmarks)
            audio_data = aud_loader.load_data(mp4_file)
        except (UnboundLocalError, TypeError, OverflowError, AssertionError):
            # If there's a loading error, skip this file
            continue

        # 4.3 Decide whether to split or use entire video
        top_level = os.path.normpath(mp4_file).split(os.sep)[-3]
        if top_level in ["trainval", "test", "main"]:
            # Treat the entire file as a single segment
            base_out_vid = mp4_file.replace(args.data_dir, dst_vid_dir)[:-4]
            base_out_txt = mp4_file.replace(args.data_dir, dst_txt_dir)[:-4]

            dst_vid = base_out_vid + ".mp4"
            dst_aud = base_out_vid + ".wav"
            dst_txt = base_out_txt + ".txt"

            # Corresponding transcript file
            txt_path = mp4_file[:-4] + ".txt"
            if not os.path.exists(txt_path):
                continue
            text_line_list = open(txt_path).read().splitlines()[0].split(" ")
            # LRS transcripts often have 2 prefix tokens (start/end times), so skip them
            text_line = " ".join(text_line_list[2:])
            content = text_line.replace("}", "").replace("{", "")

            # Process as one segment
            process_segment(
                video_data,
                audio_data,
                content,
                dst_vid,
                dst_aud,
                dst_txt,
                args.combine_av,
                text_transform,
                os.path.join(args.root_dir, args.dataset),
                args.dataset,
                f_label,
            )
            continue

        # 4.4 Otherwise, split the file according to the transcript
        txt_file_path = mp4_file[:-4] + ".txt"
        if not os.path.exists(txt_file_path):
            continue
        splitted_segments = split_file(txt_file_path, max_frames=seg_vid_len)

        for i, (content, start, end, duration) in enumerate(splitted_segments):
            if len(splitted_segments) == 1:
                # Only one segment => entire video
                v_data, a_data = video_data, audio_data
            else:
                # Slice out [start:end] frames in video and [start:end] in audio
                start_idx, end_idx = int(start * 25), int(end * 25)
                try:
                    v_data = video_data[start_idx:end_idx]
                    a_data = audio_data[:, start_idx * 640 : end_idx * 640]
                except (TypeError, IndexError):
                    continue

            # Build output filenames
            base_out_vid = mp4_file.replace(args.data_dir, dst_vid_dir)[:-4]
            base_out_txt = mp4_file.replace(args.data_dir, dst_txt_dir)[:-4]
            suffix = f"_{i:02d}"

            dst_vid = base_out_vid + suffix + ".mp4"
            dst_aud = base_out_vid + suffix + ".wav"
            dst_txt = base_out_txt + suffix + ".txt"

            # Save & merge if needed
            process_segment(
                v_data,
                a_data,
                content,
                dst_vid,
                dst_aud,
                dst_txt,
                args.combine_av,
                text_transform,
                os.path.join(args.root_dir, args.dataset),
                args.dataset,
                f_label,
            )

    # ----------------------------------------------------
    # 5. Close label file and conclude
    # ----------------------------------------------------
    f_label.close()
    print("Preprocessing completed.")


if __name__ == "__main__":
    main()