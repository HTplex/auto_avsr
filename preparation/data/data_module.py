#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torchaudio
import torchvision
import librosa
from torchvision.io import VideoReader
import numpy as np
import av
import cv2


def read_video_pyav_cpu(filename, num_threads=8):
    """
    Multi-threaded CPU decoding of an entire video into a list of NumPy frames.
    """
    container = av.open(filename, options={"threads": str(num_threads)})
    video_stream = container.streams.video[0]

    # Alternatively, set thread type on the codec context:
    # video_stream.thread_type = "FRAME"  # or "SLICE"
    # video_stream.thread_count = num_threads

    frames = []
    for frame in container.decode(video=0):
        # Convert to RGB NumPy array
        img = frame.to_rgb().to_ndarray()
        frames.append(img)

    return np.stack(frames, axis=0)


def read_video_cv(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # frame is a NumPy array in BGR format
        frames.append(frame)
        print(len(frames))


    cap.release()
    frames = np.array(frames)  # Convert to NumPy array if needed
    return frames

def fast_read_video_to_numpy(video_path):
    """
    Reads a video file into a NumPy array of shape [T, H, W, C].
    Uses TorchVision's VideoReader for faster, more efficient decoding.
    """
    reader = VideoReader(video_path, "video")  # Only read video frames
    frame_list = []

    for frame in reader:
        # frame["data"] is a torch.Tensor of shape [C, H, W]
        # Permute to [H, W, C], then convert to NumPy
        frame_np = frame["data"].permute(1, 2, 0).cpu().numpy()
        frame_list.append(frame_np)
        # print(len(frame_list))

    # Stack into a single NumPy array of shape [T, H, W, C]
    return np.stack(frame_list, axis=0)


class AVSRDataLoader:
    def __init__(self, modality, detector="retinaface", convert_gray=True, gpu_type="cuda"):
        self.modality = modality
        if modality == "video":
            if detector == "retinaface":
                from detectors.retinaface.detector import LandmarksDetector
                from detectors.retinaface.video_process import VideoProcess

                self.landmarks_detector = LandmarksDetector(device=gpu_type+":0")
                self.video_process = VideoProcess(convert_gray=convert_gray)

            if detector == "mediapipe":
                from detectors.mediapipe.detector import LandmarksDetector
                from detectors.mediapipe.video_process import VideoProcess

                self.landmarks_detector = LandmarksDetector()
                self.video_process = VideoProcess(convert_gray=convert_gray)

    def load_data(self, data_filename, landmarks=None, transform=True):
        if self.modality == "audio":
            audio, sample_rate = self.load_audio(data_filename)
            audio = self.audio_process(audio, sample_rate)
            return audio
        if self.modality == "video":
            video = self.load_video(data_filename)
            if not landmarks:
                landmarks = self.landmarks_detector(video)
            video = self.video_process(video, landmarks)
            if video is None:
                raise TypeError("video cannot be None")
            video = torch.tensor(video)
            return video

    def load_audio(self, data_filename):
        data, sample_rate = librosa.load(data_filename, sr=16000, mono=True)
        waveform = torch.from_numpy(data).unsqueeze(0)
        # old one
        # waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
        return waveform, sample_rate

    def load_video(self, data_filename):
        # return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()
        return fast_read_video_to_numpy(data_filename)
        # return read_video_cv(data_filename)

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform
