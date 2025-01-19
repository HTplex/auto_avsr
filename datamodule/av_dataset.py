#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import torch
import torchaudio
import torchvision


def load_video(path):
    """
    rtype: torch, T x C x H x W
    """
    vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    vid = vid.permute((0, 3, 1, 2))
    return vid


def load_audio(path):
    """
    rtype: torch, T x 1
    """
    waveform, sample_rate = torchaudio.load(path[:-4] + ".wav", normalize=True)
    return waveform.transpose(1, 0)


class AVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        label_path,
        subset,
        modality,
        audio_transform,
        video_transform,
        rate_ratio=640,
    ):

        self.root_dir = root_dir

        self.modality = modality
        self.rate_ratio = rate_ratio

        self.list = self.load_list(label_path)
        self.input_lengths = [int(_[2]) for _ in self.list]

        self.audio_transform = audio_transform
        self.video_transform = video_transform

    def load_list(self, label_path):
        paths_counts_labels = []
        short_clip_count = 0
        normal_clip_count = 0
        for path_count_label in open(label_path).read().splitlines():
            dataset_name, rel_path, input_length, token_id_str = path_count_label.split(",")
            input_length = int(input_length)
            if input_length < 12:
                # Skip short clips
                short_clip_count += 1
                continue
            normal_clip_count += 1
            token_id = torch.tensor([int(_) for _ in token_id_str.split()])
            paths_counts_labels.append((dataset_name, rel_path, input_length, token_id))
        # print("short clips: ", short_clip_count, "normal clips: ", normal_clip_count)
        return paths_counts_labels

    def __getitem__(self, idx):
        dataset_name, rel_path, input_length, token_id = self.list[idx]
        path = os.path.join(self.root_dir, dataset_name, rel_path)
        if self.modality == "video":
            video = load_video(path)
            # print(path)
            # print("video shape: ", video.shape)
            video = self.video_transform(video)
            # print("video shape after transform: ", video.shape)
            return {"input": video, "target": token_id}
        elif self.modality == "audio":
            audio = load_audio(path)
            audio = self.audio_transform(audio)
            return {"input": audio, "target": token_id}

    def __len__(self):
        return len(self.list)
