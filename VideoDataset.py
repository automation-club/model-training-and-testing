from pathlib import Path


import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision

class VideoDataset(Dataset):

    def __init__(self, video_path, annotations, mode, start_frame=0, frames_per_batch=128,) -> None:
        super().__init__()

        self.video_path = video_path
        self.annotations = annotations
        self.mode = mode
        self.start_frame = start_frame
        self.frames_per_batch = frames_per_batch

    def __len__(self):
        pass

    
    def __getitem__(self):

        if self.mode == "skip_empty_annotation_frames":
            # iterate through [frames_per_batch] frames and discover all the frames you need.
            # Input all needed frames to make Tensor
            pass
        
        if self.mode == "include_empty_annotation_frames":

            frames = torchvision.io.read_video(
                filename=self.video_path,
                start_pts=self.start_frame/60.0,
                end_pts=(self.start_frame+self.frames_per_batch)/60.0,
                pts_unit="sec",
            )

            annotations_tensor = torch.from_numpy(self.annotations)

            self.start_frame += self.frames_per_batch

            return (frames[0][:-1], annotations_tensor)

    def restructure_annotations(self, annotations):
        pass