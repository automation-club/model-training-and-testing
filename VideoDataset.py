from __future__ import annotations
from concurrent.futures import process
from pathlib import Path
from traceback import print_tb
from torch.utils.data import Dataset
from pytorchvideo.data.encoded_video import EncodedVideo

import numpy as np
import torchvision
import cv2

# Iterable Dataset
class VideoDataset(Dataset):
	def __init__(self, video_path, annotations_array) -> None:
		super().__init__()

		self.video_path = video_path,
		self.total_frame_count, self.video_width, self.video_height, self.video_fps = self.get_video_info(video_path)
		self.video = EncodedVideo.from_path(video_path)
		self.annotations_array = annotations_array

	def __len__(self):
		return self.total_frame_count
		
	def __getitem__(self, idx):
		data = self.video.get_clip(idx, idx+1)
		return data['video'], self.annotations_array[idx*60:idx*60+60]

	def get_video_info(self, video_path):
		cap = cv2.VideoCapture(video_path)
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
		video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
		fps = cap.get(cv2.CAP_PROP_FPS)
		return total_frames, video_width, video_height, fps
	

# Map-style Dataset
# class VideoDataset(Dataset):
 
#     def __init__(self, video_path, annotations, mode, start_frame=0, frames_per_batch=128,) -> None:
#         super().__init__()

#         self.video_path = video_path
#         self.annotations = annotations
#         self.mode = mode
#         self.start_frame = start_frame
#         self.frames_per_batch = frames_per_batch

#     def __len__(self):
#         pass

    
#     def __getitem__(self):

#         if self.mode == "skip_empty_annotation_frames":
#             # iterate through [frames_per_batch] frames and discover all the frames you need.
#             # Input all needed frames to make Tensor
#             pass
        
#         if self.mode == "include_empty_annotation_frames":

#             frames = torchvision.io.read_video(
#                 filename=self.video_path,
#                 start_pts=self.start_frame/60.0,
#                 end_pts=(self.start_frame+self.frames_per_batch)/60.0,
#                 pts_unit="sec",
#             )

#             annotations_tensor = torch.from_numpy(self.annotations[self.start_frame:self.start_frame+self.frames_per_batch])        

#             self.start_frame += self.frames_per_batch

#             return (frames[0][:-1], annotations_tensor)
