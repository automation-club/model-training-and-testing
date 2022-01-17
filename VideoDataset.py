from __future__ import annotations
from pathlib import Path
from itertools import cycle, islice
from torch.utils.data import IterableDataset

from imutils.video import FileVideoStream
import numpy as np
import torchvision
import cv2

# Iterable Dataset
class VideoDataset(IterableDataset):
	def __init__(self, video_path, annotations_array) -> None:
		super().__init__()

		self.video_path = video_path,
		self.total_frame_count, self.video_width, self.video_height, self.video_fps = self.get_video_info(video_path)
		self.video_stream = FileVideoStream(path=video_path, queue_size=256).start()
		self.annotations_array = annotations_array

	def __iter__(self):
		return self.get_stream(self.video_stream, self.annotations_array)
	
	def get_stream(self, video_stream, annotations_array):
		return cycle(self.process_data(video_stream, annotations_array))

	def process_data(self, video_stream, annotations_array):
		for idx in range(self.total_frame_count):
			yield (video_stream.read(), annotations_array[idx])
		
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
