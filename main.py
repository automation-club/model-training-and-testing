from json import load
from matplotlib import pyplot as plt
import psutil
from cProfile import label
from distutils.util import subst_vars
from inspect import isclass
from pydoc import resolve
from sqlite3 import DatabaseError

from scipy.fft import dst
from typeguard import sys
from VideoDataset import VideoDataset
from VideoFrameDataset import VideoFrameDataset
from torch.utils.data import DataLoader
from itertools import islice

import labelbox as lb
from pathlib import Path
import numpy as np
import requests
import cv2
import torch
import config

def access_labelbox_project(api_key, project_id):
    
    # Get Access to Labelbox Project
    lb_client = lb.Client(api_key=api_key)
    lb_project = lb_client.get_project(project_id=project_id)
    
    # Retrieve Video Label generator
    labelbox_project = lb_project.video_label_generator()
    labelbox_project = next(labelbox_project)
     
    return labelbox_project
    

def download_video_data(labelbox_project, save_path):

    # Download Video Data
    video_data_url = labelbox_project.data.url
    request = requests.get(video_data_url, stream=True)
    with open(save_path, "wb") as video_file:
        for chunk in request.iter_content(chunk_size=10240):
            if chunk:
                video_file.write(chunk)


def fetch_annotations(labelbox_project, video_file):
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Annotations array format - (frames, (present, x, y))
    annotations_array = np.zeros(shape=(frame_count, 3)) 
    annotations = labelbox_project.annotations
    for annotation in annotations:
        annotations_array[annotation.frame] = np.array([1, annotation.value.x, annotation.value.y])
    # print(type(labelbox_project.annotations[0]))
    return annotations_array
    
def dim(a):
    if not type(a) == list:
        return []
    return [len(a)] + dim(a[0])

if __name__ == "__main__":
    # Grabs video and annotation data from Labelbox
    labelbox_project = access_labelbox_project(api_key=config.LABELBOX_API_KEY, project_id=config.LABELBOX_PROJECT_ID)
    # download_video_data(labelbox_project=labelbox_project, save_path=config.VIDEO_PATH)
    annotations = fetch_annotations(labelbox_project, config.VIDEO_PATH)

    video_dataset = VideoDataset(
        video_path=config.VIDEO_PATH,
        annotations_array=annotations,
    )

    loader = DataLoader(
        dataset=video_dataset,
        batch_size=1,
        num_workers=1
        #TODO: Test multiple workers (local issue?)
    )

    for i, (frames, annotations) in enumerate(loader):
        print(psutil.virtual_memory().percent)
        frames = frames.type(torch.ByteTensor)
        frames = frames[0].permute(1,2,3,0)
        print(annotations.shape)
        for frame, annotation in zip(frames, annotations[0]):
            frame = np.ascontiguousarray(frame.numpy())
            frame = cv2.circle(
                img=frame,
                center=(int(annotation[1].item()), int(annotation[2].item())),
                radius=5,
                color=(255,0,0),
                thickness=-1,
            )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("pinball", frame)   
            cv2.waitKey(1)

      