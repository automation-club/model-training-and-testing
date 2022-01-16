
from __future__ import annotations
from cProfile import label
from distutils.util import subst_vars
from pydoc import resolve
from sqlite3 import DatabaseError
from VideoDataset import VideoDataset
from VideoFrameDataset import VideoFrameDataset

import torch
import labelbox as lb
from pathlib import Path
import requests
import torchvision
import cv2
import numpy as np

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
    

if __name__ == "__main__":
    # Grabs video and annotation data from Labelbox
    labelbox_project = access_labelbox_project(api_key=config.LABELBOX_API_KEY, project_id=config.LABELBOX_PROJECT_ID)
    annotations = fetch_annotations(labelbox_project, config.VIDEO_PATH)
    

    # download_video_data(labelbox_project=labelbox_project, save_path=(DATASET_PATH/VIDEO_DATA_FILE_NAME).resolve())

    test = VideoDataset(
        video_path=config.VIDEO_PATH,
        annotations=annotations,
        mode="include_empty_annotation_frames",
    )

    # one = test.__getitem__()  
    # two = test.__getitem__()

    for i in range(5):
        data = test.__getitem__()
        frames = data[0].numpy()
        annotations = data[1]
        print(annotations[0][1].item())
        for idx, frame in enumerate(frames):
            print((annotations[idx][1].item(), annotations[idx][2].item()))
            image_with_annotation = cv2.circle(
                frame,
                (int(annotations[idx][1].item()), int(annotations[idx][2].item())), 
                radius=10,
                color=(255, 0, 0),
                thickness=-1
            )

            cv2.imshow("pinball", cv2.cvtColor(image_with_annotation, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)

       
    x = [1, 2 , 3, 4]
    print(x[:-1])