from numpy import dtype, load
from requests.exceptions import ProxyError
from torch.functional import Tensor
import labelbox as lb

from models import *
from utils.utils import *
# from pytorch_objectdetecttrack.models import *
# from pytorch_objectdetecttrack.utils.utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import psutil
import pandas as pd
import urllib


def download_video_data(api_key, project_id):
    lb_client = lb.Client(api_key=api_key)
    lb_project = lb_client.get_project(project_id=project_id)
    
    labels = lb_project.video_label_generator()
    labels = next(labels)
    
    video_data_url = labels.data['url']
    urllib.urlretrieve(video_data_url, '/datasets/test-video.mp4')

def detect_img(img):
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        # transforms.Pad((
        #     max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0),max(int((imw-imh)/2),0)), (128,128,128)),
        transforms.ToTensor(),            
    ])

    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze(0)

    input_img = Variable(image_tensor.type(Tensor))

    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, 80, conf_thres, nms_thres)
        
    return(detections)

def generate_dataset_from_labelbox(api_key, project_id):

    # LabelBox credentials
    lb_client = lb.Client(api_key=api_key)
    lb_project = lb_client.get_project(project_id=project_id)

    # Export frames & X,Y point coordinates from project
    labels = lb_project.video_label_generator()
    labels = next(labels)
    frames = labels.data.value
    xycoords = labels.annotations

    # Convert frames to PyTorch Tensor
    frames_buffer = pd.DataFrame() 
    # frames_tensor = torch.Tensor()
    print(np.ndarray(frame for idx, frame in frames))
    test = torch.from_numpy(np.fromiter((frame for idx, frame in frames), float))
    print(test)
    # for idx, frame in frames:
    #     print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    #     print(idx)
    #     frames_buffer.append(frame)
    #     # if idx % 50:
        #     torch.cat(frames_tensor, torch.Tensor(frames_buffer))
        #     frames_buffer = []
            
    
    test = np.array(frames_list)
    print(test.shape)
    
if __name__ == "__main__":

    # Configuring Dataset
    LABELBOX_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJja3k0MG9pNjkyZjYwMHplcGdlM3o2anpyIiwib3JnYW5pemF0aW9uSWQiOiJja3k0MG9ocXEyZjV6MHplcGVsYjk1N3YyIiwiYXBpS2V5SWQiOiJja3lieXAxdnUwOThnMHpieDNycmdjMThiIiwic2VjcmV0IjoiY2M0YTUzYTA2MmYxMDM4NmY1MWJiNTExM2Q1YTgxYTQiLCJpYXQiOjE2NDIwMTcyODUsImV4cCI6MjI3MzE2OTI4NX0.kXsSSgzrAeFdYdryYgzdok6eiyHydLA88ZP_Pd7EnuQ"
    LABELBOX_PROJECT_ID = "cky4nw7aaohqu0zdh6d75gobs"
    
    download_video_data(api_key=LABELBOX_API_KEY, project_id=LABELBOX_PROJECT_ID)
    
    # frames_tensor, coords_tensor = \
    #     generate_dataset_from_labelbox(api_key=LABELBOX_API_KEY, project_id=LABELBOX_PROJECT_ID)
    
    # config_path = "config/yolov3.cfg"
    # weights_path = "config/yolov3.weights"
    # class_path = "config/coco.names"

    # img_size = 416
    # conf_thres = 0.3
    # nms_thres = None

    # # Load Models and Weights
    # model = Darknet(config_path=config_path, img_size=img_size)
    # model.load_weights(weights_path=weights_path)
    # model.eval()    
    # classes = load_classes(class_path)

    # tensor = torch.FloatTensor

    # img_path = "Screenshot from 2022-01-11 23-31-51.png"
    # img = Image.open(img_path).convert('RGB')

    # sus = detect_img(img)
