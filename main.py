from numpy import load
from torch.functional import Tensor

from models import *
from utils.utils import *
# from pytorch_objectdetecttrack.models import *
# from pytorch_objectdetecttrack.utils.utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def detect_img(img):
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([
        transforms.Resize((imh,imw)),
        transforms.Pad((
            max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0),max(int((imw-imh)/2),0)), (128,128,128,1)),
        transforms.ToTensor(),            
    ])

    image_tensor = img_transforms(img).float()
    return(image_tensor)


if __name__ == "__main__":

    config_path = "config/yolov3.cfg"
    weights_path = "config/yolov3.weights"
    class_path = "config/coco.names"

    img_size = 418
    confidence_thres = 0.8
    nms_thres = None

    # Load Models and Weights
    model = Darknet(config_path=config_path, img_size=img_size)
    model.load_weights(weights_path=weights_path)
    model.eval()    
    classes = load_classes(class_path)

    tensor = torch.FloatTensor

    img_path = "Screenshot from 2022-01-11 23-31-51.png"
    img = Image.open(img_path)

    sus = detect_img(img)

    #img.show()
    print(sus.shape)    
    plt.imshow(sus.permute(1,2,0))
    plt.show()