from __future__ import annotations
from pydoc import resolve
from sqlite3 import DatabaseError
from VideoFrameDataset import VideoFrameDataset

import torch
import labelbox as lb
from pathlib import Path
import requests
import torchvision


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


def fetch_annotations(labelbox_project):

    return labelbox_project.annotations
    


# def generate_dataset_from_labelbox(api_key, project_id):

#     # LabelBox credentials
#     lb_client = lb.Client(api_key=api_key)
#     lb_project = lb_client.get_project(project_id=project_id)

#     # Export frames & X,Y point coordinates from project
#     labels = lb_project.video_labelbox_project()
#     labels = next(labels)
#     frames = labels.data.value
#     xycoords = labels.annotations

#     # Convert frames to PyTorch Tensor
#     frames_buffer = pd.DataFrame() 
#     # frames_tensor = torch.Tensor()
#     print(np.ndarray(frame for idx, frame in frames))
#     test = torch.from_numpy(np.fromiter((frame for idx, frame in frames), float))
#     print(test)
#     # for idx, frame in frames:
#     #     print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
#     #     print(idx)
#     #     frames_buffer.append(frame)
#     #     # if idx % 50:
#         #     torch.cat(frames_tensor, torch.Tensor(frames_buffer))
#         #     frames_buffer = []
            
    
#     test = np.array(frames_list)
#     print(test.shape)
    
if __name__ == "__main__":

        
    # Grabs video and annotation data from Labelbox
    labelbox_project = access_labelbox_project(api_key=LABELBOX_API_KEY, project_id=LABELBOX_PROJECT_ID)
    annotations = fetch_annotations(labelbox_project)
    print(annotations)
    # download_video_data(labelbox_project=labelbox_project, save_path=(DATASET_PATH/VIDEO_DATA_FILE_NAME).resolve())

    # video_dataset = VideoFrameDataset(
    #     root_path=(DATASET_PATH).resolve(),
    #     annotationfile_path=(DATASET_PATH/ANNOTATIONS_FILE_NAME).resolve(),
    #     num_segments=1,
    #     frames_per_segment=1,
    # )


    # TODO: Split video into directory with frames to annotate and create dataset