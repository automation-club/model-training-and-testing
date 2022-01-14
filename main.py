from sqlite3 import DatabaseError
from VideoFrameDataset import VideoFrameDataset

import torch
import labelbox as lb
import os, requests
import torchvision


def access_labelbox_project(api_key, project_id):
    
    # Get Access to Labelbox Project
    lb_client = lb.Client(api_key=api_key)
    lb_project = lb_client.get_project(project_id=project_id)
    
    # Retrieve Video Label generator
    labelbox_project = lb_project.video_label_generator()
    labelbox_project = next(labelbox_project)

    return labelbox_project
    

def download_video_data(labelbox_project, save_path, file_name):

    # Download Video Data
    video_data_url = labelbox_project.data.url
    request = requests.get(video_data_url, stream=True)
    save_path = os.path.join(save_path, file_name+".mp4")
    with open(save_path, "wb") as video_file:
        for chunk in request.iter_content(chunk_size=10240):
            if chunk:
                video_file.write(chunk)


def create_annotations_file(annotations, save_path, file_name):
    print(sus)
    


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

    # CONFIGURABLE
    LABELBOX_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJja3k0MG9pNjkyZjYwMHplcGdlM3o2anpyIiwib3JnYW5pemF0aW9uSWQiOiJja3k0MG9ocXEyZjV6MHplcGVsYjk1N3YyIiwiYXBpS2V5SWQiOiJja3lieXAxdnUwOThnMHpieDNycmdjMThiIiwic2VjcmV0IjoiY2M0YTUzYTA2MmYxMDM4NmY1MWJiNTExM2Q1YTgxYTQiLCJpYXQiOjE2NDIwMTcyODUsImV4cCI6MjI3MzE2OTI4NX0.kXsSSgzrAeFdYdryYgzdok6eiyHydLA88ZP_Pd7EnuQ"
    LABELBOX_PROJECT_ID = "cky4nw7aaohqu0zdh6d75gobs"
    DATASET_PATH = "./datasets"
    VIDEO_DATA_FILE_NAME = "test-vid"
    ANNOTATIONS_FILE_NAME = "test-annotations"
    
    labelbox_project = access_labelbox_project(api_key=LABELBOX_API_KEY, project_id=LABELBOX_PROJECT_ID)

    # download_video_data(labelbox_project=labelbox_project, save_path=DATASET_PATH, file_name=VIDEO_DATA_FILE_NAME)

    print(os.path.join(DATASET_PATH, VIDEO_DATA_FILE_NAME+".mp4"))
    video_dataset = VideoFrameDataset(
        root_path=os.path.join(DATASET_PATH, VIDEO_DATA_FILE_NAME+".mp4"),
        annotationfile_path=os.path.join(DATASET_PATH, ANNOTATIONS_FILE_NAME+".txt"),
        num_segments=3,
        frames_per_segment=1,
    )
    # TODO: Split video into directory with frames to annotate and create dataset