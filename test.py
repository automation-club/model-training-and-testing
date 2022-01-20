from concurrent.futures import process
from multiprocessing import get_start_method

from typeguard import gc
import config
from imutils.video import VideoStream, FileVideoStream
import cv2
from itertools import cycle, islice
from torch.utils.data import IterableDataset, DataLoader
import torch, psutil

class TestClass(IterableDataset):
    def __init__(self) -> None:
        super().__init__()

    def process_data(self):
        for idx in range(540):
            yield torch.rand((1920,1080, 3))

    def get_stream(self):
        return cycle(self.process_data())

    def __iter__(self):
        # worker_total_num = torch.utils.data.get_worker_info().num_workers
        # worker_id = torch.utils.data.get_worker_info().id
        return self.process_data()


              

testclass = TestClass()

loader = DataLoader(testclass, batch_size=100, num_workers=0)

for batch in islice(loader, 10):
    print(psutil.virtual_memory().percent)
    print(batch.shape)

    # for i in range(14200):
    #     frame = vid.read()
    #     cv2.imshow("test", frame)

    #     cv2.waitKey(1)

# class Sus(IterableDataset):
    
#     def __init__(self) -> None:
#         super(Sus, self).__init__()
#         self.vid = FileVideoStream(config.VIDEO_PATH).start()
#         self.cap = cv2.VideoCapture(config.VIDEO_PATH)
#         self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     def __iter__(self):
#         return self.get_stream(self.vid)

#     def get_stream(self,vid_stream):
#         return cycle(self.process(vid_stream))

#     def process(self, vid):
#         for i in range(self.frame_count):
#             yield vid.read()

# data = Sus()
# loader = DataLoader(data, batch_size=100)

# i = 1


# for batch in loader:
#     frames = batch
#     for frame in frames:
#         cv2.imshow("test", cv2.cvtColor(frame.numpy(), cv2.COLOR_BGR2RGB))   
#         cv2.waitKey(1)
