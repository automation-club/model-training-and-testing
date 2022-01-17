import config
from imutils.video import VideoStream, FileVideoStream
import cv2
from itertools import cycle, islice
from torch.utils.data import IterableDataset, DataLoader

# for i in range(14200):
#     frame = vid.read()
#     cv2.imshow("test", frame)

#     cv2.waitKey(1)

class Sus(IterableDataset):
    
    def __init__(self) -> None:
        super(Sus, self).__init__()
        self.vid = FileVideoStream(config.VIDEO_PATH).start()
        self.cap = cv2.VideoCapture(config.VIDEO_PATH)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    def __iter__(self):
        return self.get_stream(self.vid)

    def get_stream(self,vid_stream):
        return cycle(self.process(vid_stream))

    def process(self, vid):
        for i in range(self.frame_count):
            yield vid.read()

data = Sus()
loader = DataLoader(data, batch_size=100)

i = 1


for batch in loader:
    frames = batch
    for frame in frames:
        cv2.imshow("test", cv2.cvtColor(frame.numpy(), cv2.COLOR_BGR2RGB))   
        cv2.waitKey(1)
