import sys
import cv2
from pytorchvideo.data.encoded_video import EncodedVideo

x = [1,2,3,4,5,6,7,8,9,10]
print(x[0:5])
encoded_vid = EncodedVideo.from_path('./datasets/test-vid.mp4')
data = encoded_vid.get_clip(0,1)
print(data['video'].shape)
print(sys.getsizeof(data['video']))
print('done')
# cap = cv2.VideoCapture('./datasets/test-vid.mp4')

# success, frame = cap.read()
# count = 1


# while success:
#     cv2.imwrite(f'./data/test-vid/{count:05d}.jpg', frame)
    
#     success, frame = cap.read()
#     count += 1

#     if count % 100 == 0:
#         print(f'{count} frames processed')

