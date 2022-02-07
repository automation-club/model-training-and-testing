from locale import CRNCYSTR
import sys
import cv2
from pydantic import DecimalIsNotFiniteError
import torch
import numpy as np

def crop_frame(frame, from_coords):
    from_ = np.array([
        [from_coords[0][0], from_coords[0][1]],
        [from_coords[1][0], from_coords[0][1]],
        [from_coords[0][0], from_coords[1][1]],
        [from_coords[1][0], from_coords[1][1]],
    ])
    to_coords = np.array([
        [0,0],
        [frame.shape[0],0],
        [0, frame.shape[1]],
        [frame.shape[0], frame.shape[1]],
    ])

    trans = cv2.getPerspectiveTransform(from_, to_coords)
    return cv2.warpPerspective(frame, trans, (frame.shape[0], frame.shape[1]))

def get_2_with_largest_area(coords):
    largest = 0
    coord1 = None
    coord2 = None
    for pair1 in coords:
        for pair2 in coords:
            area = abs((pair2[1]-pair1[1])*(pair2[0]-pair1[0]))
            if area > largest:
                largest = area
                coord1 = pair1
                coord2 = pair2

    return coord1, coord2

def detect_corner_rectangles(frame):
    # Thresholding to extract colors of the tape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    y_low = np.array([20, 130, 130])
    y_high = np.array([30,255,255])
    y_mask = cv2.inRange(hsv, y_low, y_high)

    b_low = np.array([100, 50, 20])
    b_high = np.array([130, 255, 255])
    b_mask = cv2.inRange(hsv, b_low, b_high)

    mask = cv2.bitwise_or(y_mask, b_mask)
    extracted = cv2.bitwise_and(frame, frame, mask=mask)

    # Finding tape contours
    extracted = cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY)
    cnts,_ = cv2.findContours(extracted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coords = []
    for c in cnts:
        area = cv2.contourArea(c)
        perim = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.11*perim, True)

        if (area >120 and area <250) and len(approx)==4:
            m = cv2.moments(c)
            x = int(m["m10"]/m["m00"])
            y = int(m["m01"]/m["m00"])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            coords.append([x,y])
    return frame, coords   

 

# Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# # Images
# imgs = ['/home/owhan/Downloads/homeless.jpg']  # batch of images
cap = cv2.VideoCapture('./datasets/test-vid.mp4')

while True:
    ok, frame = cap.read()

    frame, coords = detect_corner_rectangles(frame)
    coord1, coord2 = get_2_with_largest_area(coords) 
    print(coord1)
    frame = crop_frame(frame, (coord1, coord2))
    cv2.imshow("test", frame)
    cv2.waitKey(1)

cv2.imshow("test", frame)
cv2.waitKey(0)