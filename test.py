import sys
import time
import cv2
import torch
import numpy as np


def order_points_old(pts):
    pts = np.array(pts)
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def crop_frame(frame, from_coords):
    from_coords = np.array(order_points_old(from_coords), np.float32)

    to_coords = np.array([
        [0, 0],
        [frame.shape[1], 0],
        [frame.shape[1], frame.shape[0]],
        [0, frame.shape[0]],
    ], np.float32)

    trans = cv2.getPerspectiveTransform(from_coords, to_coords)
    return cv2.warpPerspective(frame, trans, (frame.shape[1], frame.shape[0]))


def get_2_with_largest_area(coords):
    largest = 0
    coord1 = None
    coord2 = None
    for pair1 in coords:
        for pair2 in coords:
            area = abs((pair2[1] - pair1[1]) * (pair2[0] - pair1[0]))
            if area > largest:
                largest = area
                coord1 = pair1
                coord2 = pair2

    return coord1, coord2


def detect_corner_rectangles(frame):
    # Thresholding to extract colors of the tape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Split frame into two parts to analyze
    sections = np.hsplit(hsv, 2)
    bottom, top = sections[0], sections[1]

    y_low = np.array([20, 130, 130])
    y_high = np.array([30, 255, 255])
    y_mask = cv2.inRange(top, y_low, y_high)

    b_low = np.array([100, 140, 20])
    b_high = np.array([130, 255, 255])
    b_mask = cv2.inRange(bottom, b_low, b_high)

    mask = np.hstack((b_mask, y_mask))
    # mask = cv2.erode(mask, None, iterations=2)
    # mask = cv2.dilate(mask, None, iterations=2)
    extracted = cv2.bitwise_and(frame, frame, mask=mask)

    extracted = cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY)
    conts, _ = cv2.findContours(extracted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coords_ = []
    for c in conts:
        area = cv2.contourArea(c)
        perim = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * perim, True)

        if (30 < area < 250) and len(approx) == 4:
            m = cv2.moments(c)
            x = int(m["m10"] / m["m00"])
            y = int(m["m01"] / m["m00"])
            # cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            coords_.append([x, y])
    return coords_


def ball_detection(frame, prev_frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    ball_low = np.array([0, 0, 0])
    ball_high = np.array([255, 100, 255])
    ball_mask = cv2.inRange(hsv, ball_low, ball_high)
    ball_mask = cv2.erode(ball_mask, None, iterations=2)
    ball_mask = cv2.dilate(ball_mask, None, iterations=2)

    extracted = cv2.bitwise_and(frame, frame, mask=ball_mask)
    extracted = cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY)
    conts = cv2.findContours(extracted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in conts[0]:
        # cv2.circle(frame, (int(x),int(y)), int(r), (255,0,0), 3)
        # area = cv2.contourArea(c)
        # perim = cv2.arcLength(c, True)
        # approx = cv2.approxPolyDP(c, 0.09*perim, True)
        cv2.drawContours(frame, [c], 0, (255, 0, 0), 4)
        # if (50 < area and area < 150) and     

    # cv2.imshow('test', frame)
    # cv2.waitKey(0)


# Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# # Images
# imgs = ['/home/owhan/Downloads/homeless.jpg']  # batch of images
if __name__ == "__main__":

    cap = cv2.VideoCapture('./datasets/green.mp4')

    coords = np.array([[0, 0], [0, 1920], [1080, 0], [1080, 1920]])
    runs = 0
    new_time = 0
    prev_time = time.time()
    ok, prev_frame = cap.read()
    while True:
        ok, frame = cap.read()
        if runs % 20 == 0:
            new_coords = detect_corner_rectangles(frame)
            if len(new_coords) == 4:
                coords = new_coords

        for coord in coords:
            cv2.circle(frame, (coord[0], coord[1]), 8, (0, 255, 0), -1)
        runs += 1

        warped = crop_frame(frame, coords)
        warped = cv2.GaussianBlur(warped, (21, 21), 0)
        ball_detection(warped, prev_frame)

        prev_frame = frame

        new_time = time.time()
        fps = str(int(1.0 / (new_time - prev_time)))
        prev_time = new_time
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(warped, "fps: " + fps, (50, 50), font, 2, (255, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow("detections", frame)
        cv2.imshow("warped", warped)
        cv2.waitKey(1)
