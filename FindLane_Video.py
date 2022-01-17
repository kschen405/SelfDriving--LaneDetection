from cmath import nan
import cv2
import numpy as np
import matplotlib.pyplot as plt


def Canny(img):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 灰階
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # 去雜訊
    canny = cv2.Canny(blur, 90, 110)
    #canny = cv2.resize(canny, (1000, 1000))
    return canny


def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def displayLines(img, lines):
    lineImg = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(lineImg, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lineImg


def makeCoordinates(img, line_para):
    try:
        slope, intercept = line_para
        y1 = img.shape[0]
        y2 = int(y1*3/5)
        x1 = int((y1-intercept)/slope)
        x2 = int((y2-intercept)/slope)
        return np.array([x1, y1, x2, y2])
    except:
        return None


def averageSlopeIntercept(img, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = makeCoordinates(img, left_fit_average)
    right_line = makeCoordinates(img, right_fit_average)
    return np.array([left_line, right_line])

# For image
# img = cv2.imread("image\/test_image.jpg")
# laneImage = np.copy(img)
# canny = canny(laneImage)
# cropped_img = region_of_interest(canny)
# lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100,
#                         np.array([]), minLineLength=40, maxLineGap=5)
# averageLine = averageSlopeIntercept(laneImage, lines)
# lineImg = displayLines(laneImage, averageLine)
# combo_img = cv2.addWeighted(laneImage, 0.8, lineImg, 1, 1)
# cv2.imshow('result', lineImg)
# cv2.waitKey(0)


cap = cv2.VideoCapture("image\/test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    if frame is None:
        break
    frame = frame.astype('uint8')
    canny = Canny(frame)
    cropped_img = region_of_interest(canny)
    lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100,
                            np.array([]), minLineLength=40, maxLineGap=5)
    averageLine = averageSlopeIntercept(frame, lines)
    lineImg = displayLines(frame, averageLine)
    combo_img = cv2.addWeighted(frame, 0.8, lineImg, 1, 1)
    cv2.imshow('result', combo_img)
    cv2.waitKey(1)
