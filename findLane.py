import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(img):

    gray = cv2.cvtColor(laneImage, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
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


def averageSlopeIntercept(ing, lines):
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


img = cv2.imread("image\/test_image.jpg")
laneImage = np.copy(img)
canny = canny(laneImage)
cropped_img = region_of_interest(canny)
lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100,
                        np.array([]), minLineLength=40, maxLineGap=5)
averageLine = averageSlopeIntercept(laneImage, lines)
lineImg = displayLines(laneImage, lines)
combo_img = cv2.addWeighted(laneImage, 0.8, lineImg, 1, 1)
cv2.imshow('result', combo_img)
cv2.waitKey(0)

# plt.imshow()
# plt.show()
