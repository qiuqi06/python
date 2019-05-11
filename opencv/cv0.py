import cv2 as cv
import numpy as np

img = cv.imread("image/000.jpg")

cv.line(img, (0, 0), (10, 10), color=(0, 255, 255), thickness=3)

triangle = np.array([[10, 30], [40, 80], [10, 90]], np.int32)
cv.fillConvexPoly(img, triangle, color=(255, 0, 0))

cv.polylines(img, [triangle], isClosed=True, color=(255, 255, 0), thickness=2)

cv.namedWindow("test")
cv.imshow("test", img)
cv.waitKey(0)