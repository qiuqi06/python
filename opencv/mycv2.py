import cv2 as cv


def access_pixels(image):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    for row in range(height):
        for col in range(width):
            for c in range(channels):
                pv = image[row, col, c]
                image[row, col, c] = 255 - pv
    cv.imshow("pixel demo", image)


src = cv.imread("image/000.jpg")
cv.namedWindow("jpg", cv.WINDOW_AUTOSIZE)
cv.imshow("init", src)
access_pixels(src)
cv.waitKey(0)
cv.destroyAllWindows()
print("hello")
