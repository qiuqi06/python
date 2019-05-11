import cv2 as cv


def get_image_info(image):
	print(type(image))
	print(image.shape)  # shape0~3  height width  channels
	print(image.size)
	print(image.dtype)


src = cv.imread("opencv/image/000.jpg")
cv.namedWindow("jpg", cv.WINDOW_AUTOSIZE)
cv.imshow("init", src)
get_image_info(src)
# cv.waitKey(0)
cv.destroyAllWindows()
print("hello")
