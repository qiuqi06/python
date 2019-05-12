import cv2 as cv
import numpy as np


def create_image_3channel():
	'''
			img=np.zeros([400,400,3],np.uint8)
	img[:,:,0]=np.ones([400,400])*255
	img[:,:,2]=np.ones([400,400])*255
	'''
	img = np.zeros([400, 400,3], np.uint8)   #init
	img[:,:,0]=np.ones([400,400])*255
	# img=img*1
	cv.imshow("new image",img)
	cv.imwrite("image/create_01.jpg",img)


def create_image_1channel():
	img = np.zeros([400, 400,1], np.uint8)
	img[:, :, 0] = np.ones([400, 400]) * 127
	cv.imshow("new image", img)
	cv.imwrite("image/create_01.jpg", img)

create_image_1channel()
cv.waitKey(0)
