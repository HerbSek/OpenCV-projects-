# import cv2
# img = cv2.imread('./sendtodali.png', cv2.IMREAD_GRAYSCALE)
# cv2.imshow('Input image', img)
# cv2.imwrite('./grayscale.png',img)
# cv2.waitKey()

import cv2
myimg = './sendtodali.png'
img = cv2.imread(myimg)
yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
cv2.imshow('Y channel', yuv_img[:, :, 0])
cv2.imshow('U channel', yuv_img[:, :, 1])
cv2.imshow('V channel', yuv_img[:, :, 2])
cv2.waitKey()
