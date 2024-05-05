# import cv2
# img = cv2.imread('./sendtodali.png', cv2.IMREAD_GRAYSCALE)
# cv2.imshow('Input image', img)
# cv2.imwrite('./grayscale.png',img)
# cv2.waitKey()

# import cv2
# myimg = './sendtodali.png'
# img = cv2.imread(myimg)
# yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
# cv2.imshow('Y channel', yuv_img[:, :, 0])
# cv2.imshow('U channel', yuv_img[:, :, 1])
# cv2.imshow('V channel', yuv_img[:, :, 2])
# cv2.waitKey()





# Image Translation

import cv2
import numpy as np 

xUnits = eval(input("Enter x units to be moved :") )
yUnits = eval(input("Enter y units to be moved :") )
img = cv2.imread('./sendtodali.png')  # Read Image 
num_rows, num_cols = img.shape[:2]   # Determine row and column size
translation_matrix = np.float32([[1,0,xUnits],[0,1,yUnits]])  # Using numpy to create a translation Matrix
img_translation = cv2.warpAffine(img, translation_matrix, ( num_cols + xUnits , num_rows + yUnits ))  # Apply the translation and get a full window
translation_matrix = np.float32([[1,0,-xUnits],[0,1,-yUnits]])
img_translation = cv2.warpAffine(img, translation_matrix, (num_cols + (xUnits * 2), num_rows + (yUnits * 2)))
cv2.imshow('Translate', img_translation)
cv2.waitKey()





