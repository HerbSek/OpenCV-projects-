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
x1Units = eval(input("Enter x1 units to be moved :") )
y1Units = eval(input("Enter y1 units to be moved :") )
# [eg. 70,110,-30,-50]
img = cv2.imread('./sendtodali.png')  # Read Image 
num_rows, num_cols = img.shape[:2]   # Determine row and column size

translation_matrix = np.float32([[1,0,xUnits],[0,1,yUnits]])  # Using numpy to create a translation Matrix

img_translation = cv2.warpAffine(img, translation_matrix, ( num_cols + xUnits , num_rows + yUnits ))  # Apply the translation and get a full window

translation_matrix = np.float32([[1,0,(x1Units)],[0,1,(y1Units)]])

img_translation = cv2.warpAffine(img_translation, translation_matrix, (num_cols + xUnits - x1Units , num_rows + yUnits - y1Units ))

cv2.imshow('Translate', img_translation)

cv2.waitKey()