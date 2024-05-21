# Image Translation

import cv2
import numpy as np 

xUnits = eval(input("Enter x units to be moved :") )
yUnits = eval(input("Enter y units to be moved :") )
x1Units = eval(input("Enter x1 units to be moved :") )
y1Units = eval(input("Enter y1 units to be moved :") )
# [eg. 70,110,-30,-50]
img = cv2.imread('Image path here')  # Read Image 
num_rows, num_cols = img.shape[:2]   # Determine row and column size

translation_matrix = np.float32([[1,0,xUnits],[0,1,yUnits]])  # Using numpy to create a translation Matrix

img_translation = cv2.warpAffine(img, translation_matrix, ( num_cols + xUnits , num_rows + yUnits ))  # Apply the translation and get a full window

translation_matrix = np.float32([[1,0,(x1Units)],[0,1,(y1Units)]])

img_translation = cv2.warpAffine(img_translation, translation_matrix, (num_cols + xUnits - x1Units , num_rows + yUnits - y1Units ))

img_scale = cv2.resize(img, (660,460), interpolation = cv2.INTER_AREA)

img_translation_scale = cv2.resize(img_translation, (660,460), interpolation = cv2.INTER_AREA)

cv2.imshow('input', img_scale)
cv2.imshow('Translate', img_translation_scale)

cv2.waitKey()