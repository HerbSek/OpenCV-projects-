import cv2
import numpy as np

img = cv2.imread('herbert.jpg')
height,width = img.shape[:2]

contrast_enhancement = eval(input('Enter degree of contractness : '))

kernel_matrix = [
    [-1,-1,-1,-1,-1],
    [-1, 2, 2, 2,-1],
    [-1, 2, contrast_enhancement, 2,-1],
    [-1, 2, 2, 2,-1],
    [-1,-1,-1,-1,-1]

       ]

normalization_matrix = np.array(kernel_matrix) / contrast_enhancement

kernel_image =  cv2.filter2D(img, -1, normalization_matrix)

img_scale = cv2.resize(img, (int(0.5*(width-1)),int(0.5*(height-1))), interpolation=cv2.INTER_AREA)

kernel_image_scale = cv2.resize(kernel_image, (int(0.5*(width-1)),int(0.5*(height-1))), interpolation=cv2.INTER_AREA)

cv2.imshow('Input', img_scale)
cv2.imshow('Output', kernel_image_scale)


cv2.waitKey()

