# chapter 1 (Applying geemetric transformations on images)

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

# import cv2
# import numpy as np 

# xUnits = eval(input("Enter x units to be moved :") )
# yUnits = eval(input("Enter y units to be moved :") )
# x1Units = eval(input("Enter x1 units to be moved :") )
# y1Units = eval(input("Enter y1 units to be moved :") )
# # [eg. 70,110,-30,-50]
# img = cv2.imread('./sendtodali.png')  # Read Image 
# num_rows, num_cols = img.shape[:2]   # Determine row and column size

# translation_matrix = np.float32([[1,0,xUnits],[0,1,yUnits]])  # Using numpy to create a translation Matrix

# img_translation = cv2.warpAffine(img, translation_matrix, ( num_cols + xUnits , num_rows + yUnits ))  # Apply the translation and get a full window

# translation_matrix = np.float32([[1,0,(x1Units)],[0,1,(y1Units)]])

# img_translation = cv2.warpAffine(img_translation, translation_matrix, (num_cols + xUnits - x1Units , num_rows + yUnits - y1Units ))

# cv2.imshow('Translate', img_translation)

# cv2.waitKey()


# Image rotation 

# import cv2

# img = cv2.imread('sendtodali.png')

# row, col = img.shape[:2]

# rotation_matrix = cv2.getRotationMatrix2D((col/2,row/2), 80, 1)

# img_rotation = cv2.warpAffine(img, rotation_matrix, (col,row))

# cv2.imshow('Rotation 90Deg' , img_rotation)
# cv2.waitKey()



# import cv2
# import numpy as np

# img = cv2.imread('sendtodali.png')
# num_rows, num_cols = img.shape[:2]
# translation_matrix = np.float32([ [1,0,int(0.5*num_cols)],[0,1,int(0.5*num_rows)] ])
# rotation_matrix = cv2.getRotationMatrix2D((num_cols, num_rows), 30, 1)
# img_translation = cv2.warpAffine(img, translation_matrix, (2*num_cols, 2*num_rows))
# img_rotation = cv2.warpAffine(img_translation, rotation_matrix,(2*num_cols, 2*num_rows))


# cv2.imshow('Rotation', img_rotation)
# cv2.waitKey()




# import cv2
# import numpy as np

# img = cv2.imread('sendtodali.png')
# row,col = img.shape[:2]
# translation_matrix = np.float32([[1,0,70], [0,1,110]])
# img_translation = cv2.warpAffine(img, translation_matrix, (col + 70,row + 110))
# translation_matrix = np.float32([[1,0,-30], [0,1,-50]])
# img_center = cv2.warpAffine(img_translation, translation_matrix, (col + 70 ,row + 110 ))

# # Image rotation
# rotation_matrix = cv2.getRotationMatrix2D((col/2,row/2), 30, 0.5 )

# img_rotation = cv2.warpAffine(img_center, rotation_matrix, ((col + 70 ),(row + 110 )))

# cv2.imshow('Rotated Image' , img_rotation)
# cv2.waitKey()




# translating the matrix 

# import cv2
# import numpy as np 


# img = cv2.imread('sendtodali.png')
# row, col = img.shape[:2]
# translation_matrix = np.float32([[1,0,(0.5*col)],[0,1,-(0.5*row)]]) 
# img_translation = cv2.warpAffine(img, translation_matrix, (col,row))
# cv2.imshow('translation', img_translation)
# cv2.waitKey()

# Image interpolation example : skewing method
# import cv2

# img = cv2.imread('sendtodali.png')

# img_skewed = cv2.resize(img, (600,450), interpolation = cv2.INTER_AREA)

# cv2.imshow('Skewed Image',img_skewed)

# cv2.waitKey()



# Affine Transformation

# import cv2
# import numpy as np

# daliImg = cv2.imread('sendtodali.png')
# height,width = daliImg.shape[:2]
# first_point = np.float32([[0,0],[width-1,0],[0,height-1]])
# affine_point = np.float32([ [0,0],[0.5*(width-1),0],[(0.5*(width-1)),height-1] ])
# matrix_affine = cv2.getAffineTransform(first_point, affine_point)
# affine_img = cv2.warpAffine(daliImg, matrix_affine, (width,height))
# cv2.imshow('Output',affine_img)
# cv2.waitKey()



# import cv2
# import numpy as np

# img = cv2.imread('sendtodali.png')
# height,width = img.shape[:2]

# input_point = np.float32([ [0,0], [width-1,0] , [0,height-1] ])
# output_point = np.float32([ [width-1,0] , [0,0], [width-1,height-1] ])

# affine_matrix = cv2.getAffineTransform(input_point, output_point)

# affine_image = cv2.warpAffine(img, affine_matrix, (width,height))

# cv2.imshow('parallelogram', affine_image)

# cv2.waitKey()



# Projection transformation : 

# import cv2 
# import numpy as np

# img = cv2.imread('sendtodali.png')
# row,col = img.shape[:2]

# input_point = np.float32([[0,0],[col-1,0],[0,row-1],[col-1,row-1]])
# output_point = np.float32([[0,0.3*(row-1)],[col-1,0],[0,0.7*(row-1)],[(col-1),row-1]])

# projection_matrix = cv2.getPerspectiveTransform(input_point,output_point)

# projection_image = cv2.warpPerspective(img, projection_matrix, (col,row))
# cv2.imshow('Prpjection', projection_image)
# cv2.waitKey()



# Exercise Projection 
# import cv2
# import numpy as np
# img = cv2.imread('sendtodali.png')
# cv2.imshow('Input', img)

# height, width = img.shape[:2]
# control_point = np.float32([[0,0],[width-1,0],[0,height-1],[width-1,height-1]])
# output_contol_point = np.float32([[0,0],[width-1,0.3*(height-1)],[0,height-1],[width-1,0.7*(height-1)]])
# projection_matrix = cv2.getPerspectiveTransform(control_point,output_contol_point)
# projection_img = cv2.warpPerspective(img, projection_matrix, (width,height))
# cv2.imshow('Output', projection_img)
# cv2.imwrite('./SendTele.png', projection_img)
# cv2.waitKey()





# import cv2
# import numpy as np

# img = cv2.imread('sendtodali.png')
# height,width = img.shape[0:2]

# first_point = np.float32([[0,0],[width-1,0],[0,height-1],[width-1,height-1]])
# second_point = np.float32([[0,0],[width-1,0.3*(height-1)],[0,height-1],[(width-1),0.7*(height-1)]])

# matrix_homopraphy = cv2.getPerspectiveTransform(first_point,second_point)

# homography_img = cv2.warpPerspective(img, matrix_homopraphy, (width,height))
# homography_scale = cv2.resize(homography_img, (550,450), interpolation = cv2.INTER_AREA)

# cv2.imshow('homograph', homography_scale)
# cv2.waitKey()





# Image Warping 

# import cv2
# import numpy as np
# import math

# img = cv2.imread('sendtodali.png')
# rows, cols = img.shape[:2]
# #####################
# # Vertical wave

# img_output = np.zeros(img.shape, dtype=img.dtype)


# for i in range(rows):
#     for j in range(cols):
#         offset_x = int(25.0 * math.sin(2 * 3.14 * i / 180))
#         offset_y = 0
#         if j+offset_x < rows:
#             img_output[i,j] = img[i,(j+offset_x)%cols]
#         else:
#             img_output[i,j] = 0


# scale_image = cv2.resize(img_output, (550,450), interpolation = cv2.INTER_AREA)



# cv2.imshow('Vertical wave', scale_image)
# cv2.waitKey()




# Chapter 2 (Edge detection and Image filtering)

# Kernel Matrix and Low pass filtering


# import cv2
# import numpy as np

# img = cv2.imread('sendtodali.png')
# # height,width = img.shape[0:2]

# # blur_matrix = np.ones((5,5), np.float32) / 25   # Normalizing image 
# # blur_image = cv2.filter2D(img, -1, blur_matrix)


# blur_image = cv2.blur(img, (4,4))
# cv2.imshow('Output', blur_image)
# cv2.waitKey()



# import cv2
# import numpy as np

# img = cv2.imread('sendtodali.png')
# # height,width = img.shape[:2]

# blur_matrix = np.ones((4,4),np.float32) / 16
# blur_image = cv2.filter2D(img,-1, blur_matrix)
# cv2.imshow('Blur4x4', blur_image)
# cv2.waitKey()


# Edge Detection 

# Sobel filter (Horizontal edge detection)
# import cv2


# img = cv2.imread('shapes.png')
# height,width = img.shape[:2] 
# horizontal_edge_Detection = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  
# img_scale = cv2.resize(horizontal_edge_Detection, (int(0.5*(width-1)),int(0.5*(height-1))), interpolation= cv2.INTER_AREA )
# cv2.imshow('Detect Horizontal Edge', img_scale)
# cv2.waitKey()





# Laplacian filter (detects both edges)

# import cv2
# img = cv2.imread('bus.png')
# # laplacian = cv2.Laplacian(img, cv2.CV_64F)
# # cv2.imshow('Laplacian Edge detector', laplacian)

# canny filter method (Better than the other two)

# canny = cv2.Canny(img,10,150)
# cv2.imshow('Canny Edge Detector',  canny )
# cv2.waitKey()


# Blurring and edge detection practice 

# import cv2
# import numpy as np

# img = cv2.imread('bus.png')
# # normalize_matrix = np.ones((5,5), np.float32) / 25
# # blur_image = cv2.filter2D(img, -1, normalize_matrix)
# # cv2.imshow('Blur image', blur_image)

# # blur_image2 = cv2.blur(img, (4,4))
# # cv2.imshow('Blur image2', blur_image2)

# edge1 = cv2.Sobel(img, cv2.CV_64F, 1,0, ksize=1)
# cv2.imshow('Horizontal Edge Detection', edge1)




# cv2.waitKey()


# import cv2

# img = cv2.imread('shapes.png')
# # laplacian_image = cv2.Laplacian(img, cv2.CV_64F)
# # cv2.imshow('Laplacian' , laplacian_image)
# # cv2.waitKey()

# canny_image = cv2.Canny(img, 30,350)
# cv2.imshow('canny Image', canny_image)
# cv2.waitKey()




# Motion Blur 

# import cv2
# import numpy as np


# img = cv2.imread('herbert.jpg')
# # Generating the kernel
# size = 15
# kernel_matrix_blur = np.zeros((size,size), dtype = img.dtype)
# kernel_matrix_blur[int(0.5*(size-1)), 0: ] = np.ones(size)
# kernel_matrix_blur = kernel_matrix_blur / size # Normalize matrix 

# output=cv2.filter2D(img, -1 , kernel_matrix_blur)
# cv2.imshow('Motion blur', output)
# cv2.waitKey()


# motion blur 

import cv2
import numpy as np

img = cv2.imread('herbert.jpg')
size = 13 
height,width = img.shape[:2]

# generating kernel

# motion_blur_array = np.zeros((size,size))
# motion_blur_array[int(0.5*(size-1)), 0: ] = np.ones(size)
# motion_blur_array = motion_blur_array / size # Normalizing array

# img_scale = cv2.resize(img, (int(0.5*(width-1)),int(0.5*(height-1))), interpolation=cv2.INTER_AREA)

# apply_blur = cv2.filter2D(img_scale, -1, motion_blur_array)

# cv2.imshow('Output', apply_blur)
# cv2.waitKey()


# read on , John Von Newmann inheritance and Othomatter theory /// Computer Architecture 


#Image Enhancing 

import cv2
import numpy as np

img = cv2.imread('herbert.jpg')
height, width = img.shape[:2]

contrast_array = np.array([[-2,-2,-2], [-2,19,-2], [-2,-2,-2]])

contrast_image = cv2.filter2D(img, -1, contrast_array)

contrast_image_scale = cv2.resize(contrast_image, (int(0.5*(width-1)),int(0.5*(height-1))), interpolation = cv2.INTER_AREA)


cv2.imshow('contrast', contrast_image_scale)
cv2.waitKey()



