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



# translating the matrix 
import cv2
import numpy as np

img = cv2.imread('sendtodali.png')
row,col = img.shape[:2]
translation_matrix = np.float32([[1,0,70], [0,1,110]])
img_translation = cv2.warpAffine(img, translation_matrix, (col + 70,row + 110))
translation_matrix = np.float32([[1,0,-30], [0,1,-50]])
img_center = cv2.warpAffine(img_translation, translation_matrix, (col + 70 + 30,row + 110 + 50))

# Image rotation
rotation_matrix = cv2.getRotationMatrix2D((col/2,row/2), 30, 0.5 )

img_rotation = cv2.warpAffine(img_center, rotation_matrix, ((col + 70 + 30),(row + 110 + 50)))

cv2.imshow('Rotated Image' , img_rotation)
cv2.waitKey()
