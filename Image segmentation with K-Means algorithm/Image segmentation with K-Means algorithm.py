# Load the required libraries   
import numpy 
import cv2     






# Import the image (By default, it will be moved into BGR color space) 
picture = cv2.imread("C:\\Users\\Halil.Durmaz\\Desktop\\Picture.jpg")






# In order to have a value for each pixel, reduce the dimension of the array. Therefore, it will be in a suitable form to apply K-means
# 1080 x 1920 x 3 ----> 2073600 x 3
# 3 represents the number of the axis of the 3 dimensional color space. For example, BGR has 3 axis
# Reduce the dimension to 2, by -1 
picture_reshape = picture.reshape((-1,3)) 






# Convert the values from uint8 type to float type 
# Because this is a requirement for the K-means function within the OpenCV
picture_reshape_float = numpy.float32(picture_reshape)














###########################################################
# Image segmentation by using K-Means clustering algorithm      
###########################################################


                            # Number of clusters
K=3
                        


                            # Criteria to terminate the algorithm
# Terminates the algorithm when any of the conditions below is TRUE 
# EPS: Terminate when it reaches the specified epsilon value
# MAX ITER: Terminate when it reaches the specified iteration value
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
            10,  # MAX ITER
            1.0) # EPS





                        # Number of Random Initialization 
attempts = 100



                        # Method of centroid assignment in the beginning
flags = cv2.KMEANS_RANDOM_CENTERS



                       
                        # Performing K-Means algorithm
# Function will give 3 outputs   
# ret ---> wss
# center ---> centroids                       
ret,label,center = cv2.kmeans(picture_reshape_float,
                              K,
                              None,
                              criteria,
                              attempts,
                              flags)                        












################################################################
# To see the output image of the work, we need additional steps
################################################################


                            # Converting it into the image form
# Convert the Center values into uint8 type
center = numpy.uint8(center)

# Convert the Label values into uint8 type + move into the color space
label_flatten = center[label.flatten()]

# Do reshape based on the shape of the original image      
label_flatten_reshaped = label_flatten.reshape((picture.shape)) 






                            # Save the output image as an image file
cv2.imwrite("C:\\Users\\Halil.Durmaz\\Desktop\\Segmented.jpg", 
            label_flatten_reshaped)
