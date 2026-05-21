# Mean shift segmentation in Python
import cv2 
import numpy as np

# loading the image
image = cv2.imread("image4.jpg")
# converting image from BGR to RGB 
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
# applying mean shift filtering
seg_image = cv2.pyrMeanShiftFiltering(image,sp =20, sr =30)
# convert back to BGR for display
seg_image_bgr = cv2.cvtColor(seg_image , cv2.COLOR_RGB2BGR)

# showing segmented and original image
cv2.imshow("Original Image" , cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
cv2.imshow("Mean shift segmentation using OpenCV only" , seg_image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()