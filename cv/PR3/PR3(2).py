import cv2
import numpy as np
from sklearn.cluster import MeanShift,estimate_bandwidth

# loading the image
image = cv2.imread("image4.jpg")
# converting BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#converting image  into 2D array (pixels * color channels)
pixels = image.reshape(-1,3)

#estimating automatic bandwidth
bandwidth = estimate_bandwidth(pixels,quantile=0.1,n_samples=1000)

#meanshift clustering
ms = MeanShift(bandwidth=bandwidth,bin_seeding=True)
ms.fit(pixels)

#get labels and cluster centers
labels = ms.labels_
cluster_centers = ms.cluster_centers_

#replacing each pixels into cluster centers
segmented_pixels = cluster_centers[labels]

#reshape to original
segmented_image = segmented_pixels.reshape(image.shape).astype(np.uint8)
# convert back to BGR to display
seg_image_bgr = cv2.cvtColor(segmented_image,cv2.COLOR_RGB2BGR)

cv2.imshow("Original Image",cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
cv2.imshow("Segmented Image using Scikit-learn",seg_image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()