import cv2
from matplotlib import pyplot as plt

image = cv2.imread('noisysalterpepper.png')
# image = cv2.imread('gaussian_noise.png')
# image = cv2.imread('uniform_noise.png')

# Apply mean filter (kernel size 3x3)
mean_blur = cv2.blur(image, (3,3)) # it is suitable for uniform noise

# Apply Gaussian smoothing
gaussian_blur = cv2.GaussianBlur(image, (3,3), 0) # it is suitable for gaussian noise

# Apply median filter
median_blur = cv2.medianBlur(image, 5) # it is suitable for salt and pepper noise (black and white dots)

# Displaying results
plt.figure(figsize=(10, 8))
plt.subplot(2,2,1)
plt.imshow(image)
plt.title("Original")
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(mean_blur)
plt.title("Mean Filter")
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(gaussian_blur)
plt.title("Gaussian Filter")
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(median_blur)
plt.title("Median Filter")
plt.axis('off')
plt.tight_layout()
plt.show()
