import cv2
import numpy as np
import matplotlib.pyplot as plt

# Loading an image in grayscale
img = cv2.imread('swan.png', 0)

if img is None:
    print(f"Error: could not load an image from {img}")
else:
    # apply custom equilization
    equ = cv2.equalizeHist(img)
    # stack the origianl and equalized image for comparison
    res = np.hstack((img, equ)) # single side by side image

    # displaying result
    plt.figure(figsize=(15,8))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(equ, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')

    plt.show()