import cv2
import matplotlib.pyplot as plt

image = cv2.imread("RUL-lung-nodule_Pina.png")
# cv2.imshow("Original Image", image)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_blur = cv2.GaussianBlur(image_gray, (5,5), 0)

# sobel Edge Detection
sobelx = cv2.Sobel(src=image_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3) # sobel edge detection on x axis
sobely = cv2.Sobel(src=image_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3) # sobel edge detection on y axis
sobelxy = cv2.Sobel(src=image_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # on both the axis

# canny edge detection
# edge = cv2.Canny(image_gray, 50, 150)

# display sobel edge detection
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.convertScaleAbs(sobelxy)

sobelx_rgb = cv2.cvtColor(sobelx, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(15, 8))
plt.subplot(1,4,1)
plt.imshow(image_gray)
plt.title('Original')
plt.axis('off')

plt.subplot(1,4,2)
plt.imshow(sobelx,  cmap='gray')
plt.title('SOBEL X')
plt.axis('off')

plt.subplot(1,4,3)
plt.imshow(sobely,  cmap='gray')
plt.title('SOBEL Y')
plt.axis('off')

plt.subplot(1,4,4)
plt.imshow(sobelxy,  cmap='gray')
plt.title('SOBEL X and Y')
plt.axis('off')
plt.show() 

# cv2.imshow('Original', image)
# cv2.imshow('SOBEL X', sobelx)
# cv2.imshow('SOBEL Y', sobely)
# cv2.imshow('SOBEL XY', sobelxy)
# cv2.imshow('Canny', edge)
# cv2.waitKey(0)