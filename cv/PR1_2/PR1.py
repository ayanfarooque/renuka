import cv2

# Read the image
image = cv2.imread("lion.jpg")
cv2.imshow("Original Image", image)
cv2.waitKey(2000)

# Convert to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", image_gray)
cv2.waitKey(2000)

# Thresholding to get binary image
ret, thresh = cv2.threshold(image_gray, 150, 255, cv2.THRESH_BINARY)
cv2.imshow("Binary Image", thresh)
cv2.waitKey(2000)

# Find contours
contours, hierarchy = cv2.findContours(
    thresh,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_NONE
)

# Draw contours on a copy of the original image
image_copy = image.copy()
cv2.drawContours(image_copy, contours, -1, (0, 0, 255), 2, cv2.LINE_AA)
cv2.imshow("Contour Image", image_copy)
cv2.waitKey(0)

cv2.destroyAllWindows()
