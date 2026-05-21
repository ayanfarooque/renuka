import cv2
import numpy as np

image = cv2.imread('objects.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur (important)
blur = cv2.GaussianBlur(gray, (11,11), 0)

# Adaptive threshold (handles lighting better)
thresh = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11, 2
)

# Morphological closing (fills gaps inside coins)
kernel = np.ones((5,5), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

# Remove small noise
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=2)

# Find contours
contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

count = 0

for cnt in contours:
    area = cv2.contourArea(cnt)

    # Tune this threshold based on your image
    if area > 1000:
        count += 1
        cv2.drawContours(image, [cnt], -1, (0,255,0), 3)

print("Number of Objects:", count)

cv2.imshow("Threshold", opening)
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()