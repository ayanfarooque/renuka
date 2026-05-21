import cv2
# import numpy as np
import matplotlib.pyplot as plt

# load a color image
img = cv2.imread("swan.png")
 
if img is None:
    print("Error: could not read an Image")
else:
    # convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # split the channels
    h, s, v = cv2.split(hsv)
    # cleate a CLAHE object argument
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
    v_equalize = clahe.apply(v)
    # merge the clannels
    hsv_equalize = cv2.merge([h,s,v_equalize])
    # convert back to BGR
    img_clahe = cv2.cvtColor(hsv_equalize, cv2.COLOR_HSV2BGR)
    # display result 
    img_color_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_clahe_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 8))
    plt.subplot(1,2,1)
    plt.imshow(img_color_rgb)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(img_clahe_rgb)
    plt.title('CLAHE Equalize')
    plt.axis('off')
    plt.show()

