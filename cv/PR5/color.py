import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_yellow(image_path):
    # load the image
    img = cv2.imread(image_path)

    # convert to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
   
    # create a mask to isolate yellow color
    mask = cv2.inRange(hsv_img, yellow_lower, yellow_upper)
    # mask = cv2.inRange(hsv_img, green_lower, green_upper)

    # find contours of yellow color
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    output_img = img.copy()

    # loop through each contour and draw bounding box
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 500:
            cv2.rectangle(output_img, (x,y), (x+w, y+h), (0,0,255), 3)
    
    # convert image to RGB
    output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    #display
    plt.figure(figsize=(12,6))

    # original image
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    # processed image
    plt.subplot(1,2,2)
    plt.imshow(output_img_rgb)
    plt.title("Yellow color detection on Image")
    plt.axis('off')
    plt.show()

    return output_img

image_path = 'fruits.png'
output_image = detect_yellow(image_path)
