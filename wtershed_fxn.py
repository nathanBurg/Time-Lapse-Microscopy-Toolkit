import cv2
import numpy as np
import random2
from skimage.segmentation import watershed



def wtershed(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    labels = watershed(gray)

    zeros = np.zeros(gray.shape, dtype="uint8")
    # Iterate through unique labels
    count = 0
    for label in np.unique(labels):
        if label == 0:
            continue

        # Create a mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        # Find contours and determine contour area
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        c = max(cnts, key=cv2.contourArea)
        color = (100 + random2.randint(50, 200), 100 + random2.randint(50, 200), 100 + random2.randint(50, 200))
        cv2.drawContours(image, [c], -1, (36,255,12), 1)
        count += 1


        if count != 1:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            cv2.circle(zeros, (cX, cY), 1, color, -1)

    '''
    print(count)
    plt.imshow(zeros)
    plt.show()
    '''

    return zeros
