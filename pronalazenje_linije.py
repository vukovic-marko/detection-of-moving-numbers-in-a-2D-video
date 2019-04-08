import cv2
import numpy as np

def pronadji_liniju(boja, frame) :
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([boja, 100, 100])
    upper_blue = np.array([boja+100, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    res = cv2.bitwise_and(frame,frame, mask= mask)

    dilated = cv2.dilate(res, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))
    eroded = cv2.erode(dilated, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))

    image = cv2.Canny(eroded, 120, 120)

    lines = cv2.HoughLinesP(image ,rho = 1,theta = 1*np.pi/180, threshold = 100,minLineLength = 150,maxLineGap = 20)

    x1t = 0; x2t = 0; y1t = 0; y2t = 0


    for line in lines:
        x1,y1,x2,y2 = line[0]
        x1t += x1; x2t += x2; y1t += y1; y2t += y2

    x1t //= lines.shape[0]; x2t //= lines.shape[0]; y1t //= lines.shape[0]; y2t //= lines.shape[0]

    return (x1t, y1t), (x2t, y2t)