import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def pronadji_konture(frame) :
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, img_bin = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    img_bin = cv2.dilate(img_bin, cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)))

    _, contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = frame.copy()

    detected_contours = []

    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)

        if (h >= 15 and h <= 25) or (w >= 15 and h >= 10): 
            detected_contours.append(contour)
            cv2.rectangle(img,(x-3,y-3),(x+w+3,y+h+3),(0,255,255),2)

    #cv2.imshow('frame', img)

    return detected_contours