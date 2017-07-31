#!/usr/bin/env python
import numpy as np
import cv2


kernel = np.ones((5,5),np.uint8)
N = 3
for j in range(1, N):
    print(j)
    img = cv2.imread('%d.jpg' %j, cv2.IMREAD_COLOR)
    width, height = img.shape[:2]
    print(width)
    print(height)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    cv2.imwrite('mesS.jpg', s)
    cv2.imwrite('mesH.jpg', h)
    cv2.imwrite('mesV.jpg', v)
    averages = 0
    summary = 0
    mHist, bins = np.histogram(s, 255, [0, 255])
    for i in range(0, 255):
        averages += mHist[i] * i
    summary = sum(mHist)
    average = averages / summary
    average = average + 15
    print(average)
    th, dst = cv2.threshold(s, average, 255, cv2.THRESH_BINARY)
    dst = cv2.blur(dst, (5, 5))
    dst = cv2.dilate(dst, kernel, iterations=1)
    mask = cv2.erode(dst, kernel, iterations=1)

    cv2.imwrite('%dSmask.jpg' %j, mask)



