#!/usr/bin/env python
import numpy as np
import cv2
from matplotlib import pyplot as plt
import csv
SZ = 20

def deskew(img):
    #img = cv2.resize(img,(480, 800) , interpolation=cv2.INTER_AREA)
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed.
        return img.copy()
    # Calculate skew based on central momemts.
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness.
    M = np.float32([[1, skew, -0.5*width*skew], [0, 1, 0]])

    # Apply affine transform
    img = cv2.warpAffine(img, M, (480, 800), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

kernel = np.ones((5,5),np.uint8)
N = 11
#gest = 10
czarny = 58
for j in range(1, N):
    print(j)
    #img = cv2.imread('%d' %gest + '/%d.jpg' %j, cv2.IMREAD_COLOR)
    img = cv2.imread('test/%d.jpg' %j, cv2.IMREAD_COLOR)
    width, height = img.shape[:2]
    print(width)
    print(height)
    if j >= czarny:
        img = cv2.bitwise_not(img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    #cv2.imwrite('mesS.jpg', s)
    #cv2.imwrite('mesH.jpg', h)
    #cv2.imwrite('mesV.jpg', v)
    averages = 0
    summary = 0
    mHist, bins = np.histogram(s, 255, [0, 255])
    for i in range(0,255):
        averages += mHist[i] * i
    summary = sum(mHist)
    average = averages / summary
    average = average + 15
    print(average)
    th, dst = cv2.threshold(s, average, 255, cv2.THRESH_BINARY)

    #hth, hdst = cv2.threshold(h, haverage, 255, cv2.THRESH_BINARY)
    dst = cv2.blur(dst, (5, 5))
    dst = cv2.dilate(dst, kernel, iterations=1)
    mask = cv2.erode(dst, kernel, iterations=1)
    #mask1 = deskew(mask)
    mask1 = cv2.resize(mask,(480, 800) , interpolation=cv2.INTER_AREA)

    #cv2.imwrite('%dmask' %gest + '/%d.jpg' %j, mask)
    cv2.imwrite('testmask/%d.jpg' %j, mask)
    #cv2.imwrite('%dmask' %gest + '/dys_%d.jpg' % j, mask1)
    cv2.imwrite('testmask/dys_%d.jpg' % j, mask1)
    #cv2.imwrite('1S/Hmask%d.jpg' % j, hmask)
    #res = cv2.bitwise_and(img, img, mask=mask)
    #cv2.imwrite('1S/%d.jpg' %j, res)
    #cv2.imwrite('source.jpg', img)
    #plt.hist(h.ravel(), 180, [0, 180])
    #plt.title('Histogram for gray scale picture')
    #plt.show()

