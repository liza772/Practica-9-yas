import cv2
import numpy as np

img_rgb = cv2.imread('circuitos.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template2 = cv2.imread('circuitos_match2.jpg',0)
w, h = template2.shape[::-1]

res = cv2.matchTemplate(img_gray,template2,cv2.TM_CCOEFF_NORMED)
threshold = 0.85
loc = np.where( res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (255,0,255), 2)

template3 = cv2.imread('circuitos_match3.jpg',0)
w, h = template2.shape[::-1]

res = cv2.matchTemplate(img_gray,template3,cv2.TM_CCOEFF_NORMED)
threshold = 0.85
loc = np.where( res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (255,255,0), 2)

roi = cv2.selectROI('roi',img_rgb)
recRoi = img_rgb[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
cv2.imwrite('roi.jpg',recRoi)

template = cv2.imread('roi.jpg',0)
template = cv2.resize(template,"800x600")
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.85
loc = np.where( res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

cv2.imshow('Detected',img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()