import cv2 as cv
from tracker import *

#create Tracker Object
tracker= EuclideanDistTracker()

cap = cv.VideoCapture('highway.mp4')

object_detector= cv.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
limits= [480, 650,800,640]
while True:
    ret,frame= cap.read()

    #extract region of interest
    roi = frame[340:600, 500:800]

    #object detection
    mask= object_detector.apply(roi)
    _, mask= cv.threshold(mask, 254, 255, cv.THRESH_BINARY)

    contours,_ =cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    detection= []
    for cnt in contours:
        #calculate area and remove small elements
        area= cv.contourArea(cnt)
        if area>100:
            # cv.drawContours(roi, [cnt], -1, (0,255,0),2)
            x,y,w,h= cv.boundingRect(cnt)

            detection.append([x,y,w,h])
    #Object Tracking
    boxes_id= tracker.update(detection)
    for box_id in boxes_id:
        x,y,w,h,id= box_id
        cv.putText(roi, str(id), (x,y-15), cv.FONT_HERSHEY_PLAIN, 1, (255,0,0), 4)
        cv.rectangle(roi, (x, y), (x + w, y + h), (0, 225, 0), 3)

    cv.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (225, 0, 225), 5)

    cv.imshow('roi',roi)
    cv.imshow('Frame', frame)
    cv.imshow('mask', mask)
    key= cv.waitKey(0)
    if key== 27:
        break
cap.release()
cv.destroyAllWindows()
