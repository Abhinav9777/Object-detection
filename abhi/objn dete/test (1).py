import cv2 as cv
from tracker import *


# Function to read speed information from notepad file
def read_speed_info(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if 'curRadarSpeed' in line:
                    speed_info = line.strip()  # Extract the line with speed information
                    speed_value = int(speed_info.split(':')[-1].strip())  # Extract the speed value
                    if speed_value > 80:
                        return speed_info
        # If no speed above 80 is found
        print("No speed information above 80 found in the notepad file.")
        return None
    except FileNotFoundError:
        print("Notepad file not found: {}".format(file_path))
        return None



#create Tracker Object
tracker= EuclideanDistTracker()

cap = cv.VideoCapture('rtsp://admin:Ibigroup10!@172.16.23.7/Streaming/Channels/101')

object_detector= cv.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
limits= [480, 580,800,560]
totalcount=[]
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

        cx, cy = (x+w// 2), (y+h// 2)
        cv.circle(roi, (cx, cy), 5, (225, 0, 225), cv.FILLED)
        if limits[0] < cx < limits[2] and limits[1] -30 < cy < limits[1] + 30:
            if totalcount.count(id) == 0:
                totalcount.append(id)
                cv.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 225, 0), 5)

    cv.putText(frame, f'Count : {len(totalcount)}', (50, 50),cv.FONT_HERSHEY_PLAIN,1, (255,0,0), 4 )

    cv.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (225, 0, 225), 5)

    cv.imshow('roi',roi)
    cv.imshow('Frame', frame)
    cv.imshow('mask', mask)
    key= cv.waitKey(0)
    if key== 27:
        break
cap.release()
cv.destroyAllWindows()
