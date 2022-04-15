# Importing Libraries

import cv2            # for processing images,videos to identify objects,faces and even handwriting

import numpy as np    # for mathematical and logical operations on arrays


# Initializing Subtractor(Algorithm)

algo = cv2.bgsegm.createBackgroundSubtractorMOG()

min_width_rect = 80
min_height_rect = 80


count_line_position = 550              # Drawing a line

def center_point(x,y,w,h):              # defining function for making center point of rectangle
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy

count_detect = []            # empty list to add counter of vehicles

offset = 6              # Allowable error between pixels

counter = 1


# Web Camera

cap = cv2.VideoCapture('video.mp4')

while True:
    ret,frame = cap.read()

    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)       # Converting to grey colour

    blur = cv2.GaussianBlur(grey,(5,5),5)               # for making blur

    img_detect = algo.apply(blur)                      # Applying on each frame

    dil = cv2.dilate(img_detect,np.ones((5,5)))              # It adds pixels on the object boundaries

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    dil_data = cv2.morphologyEx(dil,cv2.MORPH_CLOSE,kernel)           # for performing advanced morphological operations

    dil_data = cv2.morphologyEx(dil_data,cv2.MORPH_CLOSE,kernel)

    counter_sh,h = cv2.findContours(dil_data,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)          # contours are useful tools for shape analysis,object detection and recognition

    cv2.line(frame,(25,count_line_position),(1200,count_line_position),(255,0,0),5)

    # Drawing rectangle

    for (i,c) in enumerate(counter_sh):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_rect) and (h >= min_height_rect)
        if not validate_counter:
            continue
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

        # displaying counters of vehicles on rectangle(vehicle)

        cv2.putText(frame,"Vehicle: "+str(counter),(x,y-20),cv2.FONT_HERSHEY_TRIPLEX,1,(0,255,255),2)


        # for circle mark

        center = center_point(x,y,w,h)
        count_detect.append(center)
        cv2.circle(frame,center,4,(0,0,255),-1)

        for (x,y) in count_detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                counter = counter+1
            cv2.line(frame,(25,count_line_position),(1200,count_line_position),(0,127,255),5)
            count_detect.remove((x,y))
            print('Vehicle Counter: '+str(counter))


    # displaying counters with VEHICLE COUNTER name

    cv2.putText(frame,"VEHICLE COUNTER: "+str(counter),(500,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)





    #cv2.imshow('Detector',dil_data)
    cv2.imshow('Original Video',frame)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()
    