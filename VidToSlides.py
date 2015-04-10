#!/usr/bin/env python
import numpy as np
import math
import sys
import cv2
import subprocess
import random
#download youtube video with youtube-dl
url = sys.argv[1]
if len(sys.argv) > 2:
    already = sys.argv[2]
else:
    already = None
if already == None:
    name = subprocess.check_output(['youtube-dl --id  --get-filename %s' % (url)] ,shell=True) 
    download = subprocess.check_output(['youtube-dl --id  -f mp4 %s' % (url)] ,shell=True)

    location = "./" + name 
    location = location[:-1]
    print "ffmpeg -i %s -codec copy %s.avi" % (location,location)
    subprocess.check_output(["ffmpeg -i %s -codec copy %s.avi" % (location,location)],shell=True)
    location = location + '.avi'
else:
    location = url
cap = cv2.VideoCapture(location)
import sys
prev = None
runOnce = False
count = 0
prefix = "./img_"
avg = 1
num_frames = 0
num_corners = 0

max_contour_orig = np.empty(0)

def detect_lines(frame):
    #print frame[0]
    cv2.line(frame, (0,1000000),(0,-1000000),(75,75,75), 10)
    cv2.line(frame, (-1000000,len(frame)),(1000000,len(frame)),(75,75,75), 10)
    cv2.line(frame, (1000000,0),(-1000000,0),(75,75,75), 10)
    cv2.line(frame, (len(frame[0]),1000000),(len(frame[0]),-1000000),(75,75,75), 10)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    lowThresh = 0.2*high_thresh
    #cv2.imwrite(prefix + str(count) + "thresh.jpg", thresh_im)
    edges = cv2.Canny(gray,lowThresh,high_thresh)
    edges = cv2.GaussianBlur(edges, (7,7),2)
    #cv2.imwrite(prefix + str(count) + "edges.jpg", edges)
    
    lines = cv2.HoughLinesP(edges, 1, math.pi/2, 2, None, 30,1)
    for x1, y1, x2, y2 in lines[0]:
        length = get_length((x1,y1), (x2,y2))
        #print length
        if length > 500000:
            slope = get_slope((x1,y1), (x2,y2))
            if(slope == None):
                cv2.line(frame, (x1,y1 - 3),(x2,y2 + 3),(50,50,50), 10)
            elif (slope == 0):
                cv2.line(frame, (x1 - 3,y1),(x2 + 3,y2),(50,50,50), 10)
    #cv2.imwrite(prefix + str(count) + "lines.jpg", frame)
    #cv2.imshow("thresh_im",frame)
    

def get_length(point1, point2):
    dx = (point1[0] - point2[0])
    dy = (point1[1] - point2[1])
    return (((dx * dx) + (dy * dy)))



def get_slope(point1, point2):
    deltaX = point2[0] - point1[0]
    deltaY = point2[1] - point1[1]
    if deltaX == 0:
        return None
    return deltaY/deltaX


def get_largest_contour(frame, prev_max_contour):
    global count
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray, (3,3),2)
    high_thresh, thresh_im = cv2.threshold(gray, 200, 255, 0)
    lowThresh = 0.5*high_thresh
    edges = cv2.Canny(thresh_im,100,200)
    
    #cv2.imwrite(prefix + str(count) + "gray.jpg", gray)
    #cv2.imwrite(prefix + str(count) + "edges.jpg", edges)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, (5,5))
    #cv2.imshow("thresh_im",thresh_im)
    #cv2.waitKey()
    contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    max_contour_size = get_contour_size(prev_max_contour)
    max_contour = prev_max_contour
    max_contour_id = 0
    for i,contour in zip(range(len(contours)),contours):
        #print(type(contour))
        contour = cv2.approxPolyDP(contour, 100, 1)
        contour = cv2.convexHull(contour)
        contour_size = get_contour_size(contour)
        #cv2.rectangle(frame,(x,y),(x+w,y+h), (random.randint(0,255),random.randint(0,255),random.randint(0,255)),3)
        if contour_size > max_contour_size:
            print "BIGGER"
            max_contour_size = contour_size
            max_contour = contour
            max_contour_id = i

    linecolor = (0,0,255)

    #cv2.line(frame, tuple(max_contour[len(max_contour)-2][0]), start,(255,50,50), 100)

    #cv2.drawContours(frame, contours,-1, (50,0,0), 1)

    #cv2.imwrite(prefix + str(count) + "contour.jpg", frame)

    # cv2.minAreaRect(max_contour)
    #x,y,w,h = cv2.boundingRect(max_contour)
    #cv2.rectangle(gray,(x,y),(x+w,y+h), (random.randint(0,255),random.randint(0,255),random.randint(0,255)),3)
    cv2.imshow("gray",gray)
    #cv2.waitKey(10000)
    return max_contour

def get_contour_size(contour):
    if contour.size == 0:
        return 0
    return cv2.contourArea(contour)

def crop_to_contour(frame, contour):
    mask = np.zeros(frame.shape[:2],np.uint8)
    convex = cv2.convexHull(contour)
    cv2.fillConvexPoly(mask, contour, (255,255,255))
    cv2.imshow("edges",mask)
    cv2.waitKey(10)
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    #print "cropped"
    #print frame
    return frame
    

def crop(image, contour):
    if contour.size == 0:
        return image
    x,y,w,h = cv2.boundingRect(contour)
    return image[y: y + h, x: x + w]
oldFrame = None
while True:
    # Capture frame-by-frame
    for i in range(60):
        ret = cap.grab()
        if not ret:
            sys.exit(0)
    ret, frame = cap.read()
    
    # Our operations on the frame come here
    if ret:
        newFrame = cv2.resize(frame, (100,100))
        if False:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            """edges = cv2.Canny(gray,100,200)
            gray = np.float32(edges)
            dst = cv2.cornerHarris(gray,3,5,0.04)
            dst = cv2.normalize( dst );
            print dst.sum()
            # Display the resulting frame
            dst = cv2.dilate(dst,None)

        # Threshold for an optimal value, it may vary depending on the image.
            edges[dst>0.05*dst.max()]=[255]"""
            cv2.imshow("edges",gray)
        if True:

            if runOnce:
                #detect_lines(frame)
                frame2 = np.copy(frame)
                detect_lines(frame2)
                max_contour_orig = get_largest_contour(frame2, max_contour_orig)
                #print crop(frame2, max_contour_orig)
                frame = crop(frame, max_contour_orig)
                #cv2.imwrite(prefix + str(count) + ".jpg", frame)
                gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                

                dst = cv2.cornerHarris(gray,2,3,0.04)
                dst = cv2.dilate(dst,None)

                # Threshold for an optimal value, it may vary depending on the image.
                frame2[dst>0.0001*dst.max()]=[0,0,255]
                
                #dst = cv2.normalize( dst );
                diff = dst.sum()
                num_frames += 1
                avg += diff
                diff = math.fabs(int(dst.sum()))

                average_val = math.fabs(int((avg + (0 * avg))/num_frames))
                
                cv2.waitKey(1)

                print "current %s avg %s diff %s avg_threshold %s" % (diff, average_val, math.fabs(average_val - diff), average_val/4)
                #print "diff %s avg %s" % (diff, average_val)
                if math.fabs(average_val - diff) > average_val / 8:
                    cv2.imshow('frame',frame)
                    cv2.imwrite(prefix + str(count) + ".jpg", oldFrame)
                    count += 1
                    print "TRIGGERING AD DJ %s avg %s" % (math.fabs(average_val - diff), average_val/4)
                    num_frames = 0
                    
                    avg = 0
                oldFrame = frame
                if num_frames == 5:
                    num_frames = 0
                    avg = 0

            else:
                oldFrame = frame
                cv2.imwrite(prefix + str(count) + ".jpg", frame)
                count+=1
                runOnce = True
            """for x in range(len(frame)):
                for y in range(len(frame[x])):
                    frame[x][y] = [0,255,0]"""
        prev = frame
        
        
    else:
        print "done"
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()