#!/usr/bin/env python
import numpy as np
import math
import sys
import cv2
import subprocess
#download youtube video with youtube-dl
url = sys.argv[1]
name = subprocess.check_output(['youtube-dl --id  --get-filename %s' % (url)] ,shell=True) 
download = subprocess.check_output(['youtube-dl --id  -f mp4 %s' % (url)] ,shell=True)

location = "./" + name 
location = location[:-1]
print "ffmpeg -i %s -codec copy %s.avi" % (location,location)
subprocess.check_output(["ffmpeg -i %s -codec copy %s.avi" % (location,location)],shell=True)
location = location + '.avi'
cap = cv2.VideoCapture(location)
import sys
prev = None
runOnce = False
count = 0
prefix = "./img_"
avg = 1
num_frames = 0
num_corners = 0
while True:
    # Capture frame-by-frame
    for i in range(50):
        ret = cap.grab()
        if not ret:
            sys.exit(0)
    ret, frame = cap.read()
    
    # Our operations on the frame come here
    if ret:
        newFrame = cv2.resize(frame, (100,100))
        if False:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray,100,200)
            gray = np.float32(edges)
            dst = cv2.cornerHarris(gray,3,5,0.04)
            dst = cv2.normalize( dst );
            print dst.sum()
            # Display the resulting frame
            dst = cv2.dilate(dst,None)

        # Threshold for an optimal value, it may vary depending on the image.
            edges[dst>0.05*dst.max()]=[255]
            cv2.imshow("edges",edges)
        if True:

            if runOnce:
                gray = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY)
                gray = np.float32(gray)
                dst = cv2.cornerHarris(gray,2,3,0.04)
                dst = cv2.normalize( dst );
                diff = dst.sum()
                num_frames += 1
                avg += diff
                #diff = cv2.absdiff(prev, frame).sum() + 1
                #diff = math.log(diff)
                diff = math.fabs(int(dst.sum()))

                average_val = math.fabs(int((avg + (0 * avg))/num_frames))
                #print "diff %s avg %s" % (diff, average_val)
                if math.fabs(average_val - diff) > average_val / 4:
                    cv2.imwrite(prefix + str(count) + ".jpg", frame)
                    count += 1
                    #cv2.imshow('frame',frame)
                    print "diff %s avg %s" % (diff, average_val)
                    num_frames = 0
                    
                    avg = 0

                if num_frames == 30:
                    num_frames = 0
                    avg = 0
            else:
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