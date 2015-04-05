#!/usr/bin/env python
import numpy as np
import cv2
import math
import sys
cap = cv2.VideoCapture(sys.argv[1])

prev = None
runOnce = False
count = 0
prefix = "./img_"
avg = 0
num_frames = 0
threshold = int(sys.argv[2])
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    if ret:
        if True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray,100,200)
            gray = np.float32(edges)
            dst = cv2.cornerHarris(gray,2,3,0.04)
            # Display the resulting frame
            dst = cv2.dilate(dst,None)

        # Threshold for an optimal value, it may vary depending on the image.
            edges[dst>0.05*dst.max()]=[255]
            cv2.imshow("edges",edges)
        if False:

            if runOnce:
                diff = cv2.absdiff(prev, frame).sum() + 1
                diff = math.log(diff)
                num_frames += 1
                avg += diff
                average_val = int((avg + (.5 * avg))/num_frames)
                if diff > threshold:
                    cv2.imwrite(prefix + str(count) + ".jpg", frame)
                    count += 1
                    #cv2.imshow('frame',frame)
                    print "diff %s avg %s" % (diff, average_val)
            else:
                cv2.imwrite(prefix + str(count) + ".jpg", frame)
                runOnce = True
            """for x in range(len(frame)):
                for y in range(len(frame[x])):
                    frame[x][y] = [0,255,0]"""
        prev = frame
        
        count += 1
    else:
        print "done"
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()