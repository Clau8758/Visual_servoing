# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 09:55:08 2021

@author: Claus
"""

from imutils.video import VideoStream
import imutils
import time
import cv2
from cv2 import aruco
import numpy as np
import os


path = '/Users/Claus/python_scripts/'
load = np.load(os.path.join(os.path.dirname(path), "mtx_dist.npz"))
mtx = load["mtx"]
dist = load["dist"]

arucoDict = cv2.aruco.Dictionary_get(aruco.DICT_5X5_100)
arucoParams = cv2.aruco.DetectorParameters_create()
# initialize the video stream from the camera 2 sec sleep is added to varm up sensor
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)



#Define the target positions in image coordinates
#top-left, top-right, bottom-right, and bottom-left order
target_positions = np.array([(int((1280/2)-100),int((720/2)-100)),(int((1280/2)+100),int((720/2)-100)),(int((1280/2)+100),int((720/2)+100)),(int((1280/2)-100),int((720/2)+100))])


#Add Camera calibration and distortion parameters


while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 1000 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=1000)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect ArUco markers in the input frame and output corner pixel coordinates
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame,arucoDict, parameters=arucoParams)
    
    
    # Calculate the distance to marker center which is Z in the interaction matrix (the Z-coord in the tvec)
    markerSizeInCM = 5
    rvec , tvec, _ = aruco.estimatePoseSingleMarkers(corners, markerSizeInCM, mtx, dist)
    if tvec is not None:
        mag = np.linalg.norm(tvec)
    

    
	# verify *at least* one ArUco marker was detected
    if len(corners) > 0:
		# flatten the ArUco IDs list
        ids = ids.flatten()
		# loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
			# The marker corners are always returned
			# in top-left, top-right, bottom-right, and bottom-left
			# order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
			# convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            
			# draw the bounding box of the ArUCo detection
            cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
			# compute and draw the center (x, y)-coordinates of the
			# ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)  
            #Draw the correct position of the 4 corner points
            cv2.circle(frame, (target_positions[0][0],target_positions[0][1]), 4, (255, 0, 0), -1)
            cv2.circle(frame, (target_positions[1][0],target_positions[1][1]), 4, (255, 0, 0), -1)
            cv2.circle(frame, (target_positions[2][0],target_positions[2][1]), 4, (255, 0, 0), -1)
            cv2.circle(frame, (target_positions[3][0],target_positions[3][1]), 4, (255, 0, 0), -1)
            
            # draw the marker distance on the frame
            cv2.putText(frame,'Dist:'+str(mag),(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0), 2)
            
    else:
        print("[INFO] No marker detected...")
	# show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    
    
    

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()