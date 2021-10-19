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
import numpy.matlib


#Load the camera calibration parameters from the file "mtx_dist.npz". The file is generated using the calibrate_camera.py script.
path = '/Users/Claus/python_scripts/'
load = np.load(os.path.join(os.path.dirname(path), "mtx_dist.npz"))
mtx = load["mtx"]
dist = load["dist"]

#Set the type of aruco marker the algorithm should search for by changing aruco."DICT_5X5_100"
arucoDict = cv2.aruco.Dictionary_get(aruco.DICT_5X5_100)
arucoParams = cv2.aruco.DetectorParameters_create()
# initialize the video stream from the camera 2 sec sleep is added to varm up sensor
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)


#Define the target positions in image coordinates. The target positions are the desired locations of each of the aruco markers corner points. 
#top-left, top-right, bottom-right, and bottom-left order
target_positions = np.array([(int((1280/2)-100),int((720/2)-100)),(int((1280/2)+100),int((720/2)-100)),(int((1280/2)+100),int((720/2)+100)),(int((1280/2)-100),int((720/2)+100))])


while True:
    #Grabs the frame from the threaded video stream and resizes it to have a maximum width of 1000 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=1000)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Detects ArUco markers in the input frame and outputs corner pixel coordinates
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame,arucoDict, parameters=arucoParams)
    #The found corners are saved in the corners list in the order top-left, top-right, bottom-right, and bottom-left
    corners = np.array(corners)
    
    #Calculate the distance to marker center by outputting tvec. Which is used to approximate Z or depth in the interaction matrix (Depth is magnitude of tvec)
    #Using a 10 cm marker the output corresponds to depth in cm. Using markersize 5 means the output needs to be halfed to be correct in cm 
    markerSizeInCM = 10
    rvec , tvec, _ = aruco.estimatePoseSingleMarkers(corners, markerSizeInCM, mtx, dist)
    #If no marker is found the magnitude of tvec should be calculated
    if tvec is not None:
        mag = np.linalg.norm(tvec)
    
    #The following lines draws the desired position of the 4 corner points on the frame in BLUE
    #cv2.circle(frame, (target_positions[0][0],target_positions[0][1]), 4, (255, 0, 0), -1)
    #cv2.circle(frame, (target_positions[1][0],target_positions[1][1]), 4, (255, 0, 0), -1)
    #cv2.circle(frame, (target_positions[2][0],target_positions[2][1]), 4, (255, 0, 0), -1)
    #cv2.circle(frame, (target_positions[3][0],target_positions[3][1]), 4, (255, 0, 0), -1)    

    #Verify that an ArUco marker was detected or else 
    if len(corners) > 0:
	# flatten the ArUco IDs list
        ids = ids.flatten()
	# loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
	    # The marker corners are always returned
	    # in top-left, top-right, bottom-right, and bottom-left order
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
    #If no marker detected show frame and print "No marker detected"     
    else:
        print("[INFO] No marker detected...")
	# show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    
    #If a marker is found and the depth Z estimated the interaction matrix is generated
    if tvec is not None:    
        L=np.matlib.zeros((2*4,6))  
        # Gain on controller, essentially sets arm speed, although too high of a value will cause the
        # function to diverge.
	lam = 0.5
	
	#Target feature are the detected corners
        target_feature = corners.flatten()
        target_feature = target_feature[:,None]
        
        target_positions = target_positions.flatten()
        target_positions = target_positions[:,None]
        
	#Same depth are used for all four points (Probably not ideal and might not work)
        Z = mag
	
        for i in range(0,4):
            x=corners[i][0]
            y=corners[i][1]
	    #Generate L/interaction matrix which is a 8x6 matrix for 4 pts. In this implementation
            L[i*2:i*2+2,:]=np.matrix([[-1/Z,0,x/Z,x*y,-(1+x*x),y],[0,-1/Z,y/Z,1+y*y,-x*y,-x]])
	#The image feature error calculated in pixels is determined from the found corners and the desired corner positions
        error = target_feature - target_positions
	#The moore-penrose pseudoinverse of matrix is used to determine the velocity screw of the camera
        vel=-lam*np.dot(np.linalg.pinv(L),error)
        
        print(vel)
    
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
