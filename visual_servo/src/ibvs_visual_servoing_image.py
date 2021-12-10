#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from imutils.video import VideoStream
import imutils
import time
import cv2
from cv2 import aruco
import numpy as np
import os
import numpy.matlib
import glob
import pathlib


#Creating rosnode with name "desired velocity"
#rospy.init_node('desired_velocity', anonymous=True)
#Creates a ros publisher that publishes the camera velocities to the to topic velocity_controller/cmd_vel
#velocity_publisher = rospy.Publisher('/twist_controller/command', Twist, queue_size=10)
vel_msg = Twist()


#Load the camera calibration parameters from the file "mtx_dist.npz". The file is generated using the calibrate_camera.py script.
#For the simulation the camera parameters are not needed
#os.path.join(os.path.dirname(path)
load = np.load("/home/johan/GitHub/Visual_servoing/visual_servo/src/mtx_dist.npz")
mtx = load["mtx"]
dist = load["dist"]


#Set the type of aruco marker the algorithm should search for by changing aruco."DICT_5X5_100"
arucoDict = cv2.aruco.Dictionary_get(aruco.DICT_4X4_1000)
arucoParams = cv2.aruco.DetectorParameters_create()

#Define the target positions in image coordinates. The target positions are the desired locations of each of the aruco markers corner points. 
#top-left, top-right, bottom-right, and bottom-left order
#target_positions = np.array([(int((1280/2)-100),int((720/2)-100)),(int((1280/2)+100),int((720/2)-100)),(int((1280/2)+100),int((720/2)+100)),(int((1280/2)-100),int((720/2)+100))]).flatten()


while True:
    #Grabs the frame from the threaded video stream and resizes it to have a maximum width of 1000 pixels
    #frame = glob.glob('sim.jpg')
    frame = cv2.imread("/home/johan/GitHub/Visual_servoing/visual_servo/src/sim.png")
    #If another process was using the image try reading it again
    #if type(frame) is not np.ndarray:
    	#print("busy")
    #	continue    	
    #frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    down_width = 512
    down_height = 384
    down_points = (down_width, down_height)
    frame= cv2.resize(frame, down_points, interpolation= cv2.INTER_LINEAR)
    h,w,c = frame.shape 
    print("width: "+ str(w),"Height: "+ str(h))
    #Define the target positions in image coordinates. The target positions are the desired locations of each of the aruco markers corner points. 
    #top-left, top-right, bottom-right, and bottom-left order
    target_positions = np.array([(int((w/2)-180),int((h/2)-160)),(int((w/2)+180),int((h/2)-160)),(int((w/2)+180),int((h/2)+160)),(int((w/2)-180),int((h/2)+160))]).flatten()    

    
    #Detects ArUco markers in the input frame and outputs corner pixel coordinates
    #CORNERS COORDINATES IS IN (u,v) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (corners, ids, rejected) = cv2.aruco.detectMarkers(gray,arucoDict, parameters=arucoParams)
    
    #The found corners are saved in the corners list in the order top-left, top-right, bottom-right, and bottom-left
    corners = np.array(corners)
    
    #Calculate the distance to marker center by outputting tvec. Which is used to approximate Z or depth in the interaction matrix (Depth is magnitude of tvec)
    #Using a 10 cm marker the output corresponds to depth in cm. Using markersize 5 means the output needs to be halfed to be correct in cm 

    markerSizeInMeter = 0.01
    rvec , tvec, objPoints  = aruco.estimatePoseSingleMarkers(corners, markerSizeInMeter, mtx, None)
    length = 0.0025
    frame = cv2.aruco.drawAxis(frame, mtx, None, rvec, tvec, length)
    #print(rvec)
    #If no marker is found the depth of each corner should not be calculated
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)


    if tvec is not None:
        #To estimate the 3D parameters of each corner the pose of the aruco marker is used
        MarkerHalf = markerSizeInMeter / 2.0
        
        #Using the marker center as origin the corners all lie in the same xy-plane
        #converts rotation vector rvec to rotation matrix mrv
        mrv, jacobian = cv2.Rodrigues(rvec)
        print(mrv)        
        
        #in markerworld the corners are all in the xy-plane so z is zero at first

        
        ### Punktet MarkerHalf_x i X retning drejes med x-kolonden fra rotatonsmatricen
        X_half = MarkerHalf * mrv[:,0] #rotate the x = mhalf
        ### Punktet MarkerHalf_y i Y retning drejes med y-kolonden fra rotatonsmatricen
        Y_half = MarkerHalf * mrv[:,1] #rotate the y = mhalf
        
        
        
        minusX_half = X_half * (-1)
        minusY_half = Y_half * (-1)
        
        #translation 
        markercorners=np.array([(np.add(minusX_half, Y_half)),(np.add(X_half, Y_half)),(np.add( X_half, minusY_half)),(np.add(minusX_half, minusY_half))]) #order upper left,upper right,lower right,lower left in markerworld
        
        Z = np.empty([4],dtype=np.float16)
        for i, mc in enumerate(markercorners):
            markercorners[i] = np.add(tvec,mc) #add tvec to each corner
            Z[i] = (np.linalg.norm(markercorners[i])) #HVORFOR divideret med 1/100????? til cm?
        #The magnitude of each markercorner vector is the depth Z for each corner
        #print('Vec X, Y, C, dot(X,Y)', X,Y,C, np.dot(X,Y)) # just for debug
        #markercorners = np.array(markercorners,dtype=np.float32) # type needed when used as input to cv2    1637829881711501836
        #mag = np.linalg.norm(tvec)
        
    #The following lines draws the desired position of the 4 corner points on the frame in BLUE
    cv2.circle(frame, (target_positions[0],target_positions[1]), 4, (255, 0, 0), -1)
    cv2.circle(frame, (target_positions[2],target_positions[3]), 4, (255, 0, 0), -1)
    cv2.circle(frame, (target_positions[4],target_positions[5]), 4, (255, 0, 0), -1)
    cv2.circle(frame, (target_positions[6],target_positions[7]), 4, (255, 0, 0), -1)    

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
    cv2.waitKey(0)
    #input()
    #key = cv2.waitKey(1) & 0xFF
	#if the `q` key was pressed, break from the loop
    #if key == ord("q"):
    #    break
    
    #If a marker is found and the depth Z estimated the interaction matrix is generated
    if tvec is not None:    
        L=np.matlib.zeros((2*4,6))  
        # Gain on controller, essentially sets arm speed, although too high of a value will cause the
        # function to diverge.
        
        x_lin=1
        y_lin=1
        z_lin=1
        x_ang=1
        y_ang=1
        z_ang=1

        
        
        lam = np.array([[x_lin,0,0,0,0,0], #x_lin
        		 [0,y_lin,0,0,0,0],        #y_lin
        		 [0,0,z_lin,0,0,0],      #z_lin
        		 [0,0,0,x_ang,0,0],         #x_ang
        		 [0,0,0,0,y_ang,0],          #y_ang
        		 [0,0,0,0,0,z_ang]])         #z_ang
	
	#Target features are the detected corners and their xy-positions
        target_feature = corners.flatten()
        #target_feature = target_feature[:,None]
        
	#Same depth are used for all four points (Probably not ideal and might not work)
        
	
        for i in range(0,4):
            x=corners[i][0]
            y=corners[i][1]
	    #Generate L/interaction matrix which is a 8x6 matrix for 4 pts. In this implementation
            L[i*2:i*2+2,:]=np.matrix([[-1/Z[i],0,x/Z[i],x*y,-(1+x*x),y],[0,-1/Z[i],y/Z[i],1+y*y,-x*y,-x]])
	    #The image feature error calculated in pixels is determined from the found corners and the desired corner positions
        error = target_feature - target_positions
        print(error)
	   #The moore-penrose pseudoinverse of matrix is used to determine the velocity screw of the camera
        vel=np.dot(-lam,np.transpose(np.dot(np.linalg.pinv(L),error)))
        #debug=np.linalg.pinv(L)
        vel_msg.linear.x = vel[0] 
        vel_msg.linear.y = vel[1] 
        vel_msg.linear.z = vel[2]  
        vel_msg.angular.x = vel[3]
        vel_msg.angular.y = vel[4]
        vel_msg.angular.z = vel[5]
        print(vel_msg)
        #input()
        velocity_publisher.publish(vel_msg)

        
# do a bit of cleanup
cv2.destroyAllWindows()


