# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:05:01 2021

@author: Claus
"""

import numpy as np
import cv2
import glob
import yaml
import pathlib
import os

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#/home/johan/GitHub/Visual_servoing
images = glob.glob(r'/home/johan/GitHub/Visual_servoing/calibration_images/*.png')

path = 'results'
pathlib.Path(path).mkdir(parents=True, exist_ok=True) 

found = 0
mean_error = 0

for fname in images:  # Here, 10 can be changed to whatever number you like to choose
    img = cv2.imread(fname) # Capture frame-by-frame
    #print(images[im_i])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)   # Certainly, every loop objp is the same, in 3D.
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners2, ret)
        found = found + 1
        cv2.imshow('img', img)
        cv2.waitKey(10)
        # if you want to save images with detected corners 
        # uncomment following 2 lines and lines 5, 18 and 19
        image_name = path + '/result' + str(found) + '.png'
        cv2.imwrite(image_name, img)

print("Number of images used for calibration: ", found)

# When everything done, release the capture
# cap.release()
cv2.destroyAllWindows()



# calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savez(os.path.join(os.path.dirname(__file__), "mtx_dist.npz"), mtx=mtx, dist=dist)
#New camera matrix for undistort
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))


#Reprojection error
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

path = 'distresults'
pathlib.Path(path).mkdir(parents=True, exist_ok=True) 
images = glob.glob(r'calibration_images/*.png')
dist_img=0

for kage in images:  # Here, 10 can be changed to whatever number you like to choose
    img = cv2.imread(kage) # Capture frame-by-frame
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    dist_img = dist_img + 1
    #crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    image_name = path + '/distresult' + str(dist_img) + '.png'
    cv2.imwrite(image_name, img)
    
print("Number of images undistorted: ", dist_img)


# transform the matrix and distortion coefficients to writable lists
data = {'camera_matrix': np.asarray(mtx).tolist(),
        'dist_coeff': np.asarray(dist).tolist()}

# and save it to a file
with open("calibration_matrix.yaml", "w") as f:
    yaml.dump(data, f)

# done