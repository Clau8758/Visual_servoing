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
import PySpin
import EasyPySpin

cap = EasyPySpin.VideoCapture(0)

ret, frame = cap.read()

cv2.imwrite("frame.png", frame)

cap.release()
