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
#vel_msg = Twist()

a=5
input()