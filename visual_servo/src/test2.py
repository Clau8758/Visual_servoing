import cv2 as cv
import sys
img = cv.imread("/home/johan/GitHub/Visual_servoing/visual_servo/src/sim.png")
if img is None:
    sys.exit("Could not read the image.")
cv.imshow("Display window", img)
input()
cv.waitKey(0)
