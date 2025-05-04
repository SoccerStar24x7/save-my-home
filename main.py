# imports necessary libraries
import cv2
import sys

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

# makes the camera able to be used by OpenCV
source = cv2.VideoCapture(s) 

# makes a window to display the source feed
win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

while cv2.waitKey(1) != 27: # keeps the source open until ESC is pressed
    has_frame, frame = source.read()
    if not has_frame:
        break
    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)

