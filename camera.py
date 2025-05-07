# imports necessary libraries
import cv2
import sys

def get_livestream(camera):

    if len(sys.argv) > 1:
        camera = sys.argv[1]

    # makes the camera able to be used by OpenCV
    source = cv2.VideoCapture(camera = 0) 

    """
    y_start = 100
    y_end = 300
    x_start = 50
    x_end = 250
    """

    # makes a window to display the source feed
    win_name = 'Camera Preview'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while cv2.waitKey(1) != 27: # keeps the source open until ESC is pressed
        has_frame, frame = source.read()
        if not has_frame:
            break

        # cropped_stream = frame[y_start:y_end, x_start:x_end]
        cv2.imshow(win_name, frame)

    source.release()
    cv2.destroyWindow(win_name)
