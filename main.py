import numpy as np
import matplotlib.pyplot as plt
import cv2

corner_track_params = dict(maxCorners = 20, qualityLevel = 0.5, minDistance = 10, blockSize = 7)
#lucas-kanade optical flow parameters
lk_params = dict(winSize=(300,300), maxLevel = 1, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

ret , prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

#the points we want to track
prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **corner_track_params)

mask = np.zeros_like(prev_frame) #for displaying the line on video

while True:

    ret , frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    nextPts , status , error = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_pts, None, **lk_params)

    good_new = nextPts[status==1]
    good_prev = prev_pts[status==1]

    for i, (new, prev) in enumerate(zip(good_new, good_prev)):

        x_new, y_new = new.ravel()
        x_prev, y_prev = prev.ravel()
        x_new = int(x_new)
        y_new = int(y_new)
        x_prev = int(x_prev)
        y_prev = int(y_prev)
        mask = cv2.line(mask,(x_new,y_new),(x_prev,y_prev),(255,0,0),3)
        frame = cv2.circle(frame, (x_new, y_new), 8, (0,0,255), -1)
    
    img = cv2.add(frame, mask)

    cv2.imshow("Tracking", img)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

    prev_gray = frame_gray.copy()
    prev_pts = good_new.reshape(-1, 1, 2)


cv2.destroyAllWindows()
cap.release()