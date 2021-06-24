import realsense as rs
import cv2
import time
global_cam = rs.Camera()
idx=1700
while True :
    img = global_cam.capture()
    cv2.imshow('capture image', img)
    k=cv2.waitKey(1)
    if k == ord('s'):
    #time.sleep(0.1)
        print("save the capture image_{}".format(idx))
        cv2.imwrite("./capture_images/{}.png".format(idx), img)
        idx+=1
   # if idx==600:
    #    break

