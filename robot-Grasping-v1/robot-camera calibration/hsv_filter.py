import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import urx
import copy

import Robot_env.robot_util as ru

# : 시작 1 : HSV를 이용해서, 캘리브레이션 포인트(엔드이펙터) 설정
# : 다음 2 : joint collector 실행

def nothing(x):
    pass


# : import 한 kinect_snap의 global_cam 클래스를 불러온 오브젝트
# global_cam = Kinect_Snap.global_cam()  # Load Camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipe_profile = pipeline.start(config)

img = None
for x in range(30):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Intrinsics & Extrinsics
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    img = color_image
img = cv2.resize(img, (int(1280 / 2), int(720 / 2)))  # : 이미지 변형

hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # : 이미지를 RGB에서 HSV로 변환후 저장

# : opencv placeholder 윈도우를 생성 이름'result'
cv2.namedWindow('result')
cv2.namedWindow('th3')
cv2.namedWindow('bar')
cv2.resizeWindow('bar', 640, 320)

# : opencv 함수 트랙바 생성 (트랙바의 이름, 띄울 창, 0~n, 변화시)
cv2.createTrackbar('low_h', 'bar', 0, 180, nothing)
cv2.createTrackbar('low_s', 'bar', 0, 255, nothing)
cv2.createTrackbar('low_v', 'bar', 0, 255, nothing)

cv2.createTrackbar('high_h', 'bar', 0, 180, nothing)
cv2.createTrackbar('high_s', 'bar', 0, 255, nothing)
cv2.createTrackbar('high_v', 'bar', 0, 255, nothing)
cv2.setTrackbarPos('high_h', 'bar', 180)
cv2.setTrackbarPos('high_s', 'bar', 255)
cv2.setTrackbarPos('high_v', 'bar', 255)

date = "20200813"
try:
    print('making dir ... ', end='\t')
    os.mkdir(date)
except:
    print('already exist')
print('done')

put_it_on_flag = 0

# : RL robot
rob1 = ru.Robot_util("192.168.0.52")    # : Cam
rob2 = ru.Robot_util("192.168.0.29")    # : gripper
rob1_home_joint_rad = np.deg2rad([0.3209, -113.0970, -4.5383, -152.3580, 89.6613, 1.2152])  # : Cam Pose
rob2_home_joint_rad_b = np.deg2rad([40.9664, -74.1802, 117.9032, -112.9013, 247.8042, -224.6624 + 180])  # : Tray Pose - big
rob2_home_joint_rad_s = [0.7150, -1.29469, 2.0578, -1.9705, 4.3250, -3.9211]  # : Tray Pose(rad) - small

rob1.set_tcp([0, 0, 0.153, 0, 0, 0])
rob2.set_tcp([0, 0, 0.170, 0, 0, 0])

rob1.movej(rob1_home_joint_rad, 0.5, 0.5)
rob2.movej(rob2_home_joint_rad_s, 0.5, 0.5)

while True:  # : 프로그램이 돌아가는 영역 - 반복
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Intrinsics & Extrinsics
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    img = color_image
    # img = cv2.resize(img, (int(1280 / 2), int(720 / 2)))  # : 이미지 변형
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # : 이미지를 RGB에서 HSV로 변환후 저장

    # get info from track bar and appy to result    # : 트랙바의 현재 상태를 받아 저장 (해당 트랙바 이름, 뜨운 창)
    low_h = cv2.getTrackbarPos('low_h', 'bar')
    low_s = cv2.getTrackbarPos('low_s', 'bar')
    low_v = cv2.getTrackbarPos('low_v', 'bar')

    high_h = cv2.getTrackbarPos('high_h', 'bar')
    high_s = cv2.getTrackbarPos('high_s', 'bar')
    high_v = cv2.getTrackbarPos('high_v', 'bar')

    # Normal masking algorithm
    lower_color = np.array([low_h, low_s, low_v])  # : 각  h,s,v를 저장하는 배열생성
    upper_color = np.array([high_h, high_s, high_v])  # : 각 최대 값

    # : 스레스홀드를 lower_color로 지정하여, 이하는 0값을 출력, 범위안의 것은 255를 출력하여 마스크를 생성
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # : 마스크를 씌운 이미지와 마스크를 씌우지 않은 이미지에서 모두 0이 아닌경우에만 출력
    result = cv2.bitwise_and(img, img, mask=mask)

    ret, thresh = cv2.threshold(result, 16, 255, cv2.THRESH_BINARY)  # : 스레스홀드 127로 설정, 최대 255
    blurred = cv2.medianBlur(thresh, 5)  # : 메디안 필터를 이용한 블러
    blurred = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)  # : 그레이스케일로 변환
    # th3 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,
    #                             2)  # : 가우시안, 어뎁티브 스레스홀드
    _, th3 = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)

    # _, contours, hierarchy = cv2.findContours(th3.copy(), 1, 2)  # :
    # _, contours, hierarchy = cv2.findContours(th3, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)  # : opencv 3
    contours, hierarchy = cv2.findContours(th3, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)  # : opencv 4
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    # radius = 0
    max_radius = 0
    cx = None
    cy = None
    for cnt in contours:
        if cv2.contourArea(cnt) < 60000:
            # hull = cv2.convexHull(cnt)
            # (cx, cy), radius = cv2.minEnclosingCircle(hull)
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)

            if radius > max_radius:
                max_radius = radius

    fontScale = 1
    color = (0, 0, 255)  # : BGR
    location = (0, 50)
    font = cv2.FONT_ITALIC
    try:
        cx = int(cx)
        cy = int(cy)

        if 0 < max_radius <= 3:
            print("-->>hsv : put it closer!")
            cv2.rectangle(result, (0, 0), 1280, 720, (0, 0, 255), 2)  # draw circle in red color
        elif max_radius is 0:
            text = "-->>hsv : NoBall yeah. put it ON!"
            cv2.putText(result, text, location, font, fontScale, color)
            cv2.rectangle(result, (0, 0), 1280, 720, (0, 0, 255), 2)  # draw circle in red color
        else:
            cv2.circle(result, (int(cx), int(cy)), int(max_radius), (0, 0, 255), 2)  # draw circle in red color
            put_it_on_flag = 0
    except:
        put_it_on_flag += 1
        if 1 <= put_it_on_flag:
            if put_it_on_flag == 1:
                print("-->>hsv : Can't Find the ball")

        else:
            pass

        text = "-->>hsv : Can't Find the ball"
        thickness = 1
        color = (0, 0, 255)  # : BGR
        location = (0, 100)
        font = cv2.FONT_ITALIC
        cv2.rectangle(result, (0, 0), (1280, 720), (0, 0, 255), 2)  # draw circle in red color
        cv2.putText(result, text, location, font, thickness, color)

    cv2.moveWindow('bar', 1920 - int(1280 / 2) - 1, 0)
    cv2.moveWindow('result', 1920 - int(1280 / 2) - 1, 320)
    cv2.moveWindow('th3', 1920 - int(1280 / 2) - 1, 680)

    img_ = copy.deepcopy(img)
    th_list = np.argwhere(th3 == 255)
    img_[th_list[:, 0], th_list[:, 1], :] = 255
    result_ = cv2.resize(result, (int(1280 / 2), int(720 / 2)))
    # th3_ = cv2.resize(th3, (int(1280 / 2), int(720 / 2)))
    img_ = cv2.resize(img_, (int(1280 / 2), int(720 / 2)))

    cv2.imshow('result', result_)
    # cv2.imshow('th3', th3_)
    cv2.imshow('th3', img_)

    # _, contours, hierarchy = cv2.findContours(th3, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)  # :
    # cv2.drawContours(th3, contours, -1, 127, 2)
    # cv2.imshow('th3', th3)

    # cv2.imshow('bar', _)
    k = cv2.waitKey(1)
    if k == ord('s'):
        print("-->>sys :  low__Result : H : {}, S : {}, V : {}".format(low_h, low_s, low_v))
        print("-->>sys :  high_Result : H : {}, S : {}, V : {}".format(high_h, high_s, high_v))
        time_str = time.strftime('%Y%m%d-%H-%M-%S', time.localtime(time.time()))
        cv2.imwrite("./{}/img_{}.png".format(date, time_str), img)
        # cv2.imwrite("./{}/mask_{}.png".format(date, time_str), mask)
        # cv2.imwrite("./{}/result_{}.png".format(date, time_str), result)
        print("-->>sys :  img_saved ")

    if k == ord('h'):
        center = np.deg2rad([-35.45, -62.40, 117.18, -0.87, 67.47, -207.88])
        rob2.movej(center, 1, 1)
    if k == ord('['):
        left_front = [-0.7646004409924503, -0.2769095597792518, -0.09287284815199578,
                      -2.6180463593646133, 0.00021286950964079492, 9.026852509835078e-05]
        rob2.movel(left_front, 1, 1)
    if k == ord(';'):
        left_back = [-0.29020377434533084, -0.2769095596279693, -0.09287284816344174,
                     -2.6180463593017174, 0.00021286953063727208, 9.026852449625093e-05]
        rob2.movel(left_back, 1, 1)
    if k == ord(']'):
        right_front = [-0.75860210749507, 0.3132149404349904, -0.09287284813507735,
                       -2.618046359176607, 0.00021286949160195535, 9.026851967238821e-05]
        rob2.movel(right_front, 1, 1)
    if k == ord('\''):
        right_back = [-0.288681430161438, 0.31321494043497256, -0.09287284813508721,
                      -2.6180463591763017, 0.00021286949158524576, 9.026852011591973e-05]
        rob2.movel(right_back, 1, 1)

    if k & 0xFF == 27:  # ESC
        print("-->>sys :  Exit program ")

        file_name = "hsv_{}.txt".format(date)
        file_path = os.path.join(date + "/", file_name)
        f = open(file_path, "a")
        f.write("{}".format(date, time.strftime('%Y%m%d-%H-%M-%S', time.localtime(time.time()))))
        f.write("\n-->>sys :  low__Result ~ H:{0:3d}, S:{1:3d}, V:{2:3d}".format(low_h, low_s, low_v))
        f.write("\n-->>sys :  high_Result ~ H:{0:3d}, S:{1:3d}, V:{2:3d}".format(high_h, high_s, high_v))
        f.close()

        cv2.destroyAllWindows()
        break

#####################
# img_temp01 = cv2.imread(
#     'C:/Users/user/Desktop/AI_Project/ball_calib_python/20200508_LG_labeling_00/org/img_20200508-17-42-55.png')
# cv2.imshow("test_img01", img_temp01)
# cv2.waitKey(2)
#
# img_temp02 = cv2.imread(
#     'C:/Users/user/Desktop/AI_Project/ball_calib_python/20200508_LG_labeling_00/blue_cup/img_20200508-17-44-33.png')
# cv2.imshow("test_img02", img_temp02)
# cv2.waitKey(2)
#
# img_temp03 = np.array(img_temp02) - np.array(img_temp01)
# cv2.imshow("test_img03", img_temp03)
# cv2.waitKey(2)

exit()
