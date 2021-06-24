import urx
import time
import os
import numpy as np
import cv2
# from dateutil.parser import private_class
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
import pyrealsense2 as rs

import Robot_env.robot_util as ru

####  Constants  ####
# >> Result : H : 118, S : 75, V : 205
# Result : H : 115, S : 143, V : 101 # 20191025
# Result : H : 115, S : 110, V : 110
hsv_dir = "./20200810"
hsv_name = "hsv_20200810.txt"  # : 저장 파일 이름
hsv_path = os.path.join(hsv_dir, hsv_name)
f_hsv = open(hsv_path, "r")
line = []
i = 0
while True:
	text = f_hsv.readline()
	data_line = text.strip('\n')
	# data_line = text.splitlines()
	if text.__len__() == 0:
		break
	else:
		line.append(data_line)
		print("contents : " + data_line)
		i += 1

# -->>sys :  low__Result : H : 120, S : 200, V : 220
# -->>sys :  high_Result : H : 144, S : 240, V : 255
l_h = int(line[-2][-17:-14])
l_s = int(line[-2][-10:-7])
l_v = int(line[-2][-3:])

h_h = int(line[-1][-17:-14])
h_s = int(line[-1][-10:-7])
h_v = int(line[-1][-3:])
#####################

# : 2 : 각 로봇의 포즈 저장
# : 다음 3 : main 실행

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipe_profile = pipeline.start(config)

# : RL robot
rob1 = ru.Robot_util("192.168.0.2")    # : Cam
rob2 = ru.Robot_util("192.168.0.29")    # : gripper
rob1_home_joint_rad = np.deg2rad([0.3209, -113.0970, -4.5383, -152.3580, 89.6613, 1.2152])  # : Cam Pose
rob2_home_joint_rad_b = np.deg2rad([40.9664, -74.1802, 117.9032, -112.9013, 247.8042, -224.6624 + 180])  # : Tray Pose - big
rob2_home_joint_rad_s = [0.7150, -1.29469, 2.0578, -1.9705, 4.3250, -3.9211]  # : Tray Pose(rad) - small

rob1.set_tcp([0, 0, 0.153, 0, 0, 0])
rob2.set_tcp([0, 0, 0.170, 0, 0, 0])

rob1.movej(rob1_home_joint_rad, 0.5, 0.5)
rob2.movej(rob2_home_joint_rad_s, 0.5, 0.5)

#center = np.deg2rad([-35.45, -62.40, 117.18, -0.87, 67.47, -207.88])
#rob2.movej(center, 0.5, 0.5)

joint_dir = hsv_dir
file_name = "RL_joint_list_20200810_00.txt"  # : 저장 파일 이름
file_path = os.path.join(joint_dir, file_name)


def automatic_move():
	dir = hsv_dir
	joint_name = "RL_joint_list_20200714_00.txt"  # : 불러올 이전 joint 파일 이름
	joint_path = os.path.join(dir, joint_name)
	f_joint = open(joint_path, "r")
	init = 0
	for _ in range(init):
		f_joint.readline()
	for idx, line in enumerate(f_joint):
		k = line.split(' ')
		j_pt = [float(j) for j in k]
		rob2.movej(j_pt, 0.8, 0.8)



	cv2.waitKey(100)


	rob2.movej(center, 1, 1)
	time.sleep(5)
	rob2.movel(left_front, 1, 1)
	time.sleep(5)
	rob2.movel(left_back, 1, 1)
	time.sleep(5)
	rob2.movel(right_front, 1, 1)
	time.sleep(5)
	rob2.movel(right_back, 1, 1)

def main():
	f = open(file_path, "a")

	line_cnt = 0
	cx = 0
	cy = 0

	while True:

		is_free = False

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
		# cv2.imwrite('C:/Users/user/Desktop/AI_Project/ball_calib_python/_test/{}.png'.format(curr_frame), img)
		# curr_frame += 1

		# Normal masking algorithm
		lower_filter = np.array([l_h, l_s, l_v])
		upper_filter = np.array([h_h, h_s, h_v])

		hsv = np.copy(img)
		hsv = cv2.cvtColor(hsv, cv2.COLOR_RGB2HSV)
		mask = cv2.inRange(hsv, lower_filter, upper_filter)
		result = cv2.bitwise_and(img, img, mask=mask)
		# : ---- 위는 모두 마스킹

		ret, thresh = cv2.threshold(result, 16, 255, cv2.THRESH_BINARY)  # : 스레스홀드 127로 설정, 최대 255
		blurred = cv2.medianBlur(thresh, 5)  # : 메디안 필터를 이용한 블러
		blurred = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)  # : 그레이스케일로 변환
		# th3 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) # : 가우시안, 어뎁티브 스레스홀드
		_, th3 = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)

		# _, contours, hierarchy = cv2.findContours(th3, 1, 2)  # : opencv 3
		contours, hierarchy = cv2.findContours(th3, 1, 2)  # : opencv 4
		cv2.drawContours(result, contours, -1, (0, 255, 0), 1)

		# radius = 0
		max_radius = 0
		for cnt in contours:
			if cv2.contourArea(cnt) < 60000:
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
		except:
			print("-->>hsv : Can't Find the ball")

		cv2.moveWindow('Color filter', 1920 - int(1280 / 2) - 1, 320)
		cv2.imshow('Color filter', cv2.resize(result, (int(1280 / 2), int(720 / 2))))
		cv2.moveWindow('origin', 1920 - int(1280 / 2) - 1, 680)
		cv2.imshow('origin', cv2.resize(img, (int(1280 / 2), int(720 / 2))))

		key = cv2.waitKey(33)

		if key == ord('f'):  # : 프리드라이브
			rob2.set_freedrive(not is_free, 3600)  # 3600 sec.
			is_free = not is_free
			if is_free:
				print("..>> Free Drive Mode : ON !!")
			else:
				print("..>> Free Drive Mode : OFF !!")
		# modified by @hipiphock
		# automatic move
		elif key == ord('m'):
			automatic_move()

		if key == 115:  # : S   # : 저장
			if 6 < max_radius:
				curj = np.round(rob2.getj(), 4)
				f.write("{} {} {} {} {} {}\n".format(*curj))
				print("-->> Saved joint data . {}".format(line_cnt))
				line_cnt += 1
			else:
				print("..>> Not Saved joint data in {} ---- ---- ".format(line_cnt))

		if key == 113:  # : Q   # : 종료
			print("-->> Saved all.")
			cv2.destroyAllWindows()
			break

	f.close()

	print("..>> End collect  Total : {}".format(line_cnt - 1))
	exit()


if __name__ == "__main__":
	main()