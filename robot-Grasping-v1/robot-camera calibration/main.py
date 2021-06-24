import pyrealsense2 as rs
import os
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
import cv2
import urx
import numpy as np
import time
import collecting_position
import math

import Robot_env.robot_util as ru

# : RL robot
rob1 = ru.Robot_util("192.168.0.52")    # : Cam
rob2 = ru.Robot_util("192.168.0.29")    # : gripper
rob1_home_joint_rad = np.deg2rad([0.3209, -113.0970, -4.5383, -152.3580, 89.6613, 1.2152])  # : Cam Pose
rob2_home_joint_rad_b = np.deg2rad([40.9664, -74.1802, 117.9032, -112.9013, 247.8042, -224.6624 + 180])  # : Tray Pose - big
rob2_home_joint_rad_s = [0.7150, -1.29469, 2.0578, -1.9705, 4.3250, -3.9211]  # : Tray Pose(rad) - small
home = np.deg2rad([0.0, -90.0, 0.0, -90.0, 0.0, 0.0])
rob1.set_tcp([0, 0, 0.153, 0, 0, 0])
rob2.set_tcp([0, 0, 0.170, 0, 0, 0])

rob2.movej(home, 0.5, 0.5)

rob1.movej(rob1_home_joint_rad, 0.5, 0.5)
rob2.movej(rob2_home_joint_rad_s, 0.5, 0.5)

hsv_dir = "./20200813"
hsv_name = "hsv_20200813.txt"  # : hsv 필터 정보가 저장된 파일 이름
hsv_path = os.path.join(hsv_dir, hsv_name)
f_hsv = open(hsv_path, "r")
file_name = "RL_joint_list_20200813_01.txt"  # : 조인트가 저장된 파일 이름
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
#
# ############  Lower HSV  ############
# # >> Result : H : 118, S : 90, V : 175
# # Result : H : 115, S : 143, V : 101 # 20191025
# # Result : H : 115, S : 110, V : 110
#
# # -->>sys :  low__Result : H : 120, S : 200, V : 220
# # -->>sys :  high_Result : H : 144, S : 240, V : 255
# l_h = 120
# l_s = 200
# l_v = 220
#
# h_h = 144
# h_s = 240
# h_v = 255
# ####################################

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
pipe_profile = pipeline.start(config)


# : 3 : 포즈를 기반으로 bin파일 생성
# : 다음 4 : Kinect3DCalib. cpp프로젝트 실행
def get_cam_img():
	frames = pipeline.wait_for_frames()
	color_frame = frames.get_color_frame()

	align = rs.align(rs.stream.color)
	frameset = align.process(frames)

	aligned_depth_frame = frameset.get_depth_frame()
	# depth_frame = frames.get_depth_frame()

	# Intrinsics & Extrinsics
	depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
	# depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
	color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
	# color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
	depth_to_color_extrin = aligned_depth_frame.profile.get_extrinsics_to(color_frame.profile)
	# depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)

	# Convert images to numpy arrays
	# depth_image = np.asanyarray(depth_frame.get_data())
	depth_image = np.asanyarray(aligned_depth_frame.get_data())
	color_image = np.asanyarray(color_frame.get_data())

	return aligned_depth_frame, color_frame, depth_intrin, color_intrin, depth_image, color_image
	# return depth_frame, color_frame, depth_intrin, color_intrin, depth_image, color_image


def init_cam(fp):
	print("-->>sys : initializing Realsense ......")
	for num in range(0, fp):

		_, _, _, _, depth_image, color_image = get_cam_img()
		test_view = np.copy(color_image)

		hsv = cv2.cvtColor(test_view, cv2.COLOR_RGB2HSV)
		lower_blue = np.array([l_h, l_s, l_v])  #
		upper_blue = np.array([h_h, h_s, h_v])  # FIX
		mask = cv2.inRange(hsv, lower_blue, upper_blue)
		result = cv2.bitwise_and(test_view, color_image, mask=mask)

		ret, thresh = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
		blurred0 = cv2.medianBlur(thresh, 5)
		blurred = cv2.cvtColor(blurred0, cv2.COLOR_RGB2GRAY)
		# th3 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
		_, th3 = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)

		cv2.moveWindow('threshold', 2560 - int(1280 / 2) - 1, 0)
		cv2.imshow('threshold', cv2.resize(th3, (int(1280 / 2), int(720 / 2))))
		cv2.moveWindow('color_image', 2560 - int(1280 / 2) - 1, 390)
		cv2.imshow('color_image', cv2.resize(color_image, (int(1280 / 2), int(720 / 2))))
		cv2.waitKey(2)

	print("-->>sys : Realsense initializing completed.")


def main(dd):
	print("-->>sys : Starting Calibration datacollect ......")
	init_cam(120)

	DataRecord = dd
	print("-->>sys : move robot to HOME pose ......")
	rob2.movej(rob2_home_joint_rad_s, 1, 1)

	# 경로.
	DataRecord.OpenDataFile('a', './{}/joint_list_20200813_RL_00_alignedxyz.bin'.format(hsv_dir))  # w : 새로 작성, a : 덮어쓰기
	path = './{}/{}'.format(hsv_dir, file_name)

	curr_frame = 0

	file = open(path, "r")

	init = 0
	for _ in range(init):
		file.readline()

	for idx, line in enumerate(file):
		try:
			if idx == 319:
				aaaa = 0
			k = line.split(' ')
			j_pt = [float(j) for j in k]
			rob2.movej(j_pt, 0.8, 0.8)

			cv2.waitKey(100)

			endEffector_pos = [round(x, 4) for x in rob2.getl()[0:3]]

			depth_frame, _, depth_intrin, _, depth_image, color_image = get_cam_img()
			g_view = np.copy(color_image)

			# Filter
			hsv = cv2.cvtColor(g_view, cv2.COLOR_RGB2HSV)
			lower_blue = np.array([l_h, l_s, l_v])  #
			upper_blue = np.array([255, 255, 255])  # FIX
			mask = cv2.inRange(hsv, lower_blue, upper_blue)
			result = cv2.bitwise_and(g_view, color_image, mask=mask)

			ret, thresh = cv2.threshold(result, 16, 255, cv2.THRESH_BINARY)
			blurred0 = cv2.medianBlur(thresh, 5)
			blurred = cv2.cvtColor(blurred0, cv2.COLOR_RGB2GRAY)
			# th3 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
			_, th3 = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)

			# _, contours, hierarchy = cv2.findContours(th3, 1, 2)  # : cv2 3
			contours, hierarchy = cv2.findContours(th3, 1, 2)  # : cv2 4
			cv2.drawContours(color_image, contours, -1, (0, 255, 0), 2)

			cv2.moveWindow('threshold', 1920 - int(1280 / 2) - 1, 0)
			cv2.imshow('threshold', cv2.resize(th3, (int(1280 / 2), int(720 / 2))))
			cv2.moveWindow('color_image', 1920 - int(1280 / 2) - 1, 390)
			cv2.imshow('color_image', cv2.resize(color_image, (int(1280 / 2), int(720 / 2))))
			cv2.waitKey(2)

			index = np.argwhere(np.squeeze(th3) == 255)
			h = index[:, 0]
			w = index[:, 1]
			##########
			# _, contours, hierarchy = cv2.findContours(th3, 1, 2)
			#
			# # for i in h:
			# #     for j in w:
			# #         depth = depth_frame.get_distance(j, i)
			# #         depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [j, i], depth)
			# max_radius = 0
			# for cnt in contours:
			# 	if cv2.contourArea(cnt) < 60000:
			# 		(cx, cy), radius = cv2.minEnclosingCircle(cnt)
			#
			# 		if radius > max_radius:
			# 			max_radius = radius
			#
			# c_x = int(np.mean(w))
			# c_y = int(np.mean(h))
			#
			# cv2.circle(g_view, (int(c_x), int(c_y)), int(radius), (0, 0, 255), 2)  # draw circle in red color
			# g_view = cv2.resize(g_view, (int(1280 / 2), int(720 / 2)))
			# cv2.imshow('input', g_view)
			# cv2.waitKey(1)
			###########
			c_x = int(np.mean(w))
			c_y = int(np.mean(h))

			depth0 = depth_frame.get_distance(c_x, c_y)
			depth0_ = depth_image[c_y, c_x]

			depth_point0 = rs.rs2_deproject_pixel_to_point(depth_intrin, [c_x, c_y], depth0)

			# if depth0 == 0:
			# 	continue

			camera_x = []
			camera_y = []
			camera_z = []
			for i in range(c_y - 3, c_y + 3 + 1):
				for j in range(c_x - 3, c_x + 3 + 1):
					depth_1 = depth_frame.get_distance(j, i)
					# if depth_1 == 0:
					# 	continue
					depth_point_1 = rs.rs2_deproject_pixel_to_point(depth_intrin, [j, i], depth_1)
					camera_x.append(depth_point_1[0])
					camera_y.append(depth_point_1[1])
					camera_z.append(depth_point_1[2])

			# camera_y = camera_y / 7.0
			# camera_x = camera_x / 7.0
			# camera_z = camera_z / 7.0
			camera_x_avg1 = sum(camera_x) / camera_x.__len__()
			camera_y_avg1 = sum(camera_y) / camera_y.__len__()
			camera_z_avg1 = sum(camera_z)/camera_z.__len__()

			sorted_y = camera_y.copy()
			sorted_y.sort()
			sorted_y_array = np.array(sorted_y)

			camera_x2 = []
			camera_y2 = []
			camera_z2 = []
			for k in range(c_y - 3, c_y + 3 + 1):
				for l in range(c_x - 3, c_x + 3 + 1):
					depth_2 = depth_frame.get_distance(l, k)
					if depth_2 == 0:
						continue
					depth_point_2 = rs.rs2_deproject_pixel_to_point(depth_intrin, [l, k], depth_2)
					if ((camera_y_avg1 - abs(camera_y_avg1/10)) < depth_point_2[1] < (camera_y_avg1 + abs(camera_y_avg1/10))) and \
							((camera_x_avg1 - abs(camera_x_avg1/10)) < depth_point_2[0] < (camera_x_avg1 + abs(camera_x_avg1/10))) and \
							((camera_z_avg1 - abs(camera_z_avg1/10)) < depth_point_2[2] < (camera_z_avg1 + abs(camera_z_avg1/10))):
							camera_x2.append(depth_point_2[0])
							camera_y2.append(depth_point_2[1])
							camera_z2.append(depth_point_2[2])

			camera_x_avg2 = sum(camera_x2) / camera_x2.__len__()
			camera_y_avg2 = sum(camera_y2) / camera_y2.__len__()
			camera_z_avg2 = sum(camera_z2)/camera_z2.__len__()

			DataRecord.DataInsert([endEffector_pos[0], endEffector_pos[1], endEffector_pos[2]],
			                      [camera_x_avg2, camera_y_avg2, camera_z_avg2])    # -#-
			# [int(camera_y), int(camera_x), camera_z])

			print("----e10")
			#        g_view = cv2.resize(g_view, (int(1920 / 3), int(1080 / 3)))
			#        cv2.imshow('input', g_view)
			#        cv2.waitKey(1)
			print("Current value: ", idx)

			if idx == 750:
				break
		except:
			print('frame passed {}'.format(idx))
			continue

	# rob.movej((0, math.radians(-90), 0, math.radians(-90), 0, 0), 2, 2)
	rob2.movej(rob2_home_joint_rad_s, 1, 1)
	DataRecord.CloseData()

	# 정상적으로 저장 완료 된 상태.
	print('finish')
	rob2.set_tcp([0, 0, 0, 0, 0, 0])
	# rob.close()


if __name__ == "__main__":
	DataRecord = collecting_position.DataRecord()
	main(DataRecord)
