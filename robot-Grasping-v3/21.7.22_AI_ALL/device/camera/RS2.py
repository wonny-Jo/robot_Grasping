import pyrealsense2 as rs
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import os
from config import M_k2b

class realsense:
    def __init__(self, img_size = [1280,720], frame=30):
        # Configure depth and color streams
        self.rs=rs
        self.pipeline = rs.pipeline()
        self.config1 = rs.config()
        # self.config1.enable_device('013222072876')
        self.config1.enable_device('013222070199')
        self.config1.enable_stream(rs.stream.depth, img_size[0], img_size[1], rs.format.z16, frame)
        self.config1.enable_stream(rs.stream.color, img_size[0], img_size[1], rs.format.bgr8, frame)
        # Start streaming
        self.pipeline.start(self.config1)
        self.frames = self.pipeline.wait_for_frames()

        # Camera burn in.
        for _ in range(30):
            self.frames = self.pipeline.wait_for_frames()
            _ = self.frames.get_color_frame()
        self.color_frame = self.frames.get_color_frame()
        self.align = rs.align(rs.stream.color)
        self.frameset = self.align.process(self.frames)
        self.depth_frame = self.frameset.get_depth_frame()
        self.depth_intrin = self.depth_frame.profile.as_video_stream_profile().intrinsics
        #Wait for a coherent pair of frames: depth and color

    def get_img(self, out_type):
        frame = self.pipeline.wait_for_frames()

        if out_type is "rgb":
            img_frame = frame.get_color_frame()
            output_img = np.array(img_frame.get_data())
            return output_img

        elif out_type is "depth":
            depth_frame = frame.get_depth_frame()
            output_depth = np.array(depth_frame.get_data())
            return output_depth

        elif out_type is "all":
            img_frame = frame.get_color_frame()
            depth_frame = frame.get_depth_frame()
            output_img = np.array(img_frame.get_data())
            output_depth = np.array(depth_frame.get_data())

            # depth_dist = np.zeros([480, 640])
            # for y in range(480):
            #     for x in range(640):
            #         depth_dist[y][x] = depth_frame.get_distance(x, y)
            #
            #
            #
            # output_depth = cv2.applyColorMap(cv2.convertScaleAbs(output_depth, alpha=0.03), cv2.COLORMAP_JET)
            # cv2.imshow('a', output_depth)
            return output_img, output_depth

    def view(self):
        while True:
            t1 = time.time()
            isGetImage = False
            frames_1 = self.pipeline.wait_for_frames()
            frames_2 = self.pipeline2.wait_for_frames()
            frames_3 = self.pipeline3.wait_for_frames()

            while isGetImage == False:
                # depth_frame = frames.get_depth_frame()
                color_frame_1 = frames_1.get_color_frame()
                color_frame_2 = frames_2.get_color_frame()
                color_frame_3 = frames_3.get_color_frame()
                if not color_frame_1 or not color_frame_2 or not color_frame_3:
                    continue
                else:
                    isGetImage = True

            color_image_1 = np.asanyarray(color_frame_1.get_data())
            color_image_2 = np.asanyarray(color_frame_2.get_data())
            color_image_3 = np.asanyarray(color_frame_3.get_data())

            # Stack all images horizontally
            images = np.hstack((color_image_1, color_image_2, color_image_3))
            # Show images from both cameras
            cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
            cv2.imshow('RealSense', images)

            print(time.time()-t1)

            cv2.waitKey(1)

            # Save images and depth maps from both cameras by pressing 's'
            ch = cv2.waitKey(25)

    def pxl2xyz(self, mean_xy_point,depth=0.0):
        xyz_list = []
        # ---- ---- ---- ----
        c_y = round(mean_xy_point[0])
        c_x = round(mean_xy_point[1])

        if depth != 0.0:
            depth_point_1 = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [c_x, c_y], depth)
            camera_y_avg1 = depth_point_1[1]
            camera_x_avg1 = depth_point_1[0]
            camera_z_avg1 = depth_point_1[2]
            cam_pos = np.array([[camera_x_avg1, camera_y_avg1, camera_z_avg1, 1]])
            xyz = np.dot(M_k2b, cam_pos.T)
            xyz = xyz.flatten()[:-1]
            mean_xyz = np.array(xyz)
        else:
            camera_x = []
            camera_y = []
            camera_z = []

            for i in range(c_y - 5, c_y + 5 + 1):
                if i<0:
                    continue
                if i>=720:
                    break
                for j in range(c_x - 5, c_x + 5 + 1):
                    if j < 0:
                        continue
                    if j >= 1280:
                        break
                    depth_1 = self.depth_frame.get_distance(j, i)
                    if depth_1 == 0.0:
                        continue
                    depth_point_1 = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [j, i], depth_1)
                    camera_y.append(depth_point_1[1])
                    camera_x.append(depth_point_1[0])
                    camera_z.append(depth_point_1[2])

            if 0 < camera_y.__len__() and 0 < camera_x.__len__() and 0 < camera_z.__len__():
                camera_y_avg1 = sum(camera_y) / camera_y.__len__()
                camera_x_avg1 = sum(camera_x) / camera_x.__len__()
                camera_z_avg1 = sum(camera_z) / camera_z.__len__()

        # camera_x2 = []
        # camera_y2 = []
        # camera_z2 = []
        # for k in range(c_y - 3, c_y + 3 + 1):
        #     for l in range(c_x - 3, c_x + 3 + 1):
        #         depth_2 = self.depth_frame.get_distance(l, k)
        #         if depth_2 == 0.0:
        #             continue
        #         depth_point_2 = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [l, k], depth_2)
        #         if ((camera_y_avg1 - abs(camera_y_avg1 / 10)) < depth_point_2[1] < (
        #                 camera_y_avg1 + abs(camera_y_avg1 / 10))) and \
        #                 ((camera_x_avg1 - abs(camera_x_avg1 / 10)) < depth_point_2[0] < (
        #                         camera_x_avg1 + abs(camera_x_avg1 / 10))) and \
        #                 ((camera_z_avg1 - abs(camera_z_avg1 / 10)) < depth_point_2[2] < (
        #                         camera_z_avg1 + abs(camera_z_avg1 / 10))):
        #             camera_y2.append(depth_point_2[1])
        #             camera_x2.append(depth_point_2[0])
        #             camera_z2.append(depth_point_2[2])
        # if camera_y2.__len__() > 0 and camera_x2.__len__() > 0 and camera_z2.__len__() > 0:
        #     camera_y_avg2 = sum(camera_y2) / camera_y2.__len__()  # -#- 오류
        #     camera_x_avg2 = sum(camera_x2) / camera_x2.__len__()
        #     camera_z_avg2 = sum(camera_z2) / camera_z2.__len__()

                # cam_pos = np.array([[camera_x_avg2, camera_y_avg2, camera_z_avg2, 1]])  # -#-
                cam_pos = np.array([[camera_x_avg1, camera_y_avg1, camera_z_avg1, 1]])
                xyz = np.dot(M_k2b, cam_pos.T)
                xyz = xyz.flatten()[:-1]
                xyz_list.append(xyz)

                xyz_list = np.array(xyz_list)

                while np.any(np.isinf(xyz_list)) or np.any(np.isnan(xyz_list)):
                    nan_idx = np.sort(np.transpose(np.argwhere(np.isnan(xyz_list))[0::3])[0])
                    for x in reversed(nan_idx):
                        xyz_list = np.delete(xyz_list, x, 0)

                mean_xyz = np.mean(xyz_list, axis=0)
            else:
                depth_img = np.array(self.depth_frame.get_data())
                cv2.imshow("view_depth", depth_img)
                cv2.waitKey(1)
                return None

        if np.any(np.isinf(mean_xyz)) or np.any(np.isnan(mean_xyz)):
            return None
        else:
            return mean_xyz

    def path2xyz(self, path):

        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

        pxl_patch = path

        # Seg 이미지에서의 경로를 카메라 기준 경로로 바꿔주는 작업 - 수정 필요
        for idx, [y, x] in enumerate(pxl_patch):
            pxl_patch[idx][1] = x * (916 / 360) + 187
            pxl_patch[idx][0] = y * (916 / 360) - (916 - 720) / 2

        pxl_patch = np.array(pxl_patch).astype(np.uint32)  # round? int?

        xyz_list = []
        camera_x = []
        camera_y = []
        camera_z = []
        for c_y, c_x in pxl_patch:
            depth = depth_frame.get_distance(c_x, c_y)
            if depth == 0.0:
                continue
            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [c_x, c_y], depth)
            camera_y.append(depth_point[0])
            camera_x.append(depth_point[1])
            camera_z.append(depth_point[2])

        camera_y_avg = sum(camera_y) / camera_y.__len__()
        camera_x_avg = sum(camera_x) / camera_x.__len__()
        camera_z_avg = sum(camera_z) / camera_z.__len__()

        cam_pos = np.array([[camera_y_avg, camera_x_avg, camera_z_avg, 1]])
        xyz = np.dot(M_k2b, cam_pos.T)
        xyz_list.append(xyz)

        xyz_list = np.array(xyz_list)

        while np.any(np.isinf(xyz_list)) or np.any(np.isnan(xyz_list)):
            nan_idx = np.sort(np.transpose(np.argwhere(np.isnan(xyz_list))[0::3])[0])
            for x in reversed(nan_idx):
                xyz_list = np.delete(xyz_list, x, 0)

        if np.any(np.isinf(xyz_list)) or np.any(np.isnan(xyz_list)):
            return None
        else:
            return xyz_list


if __name__ == '__main__':
    rs = realsense()
    img_list = rs.get_img_from([1,3])

    images = np.hstack((img_list[0], img_list[1]))
    cv2.imshow('img',images)
    pass