import pyrealsense2 as rs
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import os

class realsense:
    def __init__(self, img_size = [848,480], frame=30):
        # Configure depth and color streams
        self.pipeline1 = rs.pipeline()
        self.config1 = rs.config()
        # self.config1.enable_device('013222072876')
        self.config1.enable_device('013222070199')
        self.config1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, frame)
        self.config1.enable_stream(rs.stream.color, img_size[0], img_size[1], rs.format.bgr8, frame)

        # Start streaming
        self.pipeline1.start(self.config1)

        # Wait for a coherent pair of frames: depth and color
        self.pipeline1.wait_for_frames()
        time.sleep(0.1)

    def get_img(self, out_type):

        frame = self.pipeline1.wait_for_frames()

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

if __name__ == '__main__':
    rs = realsense()
    img_list = rs.get_img_from([1,3])

    images = np.hstack((img_list[0], img_list[1]))
    cv2.imshow('img',images)
    pass