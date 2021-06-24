import argparse

from Robot_env import robot_env
# from segmentation import segmentation_graph
from object_detection import Seg_detector
from Robot_env.config import RL_Obj_List
import random
import copy

import logging
import sys
import numpy as np
from Robot_env.scattering_easy import get_distance

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--use_scatter', type=bool, default=True, help="use scattering")
parser.add_argument('--num_scattering', type=int, default=2, help="the number of scattering")
parser.add_argument('--seg_path', type=str, default="./segmentation/checkpoint/", help="segmentation checkpoint path")  # #--# 교체 예정
parser.add_argument('--detector_path', type=str, default="./object_detection/checkpoint/", help="object_detection checkpoint path")
parser.add_argument('--seg_threshold', type=float, default=0.60, help="segmentation threshold")

args = parser.parse_args()

socket_ip1 = "192.168.0.52"  # 오른쪽 팔(카메라)
socket_ip2 = "192.168.0.29"  # 왼쪽 팔

logger = logging.getLogger("Agent")

class Agent:
    def __init__(self, rob):
        self.robot = rob
        self.obj_list = [i for i in range(9, 13)]
        self.obj_list += [i for i in range(15, 29)]
        self.obj_list += [i for i in range(31, 34)]
        self.obj_list += [37, 41]
        self.picking_obj_list = [i for i in range(9, 13)]
        self.picking_obj_list += [i for i in range(21, 27)]
        self.drawer_list = [1, 2]       # : drawer
        self.drawer_obj_list = [17, 18, 19, 20]
        self.bin_list = [3, 4]          # : bin
        self.bin_obj_list = [15, 16]
        self.bottle_lid_list = [38, 39]
        self.pen_lid_list = [31, 32]
        self.holder_list = [5, 6]                       # : 5:green     6:black
        self.pen_list = [27, 28]                        # : 27:namepen  28:marker
        self.wide_object_list = [7, 8, 34, 35, 40]      # : 7:black     8:pink
        self.cleaner_list = [37, 41]

        self.shuffled_list = []

    def set_obj(self, org_list):
        shuffled_list = copy.deepcopy(org_list)
        random.shuffle(shuffled_list)
        return shuffled_list

    def run(self):
        rob = self.robot
        episode_num = 1
        rob.rob1.getl()
        rob.reset()
        hasFind = True
        # ---- ---- ---- ---- Picking ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
        logger.info("STARTING PICKING TEST")

        # 스캐터링 없는 버젼
        picking_list = self.set_obj(self.picking_obj_list)
        for target_cls in picking_list:
            if hasFind is True:
                rob.env_img_update()
            target_xyz, target_imgmean, target_pxl = rob.get_obj_pos(target_cls)
            if target_xyz is None:
                hasFind = False
                logger.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                continue
            hasFind = True
            logger.info("Current Target: {}".format(RL_Obj_List[target_cls][0]))
            # self.robot.seg_model.emphasize_target(self.robot.color_seg_img, target_cls)
            rob.grasp_placing_box(target_cls, target_imgmean, target_xyz)


        # #스캐터링과 같이 동작하는 버젼 - 작업중
        # obj_list = self.set_obj(self.obj_list)
        # for target_cls in obj_list:
        #     # tmp
        #     rob.env_img_update()
        #     # while 1:
        #     #     if hasFind is True:
        #     #         rob.env_img_update()
        #     #     target_xyz, target_imgmean, target_pxl = rob.get_obj_pos(target_cls)
        #     #     if target_xyz is None:
        #     #         hasFind=False
        #     #         logger.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
        #     #         break
        #     #     hasFind=True
        #         # distance_array = get_distance(self.robot.color_seg_img, self.robot.detected_obj_list)
        #         # check = False
        #         # for i in self.robot.detected_obj_list:
        #         #     if distance_array[target_cls][i] < 9:
        #         #         logger.info("Scattering Target: {}, {}".format(RL_Obj_List[target_cls][0], RL_Obj_List[i][0]))
        #         #         target2_xyz, _, _ = rob.get_obj_pos(i)
        #         #         rob.scatter_move_gripper(target_xyz, target2_xyz)
        #         #         check = True
        #         #         break
        #         # if check is False:
        #         #     break
        #     if target_cls in self.picking_obj_list:
        #         logger.info("Current Target: {}".format(RL_Obj_List[target_cls][0]))
        #         self.robot.seg_model.emphasize_target(self.robot.color_seg_img, target_cls)
        #         print(1)
        #         # rob.grasp_placing_box(target_cls, target_imgmean, target_xyz)


        #  ---- ---- ---- ---- drawer ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
        logger.info("STARTING DRAWER TEST")
        drawer_xyz = None
        drawer_list = self.set_obj(self.drawer_list)
        for target_cls in drawer_list:
            if hasFind is True:
                rob.env_img_update()
            target_xyz, target_imgmean, target_pxl = self.robot.get_obj_pos(target_cls)
            if target_xyz is None:
                hasFind = False
                logger.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                continue
            hasFind = True
            logger.info("Current Target: {}".format(RL_Obj_List[target_cls][0]))
            drawer_xyz = target_xyz
            break
        if drawer_xyz is not None:
            obj_list = self.set_obj(self.drawer_obj_list)
            for target_cls in obj_list:
                if hasFind is True:
                   rob.env_img_update()
                target_xyz, target_imgmean, target_pxl = rob.get_obj_pos(target_cls)
                if target_xyz is None:
                    hasFind = False
                    logger.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                    continue
                hasFind = True
                logger.info("Current Target: {}".format(RL_Obj_List[target_cls][0]))
                # emphasize_target(target_cls)
                rob.grasp_placing_drawer(target_cls, target_imgmean, target_xyz)
                rob.open_drawer(drawer_xyz)
                rob.grasp_place_drawer_obj(drawer_xyz)
                rob.close_drawer(drawer_xyz)

        # ---- ---- ---- ---- bin ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
        logger.info("STARTING BIN TEST")
        bin_list = self.set_obj(self.bin_list)
        for bin_cls in bin_list:
            if hasFind is True:
                rob.env_img_update()
            bin_xyz, bin_imgmean, bin_pxl = self.robot.get_obj_pos(bin_cls)
            if bin_xyz is None:
                hasFind = False
                logger.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                continue

            hasFind = True
            obj_list = self.set_obj(self.bin_obj_list)
            for target_cls in obj_list:
                if hasFind is True:
                    rob.env_img_update()

                target_xyz, target_imgmean, target_pxl = rob.get_obj_pos(target_cls)
                if target_xyz is None:
                    hasFind = False
                    logger.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                    continue

                hasFind = True
                logger.info("Current Target: {}".format(RL_Obj_List[target_cls][0]))
                rob.grasp_placing_bin(target_cls, target_imgmean, target_xyz, bin_xyz)

         # ---- ---- ---- ---- bottle lid opening ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
        logger.info("STARTING BOTTLE LID OPEN TEST")
        bottle_lid_list = self.set_obj(self.bottle_lid_list)
        for target_cls in bottle_lid_list:
            if hasFind is True:
                rob.env_img_update()
            target_xyz, mean_xy, target_pxl = rob.get_obj_pos(target_cls)
            if target_xyz is None:
                hasFind = False
                logger.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                continue
            hasFind = True
            rob.grasp_open_bottle_lid(target_cls, target_imgmean, target_xyz)

        # ---- ---- ---- ---- Pen lid opening ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
        logger.info("STARTING PEN LID OPEN TEST")
        pen_lid_list = self.set_obj(self.pen_lid_list)
        for target_cls in pen_lid_list:
            if hasFind is True:
                rob.env_img_update()
            target_xyz, mean_xy, target_pxl = rob.get_obj_pos(target_cls)
            if target_xyz is None:
                hasFind = False
                logger.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                continue
            hasFind = True
            rob.grasp_open_pen_lid(target_cls, target_imgmean, target_xyz)

        # ---- ---- ---- ---- Pen ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
        logger.info("STARTING PENHOLDER TEST")
        holder_list = self.set_obj(self.holder_list)
        h_loc = None
        for target_cls in holder_list:
            if hasFind is True:
                rob.env_img_update()
            target_xyz, target_imgmean, target_pxl = rob.get_obj_pos(target_cls)
            if target_xyz is None:
                hasFind = False
                logger.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                continue
            hasFind = True
            logger.info("Current Target: {}".format(RL_Obj_List[target_cls][0]))
            h_loc = rob.grasp_holder(target_cls, target_xyz)
            break
        if h_loc is not None:
            pen_list = self.set_obj(self.pen_list)
            for target_cls in pen_list:
                if hasFind is True:
                   rob.env_img_update()
                target_xyz, _, target_pxl = rob.get_obj_pos(target_cls)
                if target_xyz is None:
                    hasFind = False
                    logger.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                    continue
                hasFind = True
                logger.info("Current Target: {}".format(RL_Obj_List[target_cls][0]))
                rob.grasp_pen(target_cls, target_xyz)
                rob.placing_toholder(h_loc)
            rob.holder_toplace(h_loc)

        # ---- ---- ---- ---- wide object ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
        logger.info("STARTING KEYBOARD TEST")
        wide_object_list = self.set_obj(self.wide_object_list)
        for target_cls in wide_object_list:
            if hasFind is True:
                rob.env_img_update()
            target_xyz, mean_xy, target_pxl = rob.get_obj_pos(target_cls)
            if target_xyz is None:
                hasFind = False
                logger.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                continue
            hasFind = True
            rob.grasp_placing_keyboard(target_cls, mean_xy)

        # ---- ---- ---- ---- desk cleaner ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
        logger.info("STARTING DESK CLEANING TEST")
        hasFind = True
        cleaner_list = self.set_obj(self.cleaner_list)
        for target_cls in cleaner_list:
            if hasFind is True:
                rob.env_img_update()
            target_xyz, target_imgmean, target_pxl = rob.get_obj_pos(target_cls)
            if target_xyz is None:
                hasFind = False
                logger.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                continue
            hasFind = True
            logger.info("Current Target: {}".format(RL_Obj_List[target_cls][0]))
            rob.grasp_moving_cleaner(target_cls, target_imgmean, target_xyz)



if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    segmentation_model = Seg_detector.Segment()
    robot = robot_env.Robot(socket_ip1, socket_ip2, segmentation_model, args.seg_threshold)

    agent = Agent(robot)
    agent.run()
