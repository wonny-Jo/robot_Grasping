from numpy import linalg as la

import numpy as np
import random
import math3d
import math
import time
# import urx
import socket

from . import urx_

def get_inserting_control_params():
    control_params = {}
    control_params.update({"mode": "base"})
    control_params.update({"enable_axis": [0, 0, 1, 0, 0, 0]})
    control_params.update({"target_axis": [0, 0, 0, 0, 0, 0]})
    control_params.update({"limit_axis": [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]})
    control_params.update({"spd_acc": 0.25, "spd_time": 0.205})
    control_params.update({"step_time": 0.05})

    control_params.update({"power_limit_ft": np.array([20.0, 20.0, 30.0, 2.0, 2.0, 2.0])})
    control_params.update({"hdmi_limit_ft": np.array([20.0, 20.0, 30.0, 2.5, 2.5, 2.5])})

    return control_params

def fm_rob_control(ur, z_force, fs_input, control_params):
    cmd_header = "def myProg():\n"
    cmd_mode = control_params["mode"]

    if cmd_mode is "base":
        cmd_base = "force_mode(p[0,0,0,0,0,0]"
    elif cmd_mode is "tool":
        cmd_base = "force_mode(get_actual_tcp_pose()"

    enable_axis = control_params["enable_axis"]
    target_axis = control_params["target_axis"]
    limit_axis = control_params["limit_axis"]
    spd_time = control_params["spd_time"]

    cmd_ft = ", [" \
             + str(enable_axis[0]) + "," + str(enable_axis[1]) + "," + str(enable_axis[2]) + "," \
             + str(enable_axis[3]) + "," + str(enable_axis[4]) + "," + str(enable_axis[5]) + "], [" \
             + str(target_axis[0]) + "," + str(target_axis[1]) + "," + str(z_force) + "," \
             + str(target_axis[3]) + "," + str(target_axis[4]) + "," + str(target_axis[5]) + "], " \
             + str(2) + ", [" \
             + str(limit_axis[0]) + "," + str(limit_axis[1]) + "," + str(limit_axis[2]) + "," \
             + str(limit_axis[3]) + "," + str(limit_axis[4]) + "," + str(limit_axis[5]) + "])\n"

    cmd_sync = "sync()\n"

    cmd_speedl = "movel(p[" \
                 + str(fs_input[0]) + "," + str(fs_input[1]) + "," + str(fs_input[2]) + "," \
                 + str(fs_input[3]) + "," + str(fs_input[4]) + "," + str(fs_input[5]) + "]," \
                 + str(0.03) + "," + str(0.03) + "," + str(0.3) + "," + str(0) + ")\n"

    cmd_sleep = "sleep(" + str(0.3) + ")\n"

    cmd_end = "end\n"

    prog = cmd_header + cmd_base + cmd_ft + cmd_sync + cmd_speedl + cmd_sleep + cmd_end
    ur.ur_rob.send_program(prog)

def waiting_target(ur3_obj, target_pose, threshold):
    target_dis = 1000
    t = time.time()
    while abs(target_dis) >= threshold:
        current_pose  = ur3_obj.ur_rob.rtmon.get_all_data()['tcp']
        target_dis = np.linalg.norm(target_pose - current_pose)
        if time.time() - t > 15 :
            break

def waiting_joint(ur3_obj, target_pose, threshold):
    target_dis = 1000
    t = time.time()
    while abs(target_dis) >= threshold:
        current_pose = np.array(ur3_obj.ur_rob.rtmon.get_all_data()['qActual'])
        target_dis = np.linalg.norm(target_pose - current_pose)
        if time.time() - t > 15 :
            break

def wait_gripper(grp, target, min_t):

    request = "GET " + "OBJ"
    grp.sendall(request.encode())
    data = {}
    data.update({"OBJ": int(grp.recv(1024))})


def target_distance(ur3_obj, target_pose):
    current_pose = np.array(ur3_obj.ur_rob.getl())
    target_dis = np.linalg.norm(target_pose[0:3] - current_pose[0:3])

    t1 = math3d.Transform(np.array(target_pose[3:6]), np.array([0, 0, 0]))
    r1 = t1.orient.array

    t2 = math3d.Transform(np.array(current_pose[3:6]), np.array([0, 0, 0]))
    r2 = t2.orient.array

    target_r = np.linalg.norm(r1 - r2)

    target_dis = target_dis + target_r

    return target_dis

def target_distance2(ur3_obj, target_pose):
    current_pose = np.array(ur3_obj.ur_rob.getl())
    target_dis = np.linalg.norm(target_pose[0:2] - current_pose[0:2])

    t1 = math3d.Transform(np.array(target_pose[3:6]), np.array([0, 0, 0]))
    r1 = t1.orient.array

    t2 = math3d.Transform(np.array(current_pose[3:6]), np.array([0, 0, 0]))
    r2 = t2.orient.array

    target_r = np.linalg.norm(r1 - r2)

    # target_dis = target_dis + target_r

    return target_dis , target_r


class UR:
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip
        self.ur_rob = urx_.URRobot(robot_ip, True)
        self.rob = urx_.Robot(robot_ip)
        self.grp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.grp.connect((robot_ip, 63352))
        self.random_area = []
        self.object_location = []

    def set_random_area(self, pose, pose_range):
        self.random_area = []
        for i in range(3):
            temp = [pose[i] - pose_range[i], pose[i] + pose_range[i]]
            self.random_area.append(temp)
        for i in range(3, 6):
            temp = [-pose_range[i], pose_range[i]]
            self.random_area.append(temp)

    def get_random_pose(self, cur_pose):
        random_p = []

        for i in range(6):
            tmp_val = random.uniform(self.random_area[i][0], self.random_area[i][1])
            random_p.append(tmp_val)

        tar_pose = self.get_rotate_pose(cur_pose, random_p[3:6])
        tar_pose[0:2] = random_p[0:2]
        return np.array(tar_pose)

    def get_rotate_pose(self, pose_mat, tar_rot):
        tar_pose_mat = math3d.Transform(np.array(tar_rot), np.array([0, 0, 0]))
        tar_mat = pose_mat * tar_pose_mat
        tar_vec = tar_mat.pose_vector

        return tar_vec


    def setGripper(self, val):
        g_val = 'def myProg():\n' + \
                'socket_close("gripper_socket")\n' + \
                'socket_open("' + str(self.robot_ip) + '",63352,"gripper_socket")\n' + \
                'socket_set_var("POS",' + str(val) + ',"gripper_socket")\n' + \
                'sync()\n' + \
                'sleep(0.1)\n' + \
                'end\n'


        self.ur_rob.send_program(g_val)

        if val == 255:
            t1 = time.time()
            while True:
                request = "GET " + "OBJ"
                self.grp.sendall(request.encode())
                data = {}
                data.update({"OBJ": int(self.grp.recv(1024))})

                if data['OBJ'] == 2 :
                    break
                elif time.time() - t1 > 1:
                    break
        elif val == 0:
            t1 = time.time()
            while True:
                request = "GET " + "POS"
                self.grp.sendall(request.encode())
                data = {}
                data.update({"POS": int(self.grp.recv(1024))})

                if data['POS'] < 10 :
                    break
                elif time.time() - t1 > 1:
                    break