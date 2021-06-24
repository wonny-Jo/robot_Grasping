#from torch.lib import *
from func_process import *
from device.thread import *
from config import *
import cv2, os, pickle

global a

ur = UR("192.168.0.52")
ur2 = UR("192.168.0.29")

def target_task(target_list):
    if target_list[0] == 'HDMI_HUB' or target_list[0] == 'USB_HUB':
        hub_cable_task(ur,ur2,target_list)
    elif target_list[0] == 'ORANGE_PENCIL' or target_list[0] == 'BLUE_PENCIL':
        pencil_task(ur,ur2,target_list)
    elif target_list[0] == 'WHITE_PLUG' or target_list[0] == 'BLACK_PLUG':
        plug_multitap_task(ur,ur2,target_list)



if __name__ == '__main__':

    set_thread()
    start_thread()
    ur.rob.set_tcp([0, 0, 0.153, 0, 0, 0])
    acc = 0.5
    min_t = 0.1
    v_w = 0.2


    # 초기 위치 이동
    ur.setGripper(0)    # 0이 오픈 / 255가 클로즈
    ur2.setGripper(0)
    ur2.ur_rob.movej(pose_base, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur2, pose_base, 0.001)
    ur.ur_rob.movej(pose_start, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur, pose_start, 0.001)


    target_lists = [['HDMI_HUB','HDMI'],
                    ['USB_HUB','USB_C'],
                    ['ORANGE_PENCIL','ORANGE_SHARPENER','SILVER_PENCILCASE'],
                    ['BLUE_PENCIL','BLUE_SHARPENER','SILVER_PENCILCASE'],
                    ['WHITE_PLUG','BLACK_MULTITAP'],
                    ]

    ###test용
    target_task(['ORANGE_PENCIL','ORANGE_SHARPENER','SILVER_PENCILCASE'])
    #target_task(['WHITE_PLUG','BLACK_MULTITAP'])
    ####

    # for target_list in target_lists:
    #     if isDetected(target_list)==True:
    #         target_task(target_list)

