from model.function import *
from device.robot.universal_robot import *
import copy

# robot 주소
ur = UR("192.168.0.52")
ur2 = UR("192.168.0.29")

#model_yolo1 = init_YOLO("./model/yolov4_210810_model1.cfg","./model/yolov4_210810_model1.weights","./model/210810_model1.data") # pack in hole
#model_yolo3 = init_YOLO("./model/yolov4_210810_model3.cfg","./model/yolov4_210810_model3.weights","./model/210810_model3.data") # picking, bottle_lid, wide_object, desk_cleaner
#model_bin = init_YOLO("./model/model_bin.cfg","./model/model_bin.weights","./model/model_bin.data")
#model_drawer = init_YOLO("./model/model_drawer.cfg","./model/model_drawer.weights","./model/model_drawer.data")
#model_penholder = init_YOLO("./model/model_penholder.cfg","./model/model_penholder.weights","./model/model_penholder.data")
#model_hub = init_YOLO("./model/model_hub.cfg","./model/model_hub.weights","./model/model_hub.data")
#model_yolo2 = init_YOLO("./model/yolov4_210810_model2.cfg","./model/yolov4_210810_model2.weights","./model/210810_model2.data") # bin, drawer, pen holder

# model_multitap = init_YOLO("./model/model_multitap.cfg","./model/model_multitap.weights","./model/model_multitap.data")
# model_pencilcase = init_YOLO("./model/model_pencilcase.cfg","./model/model_pencilcase.weights","./model/model_pencilcase.data")
# model_picking = init_YOLO("./model/model_picking.cfg","./model/model_picking.weights","./model/model_picking.data")
# model_cwb = init_YOLO("./model/model_cwb.cfg","./model/model_cwb.weights","./model/model_cwb.data")
# model_bd = init_YOLO("./model/model_bd.cfg","./model/model_bd.weights","./model/model_bd.data")
# model_hp = init_YOLO("./model/model_hp.cfg","./model/model_hp.weights","./model/model_hp.data")
#

model_picking = init_YOLO("./model/model_picking.cfg","./model/model_picking.weights","./model/model_picking.data")
model_picking2 = init_YOLO("./model/model_picking2.cfg","./model/model_picking2.weights","./model/model_picking2.data")
model_mp = init_YOLO("./model/model_mp.cfg","./model/model_mp.weights","./model/model_mp.data")
model_cwb = init_YOLO("./model/model_cwb.cfg","./model/model_cwb.weights","./model/model_cwb.data")
model_bd = init_YOLO("./model/model_bd.cfg","./model/model_bd.weights","./model/model_bd.data")
model_hp = init_YOLO("./model/model_hp.cfg","./model/model_hp.weights","./model/model_hp.data")

##### 고정 위치 좌표들

with open("./data/trajectory_approach", 'rb') as f:
    traj_approch = pickle.load(f)

with open("./data/joint_poses", 'rb') as f:  # 잡기전 대기 자세
    joint_poses=pickle.load(f)

with open("./data/robot_main/pose_cam", 'rb') as f: # 잡기전 대기 자세
    pose_hub_start = pickle.load(f)['qActual']

home = np.deg2rad([0.0, -90.0, 0.0, -90.0, 0.0, 0.0])
pose_start = np.deg2rad([0.3209, -113.0970, -4.5383, -152.3580, 89.6613, 1.2152])
ur_starting_pose = np.deg2rad([-90.0, -80.0, -120.0, -70.0, 90.0, 0])
ur_back_pose = np.deg2rad([0.0, 0.0, -90.0, -90.0, 0.0, 0.0])
ur2_starting_pose = np.deg2rad([90.0, -100.0, 120.0, -110.0, -90.0, 0])
ur2_back_pose = np.deg2rad([0.0, -180.0, 90.0, -90.0, 0.0, 0.0])
ur2_placing_pose = np.deg2rad([90.0, -120.0, 140.0, -110.0, -90.0, 0])
ur2_holder_starting_pose = np.deg2rad([51.90, -96.60, 116.29, -110.19, -89.78, -38.11])
ur_initial_pose = np.deg2rad([-20.0, -110.0, -70.0, -90.0, 90.0, -20.0])
ur2_initial_pose = np.deg2rad([20.0, -70.0, 70.0, -90.0, -90.0, 20.0])


pose_pre_transfer1_cable=np.deg2rad([-4.11638107,-74.88782329,-74.09828074,-121.00496004,89.91389431,86.13635392])
pose_transfer1_cable=np.deg2rad([24.02469692,-71.67361823,-76.14466006,-122.16928872,89.93107907,114.28271654])
pose_USB=np.deg2rad([0.80728055,-80.19643756,126.46291958,-133.73304487,-179.99992669,-180])

# pose_pre_transfer1_cable=joint_poses['pose_pre_transfer1_cable']
# pose_transfer1_cable=joint_poses['pose_transfer1_cable']
pose_pre_transfer2_cable=joint_poses['pose_pre_transfer2_cable']
pose_transfer2_cable=joint_poses['pose_transfer2_cable']
pose_insert_hdmi=joint_poses['pose_insert_hdmi']
pose_insert_usb=joint_poses['pose_insert_usb']
pose_hub_ur=joint_poses['pose_hub_ur']
pose_base=joint_poses['pose_base']
pose_transfer1=joint_poses['pose_transfer1']
pose_transfer2=joint_poses['pose_transfer2']
pose_hub_ur2=joint_poses['pose_hub_ur2']
pose_hdmi=joint_poses['pose_hdmi']
#pose_USB=joint_poses['pose_USB']
pose_complete_hdmi=joint_poses['pose_complete_hdmi']


ur2_pencilcase_lid_position1 = np.deg2rad([26.25,-68.96,85.44,-91.10,-121.01,-59.72])
ur2_pencilcase_lid_position2 = np.deg2rad([26.03,-52.06,99.07,-121.48,-121.24,-59.44])
ur2_sharpener_position=np.deg2rad([-6.51,-102.68,121.81,-118.85,-153.11,-100.40])
ur_orange_sharpener_inserting_position = np.deg2rad([11.86,-55.60,-112.03,-191.36,-111.04,0.16])
ur_blue_sharpener_inserting_position = np.deg2rad([-0.44,-57.97,-114.96,-185.93,-123.35,0.41])
ur2_sharpener_setting_position= np.deg2rad([16.80,-101.40,118.15,-107.48,-89.33,-73.24])
ur2_sharpener_waiting_position = np.deg2rad([52.86,-69.37,80.92,-42.44,-121.27,-17.58])
ur_pencil_insering_pose=np.deg2rad([8.66,-66.57,-138.96,25.82,113.75,0.0])
ur_pencilcase_inserting_pose=np.deg2rad([2.22,-102.65,-124.07,-33.43,158.46,-80.37])
#ur_pencilcase_inserting_pose2=np.deg2rad([2.49,-102.75,-124.17,-32.55,158.42,-79.64])
ur2_pencilcase_inserting_pose=np.deg2rad([-2.02,-94.58,101.15,-99.81,-126.07,-93.10])
ur2_pencilcase_placing_pose=np.deg2rad([74.71,-142.02,121.66,-69.86,-89.94,-15.95])
#####
ur_plug_pose=np.deg2rad([85.80,-25.35,-100.55,-52.95,93.24,0.0])
ur2_plug_pose=np.deg2rad([-85.27,-155.64,97.77,-122.09,-94.71,-90.0])
ur2_greenmultitap_position1=np.deg2rad([5.43,-103.35,132.09,-119.50,-89.25,-84.46])
ur2_greenmultitap_position2=np.deg2rad([5.30,-74.99,144.69,-160.45,-89.74,-84.18])
ur2_blackmultitap_position1=np.deg2rad([5.43,-103.35,132.09,-119.50,-89.25,-84.46])###
ur2_blackmultitap_position2=np.deg2rad([5.30,-74.99,144.69,-160.44,-89.74,-84.18])###
ur2_multitap_pose=np.deg2rad([38.31,-57.30,113.09,-91.15,-136.76,-27.27])
ur_whiteplug_inserting_position = np.deg2rad([49.74,-94.55,-135.46,-40.45,89.95,106.76])
ur_blackplug_inserting_position= np.deg2rad([54.57,-96.58,-134.13,-39.75,89.87,98.99])
ur2_multitap_back_position1=np.deg2rad([27.27,-63.41,86.28,-46.30,-79.29,4.14])
ur2_multitap_back_position2=np.deg2rad([29.01,-87.87,123.93,-61.09,-66.25,10.03])

ur_multitap_position=np.deg2rad([17.20,-135.28,-84.50,-50.76,89.84,104.71])

ur2_multitap_position=np.deg2rad([-14.59,-50.68,80.29,-118.51,-92.35,-106.13])
# ur_plug_waiting_position = np.deg2rad([13.93, -99.64, -70.33, -98.26, 90.00, 104.16])
# ur2_plug_grap_start_position = np.deg2rad([24.35, -70.15, 77.5, -25.29, -170.34, -15.55])
# ur2_plug_grap_end_position = np.deg2rad([0.67, -73.88, 89.21, -59.19, -180.93, -45.80])
# ur_plug_grap_position = np.deg2rad([-10.04, -116.16, -70.19, -163.75, 170.57, 9.81])
#####


ur_bottle_back_position = np.deg2rad([-43.53, -104.28, -89.04, -173.44, -183.65, -7.80])
#ur_bottle_back_position = np.deg2rad([-39.00, -113.35, -122.70, -114.19, -177.21, 9.52])
ur_bottle_holding_position = np.deg2rad([-9.18, -105.66, -93.86, -167.51, -167.41, -9.30])
# ur_bottle_holding_position = np.deg2rad([-7.63, -105.87, -93.55, -166.85, -165.87, -8.53])
#ur_bottle_holding_position = np.deg2rad([-5.39, -126.21, -121.24, -108.16, -148.23, 4.38])
ur2_bottle_center_position = np.deg2rad([-26.99, -82.21, 68.12, -78.27, -89.03, 180.00])
ur2_bottle_lid_open_joint = np.deg2rad([-26.99, -82.21, 68.12, -78.27, -89.03, -240.00])
ur2_back_path1 = np.deg2rad([31.63, -104.47, 92.88, -80.02, -88.11, 8.55])
ur2_back_path2 = np.deg2rad([64.22, -133.21, 123.90, -82.35, -88.06, 2.85])
# ur2_back_path2 = np.deg2rad([69.86, -124.32, 121.09, -88.53, -88.23, 8.60])

ur_pen_lid_position = np.deg2rad([-15.20, -104.14, -69.34, -186.35, 170.98, 0])
ur2_pen_lid_position1 = np.deg2rad([-5.34, -82.00, 71.91, 7.45, 172.72, -182.43])
ur2_pen_lid_position2 = np.deg2rad([-5.78, -78.77, 58.79, 19.58, 172.25, -180.05])
#ur2_pen_lid_position2 = np.deg2rad([-5.51, -78.75, 60.61, 17.79, 172.54, -0.02])

ur2_holder_setting_pose = np.deg2rad([24.03,-94.59,137.18,-133.31,-89.64,-65.71])
ur2_holder_setting_pose2_without_tail = np.deg2rad([23.97, -82.07, 142.05, -150.69, -89.86, -65.59])
ur2_holder_setting_pose2_with_tail= np.deg2rad([23.99, -87.10, 140.59, -144.20, -89.77, -65.63])
ur2_holder_setting_pose3 = np.deg2rad([22.21,-87.19,135.01,-138.54,-89.68,-67.46])
#ur2_holder_setting_pose3 = np.deg2rad([22.86,-87.38,135.21,-138.55,-89.69,-66.81])
ur2_holder_setting_pose4 = np.deg2rad([43.11,-59.35,108.74,-95.00,-124.68,-29.35])
ur_holder_insert_pose = np.deg2rad([0.0, -90.0, -90.0, -90.0, 90.0, 90.0])
ur_holder_insert_pose2 = [-0.019875828419820607, -1.3842194716082972, -1.9088614622699183,
                 -2.4182804266559046, 3.1394150257110596, 0.5713483095169067]
ur2_holder_insert_pose=np.deg2rad([3.54,-103.15,123.99,-105.31,-141.75,-80.07])
ur2_holder_back_pose=np.deg2rad([58.98,-71.66,108.57,-78.36,-113.12,-16.80])
ur2_holder_back_pose2_without_tail=np.deg2rad([58.90,-60.48,112.97,-94.74,-113.25,-16.57])
ur2_holder_back_pose2_with_tail=np.deg2rad([58.91,-62.39,112.56,-92.43,-113.24,-16.61])

ur2_green_desk_cleaner_center_joint =  np.deg2rad([-1.19, -55.53, 103.28, -139.09, -89.33, -90.03])
ur2_blue_desk_cleaner_center_joint =  np.deg2rad([-1.19, -55.34, 103.32, -139.32, -89.33, -90.03])

ur_move_path_j = np.deg2rad([-32.75, -97.27, -126.14, -46.94, 89.61, 57.17])
ur_move_path_j1 = np.deg2rad([-10.51, -120.68, -130.60, -136.61, -11.99, -65.61])
ur_move_path_j2 = np.deg2rad([5.56, -116.55, -145.49, -104.21, 5.68, -86.02])

#######################################################################################################


center_point = {'CAMERA': [424, 240],
                #'DEFAULT': [525, 285],
                'DEFAULT': [605, 285],
                'ORANGE_PENCIL': [492, 291],
                'BLUE_PENCIL': [482, 323],
                'BLUE_SHARPENER': [490, 364],
                'BLACK_MULTITAP': [469, 356],
                'GREEN_MULTITAP': [496,305],##
                'WHITE_PLUG':[487,347],
                'BLACK_PLUG':[469,348],##333
                }  # 욜로 바운딩 박스 중심 좌표가 맞춰야 할 센터포인트

offsets={#y-0.0298551393
    'ur_offset':[-0.000671315-0.007527962,0.21389817518-0.021442295],#[-0.00715,0.212691]
    'bin_ur_offset':[0.003238813,0.21463580918],
    'pencilcase_ur_offset':[-0.002301443,0.221116841],
    'bottle': [0.001630017, -0.0377289918],
    'bottle_left': [0.000644781,-0.013004244],
    'bottle_right': [-0.001846623,0.030835143],
    'bottle_up': [-0.014868018,0.0022197943],
    'bottle_down': [0.013732438,0.0020260416],
    'pencilcase': [0.00265431, -0.01466258],
    'pencilcase_left': [-0.00147979,-0.01560547],
    'pencilcase_right': [-0.00486585,0.02715217],
    'pencilcase_up': [-0.01268375,0.00147263],
    'pencilcase_down': [0.00743479,-0.001828],
    'holder': [0.00704465 + 0.004864936, -0.02473583 - 0.0230779687],
    'holder_left': [-0.00123356, -0.02448734],
    'holder_right': [-0.000626458, 0.031885405],
    'holder_up': [-0.02295479, 0.003267413],
    'holder_down': [0.00467535, 0.00148786],
    'box': [0.005358316,-0.021030952],
    'box_left': [0.002963721, -0.028775398],
    'box_right': [0.002592996, 0.035170459],
    'box_up': [-0.0142009, 0.00225581],
    'box_down': [0.019180837, -0.0022966794],
    'cup': [-0.002743623+0.004864936, -0.0281167803-0.0230779687],
    'cup_left': [0.002963721, -0.028775398],
    'cup_right': [0.002592996, 0.035170459],
    'cup_up': [-0.0142009, 0.00225581],
    'cup_down': [0.019180837, -0.0022966794],
    'multitap': [0 + 0.004864936, -0.0197471694 - 0.0230779687],  # [0.001335784,-0.0159228765],
    'multitap_left': [0.000860688, -0.022395938],
    'multitap_right': [0, 0.022189864],
    'multitap_up': [-0.019121491, 0.0016252994],
    'multitap_down': [0.021158295, 0.0000000001],
    'glue':[-0.001181845+0.004864936,-0.0197969176-0.0230779687],
    'glue_left': [-0.000262536,-0.018532573],
    'glue_right': [0.001381521,0.030785199],
    'glue_up':[-0.01555703,0.0051989624],
    'glue_down':[0.016488032,-0.0081679948],




}

pixel_point={ #center x, center y, left, right, up, down
    'bottle':[368,338,45,913,47,654],
    'pencilcase': [414,351,41,913,195,565],
    'holder':[426,357,88,861,185,504],
    'box':[446,340,62,838,111,592],
    'cup':[427,350,62,838,111,592],
    'multitap':[482,349,175,740,152,530],
    'glue':[437,367,92,837,105,598],
}

depth_w={
    'bottle':0.6484503641414863,
    'pencilcase':0.6752562651288212,
    'holder':0.7736848498580137,
    'box':0.8737857530311662,
    'cup': 0.8036848498580137,
    'multitap': 0.8190340549767423,
    'glue':0.854256272315979,
}

object_z = {
    'DEFAULT':0.146,
    'MULTITAP':0.37,
    'HDMI_HUB':-0.108579240,
    'USB_C_HUB':-0.11504811,
    'HDMI':-0.108579240,
    'USB_C':-0.108579240,
    'SILVER_PENCILCASE':0.040132,
    'RED_PENCILCASE': 0.040132,
    'ORANGE_SHARPENER':-0.08838819,
    'BLUE_SHARPENER': -0.09838819,
    'ORANGE_PENCIL_INSERT': 0.365,
    'BLUE_PENCIL_INSERT':0.365,
    'ORANGE_PENCIL':-0.1106,
    'BLUE_PENCIL':-0.1106,
    'BLACK_MULTITAP':-0.134,
    'GREEN_MULTITAP':-0.134,
    'WHITE_PLUG': -0.102,
    'BLACK_PLUG': -0.102,
    'WHITE_PLUG_INSERT': -0.0648,
    'BLACK_PLUG_INSERT': -0.062,
    'z_lift' : 0.015,
    'z_tray' : -0.105,
    # 카메라 없는 손 기준
    'BLACK_TAPE': -0.144,
    'WHITE_TAPE': -0.144,
    'SMALL_BOX':-0.13,
    'BIG_BOX':-0.12,
    'GLUE_PEN':-0.135,
    'GLUE_STICK':-0.135,
    'RED_CUP':-0.12,
    'PINK_CUP':-0.12,
    'PINK_STAPLER':-0.135,
    'STAN_STAPLER':-0.135,
    'YOGURT':-0.103,
    'MILK':-0.085,
    'APRICOT_BOTTLE': 0.0553, #0.0639,
    'GREY_BOTTLE': 0.0553,#0.0639,
    'GREEN_HOLDER':-0.07,
    'BLACK_HOLDER':-0.10,
    'BLACK_MARKER':-0.14,
    'RED_MARKER':-0.14,
    'BLACK_NAMEPEN':-0.14,
    'SILVER_NAMEPEN':-0.14,
    'GREEN_DESK_CLEANER':-0.12,
    'BLUE_DESK_CLEANER':-0.12,
    'SMALL_USB': -0.14,
    'BIG_USB':-0.14,
    'SMALL_ERASER':-0.14,
    'BIG_ERASER':-0.14,
    'GREY_CUP':-0.12,
    'GREEN_CUP':-0.12,
    'BLUE_CUP':-0.12,
    'PURPLE_CUP':-0.12,
    'SILVER_CUP':-0.12,
    'WHITE_BOX':-0.14,
    'RED_BOX':-0.14,
    'YELLOW_BOX':-0.12,
    'GREEN_BOX':-0.12,
    'PINK_BOX':-0.12,
}

object_w={#0.001단위로 맞출것
    'GREEN_BOOK':0.1475,#
    'BLUE_BOOK':0.1485,
    'BLACK_FILE_HOLDER':0.177,
    'PINK_FILE_HOLDER':0.177,
    'BLACK_KEYBOARD':0.150,
    'PINK_KEYBOARD':0.151,
}

goal_positions={
    'GREEN_BOOK':[-0.663, 0.312],
    'BLUE_BOOK':[-0.663, 0.312],
    'BLACK_FILE_HOLDER':[-0.663, 0.312],
    'PINK_FILE_HOLDER':[-0.663, 0.312],
    'BLACK_KEYBOARD':[-0.713, 0.312],
    'PINK_KEYBOARD':[-0.713, 0.312],
}

targets = ['RED_CUP', 'PINK_CUP','SMALL_BOX','BIG_BOX','PINK_STAPLER','STAN_STAPLER','GLUE_PEN','GLUE_STICK',
            'BLACK_TAPE','WHITE_TAPE',
           'GREY_CUP', 'GREEN_CUP', 'BLUE_CUP', 'PURPLE_CUP', 'SILVER_CUP', 'WHITE_BOX', 'RED_BOX', 'YELLOW_BOX',
           'GREEN_BOX', 'PINK_BOX',
            'WHITE_BIN','STAN_BIN','MILK','YOGURT',
            'LIGHT_DRAWER','DARK_DRAWER', 'SMALL_USB','BIG_USB','SMALL_ERASER', 'BIG_ERASER',
            'APRICOT_BOTTLE','GREY_BOTTLE',
            'GREEN_HOLDER','BLACK_HOLDER', 'BLACK_MARKER','RED_MARKER','BLACK_NAMEPEN','SILVER_NAMEPEN',
            'GREEN_BOOK','BLUE_BOOK','BLACK_FILE_HOLDER','PINK_FILE_HOLDER','BLACK_KEYBOARD','PINK_KEYBOARD',
            'GREEN_DESK_CLEANER', 'BLUE_DESK_CLEANER',
            'ORANGE_PENCIL', 'ORANGE_SHARPENER', 'SILVER_PENCILCASE',
            'BLUE_PENCIL', 'BLUE_SHARPENER', 'RED_PENCILCASE',
            'BLACK_PLUG', 'BLACK_MULTITAP',
            'WHITE_PLUG', 'GREEN_MULTITAP',
            'HDMI_HUB', 'HDMI',
            'USB_C_HUB', 'USB_C',
           ]


tails = [
    'HDMI_TAIL','USB_C_TAIL','HDMI_HUB_TAIL','USB_C_HUB_TAIL','WHITE_PLUG_TAIL','BLACK_PLUG_TAIL','BLACK_MULTITAP_TAIL',
    'GREEN_MULTITAP_TAIL','ORANGE_PENCIL_TAIL','BLUE_PENCIL_TAIL','BLUE_SHARPENER_TAIL','ORANGE_SHARPENER_TAIL',
    'GREEN_DESK_CLEANER_TAIL','BLUE_DESK_CLEANER_TAIL','RED_CUP_TAIL','PINK_CUP_TAIL','SMALL_BOX_TAIL','BIG_BOX_TAIL',
    'PINK_STAPLER_TAIL','STAN_STAPLER_TAIL','GLUE_PEN_TAIL','GLUE_STICK_TAIL','BLACK_MARKER_TAIL','RED_MARKER_TAIL',
    'BLACK_NAMEPEN_TAIL','SILVER_NAMEPEN_TAIL','MILK_TAIL','YOGURT_TAIL','SMALL_USB_TAIL','BIG_USB_TAIL',
    'SMALL_ERASER_TAIL','BIG_ERASER_TAIL','GREEN_BOOK_TAIL','BLUE_BOOK_TAIL','BLACK_FILE_HOLDER_TAIL',
    'PINK_FILE_HOLDER_TAIL','BLACK_KEYBOARD_TAIL','PINK_KEYBOARD_TAIL','GREEN_HOLDER_TAIL',
    'GREY_CUP_TAIL','GREEN_CUP_TAIL','BLUE_CUP_TAIL','PURPLE_CUP_TAIL','SILVER_CUP_TAIL','WHITE_BOX_TAIL',
    'RED_BOX_TAIL','YELLOW_BOX_TAIL'

]


picking_list = ['RED_CUP', 'PINK_CUP','SMALL_BOX','BIG_BOX','PINK_STAPLER','STAN_STAPLER','GLUE_PEN','GLUE_STICK',
                'BLACK_TAPE','WHITE_TAPE', 'GREY_CUP', 'GREEN_CUP', 'BLUE_CUP', 'PURPLE_CUP', 'SILVER_CUP', 'WHITE_BOX',
                'RED_BOX', 'YELLOW_BOX','GREEN_BOX', 'PINK_BOX',]
bin_list=['WHITE_BIN','STAN_BIN']
bin_obj_list=['MILK','YOGURT']
drawer_list=['LIGHT_DRAWER','DARK_DRAWER']
drawer_obj_list=[ 'SMALL_USB','BIG_USB','SMALL_ERASER', 'BIG_ERASER']
bottle_lid_list=['APRICOT_BOTTLE','GREY_BOTTLE']
pen_lid_list=['BLACK_MARKER','RED_MARKER','BLACK_NAMEPEN','SILVER_NAMEPEN']
holder_list=['GREEN_HOLDER','BLACK_HOLDER']
wide_object_list=['GREEN_BOOK','BLUE_BOOK','BLACK_FILE_HOLDER','PINK_FILE_HOLDER','BLACK_KEYBOARD','PINK_KEYBOARD']
cleaner_list=['GREEN_DESK_CLEANER', 'BLUE_DESK_CLEANER']
pencil_list1=['ORANGE_PENCIL', 'ORANGE_SHARPENER', 'SILVER_PENCILCASE']
pencil_list2=['BLUE_PENCIL', 'BLUE_SHARPENER', 'RED_PENCILCASE']
multitap_list1=['BLACK_PLUG', 'BLACK_MULTITAP']
multitap_list2=['WHITE_PLUG', 'GREEN_MULTITAP']







#카메라 로봇 캘리브레이션 매칭 행렬
# M_k2b = np.array([[0.00213347, 1.02294, 0.229744, -0.709588],
#                   [1.01111, -0.0091172, -0.0118871, 0.268584],
#                   [0.00455328, 0.00479986, -0.597944, 0.408516],
#                   [0, 0, 0, 1]])

M_k2b = np.array([[0.011915, 0.922167, 0.000889179, -0.51526],
                  [0.933015, -0.00308057, 0.00291996, 0.288798],
                  [0.0240708, -0.00599485, -0.748247, 0.567094],
                  [0, 0, 0, 1]])

