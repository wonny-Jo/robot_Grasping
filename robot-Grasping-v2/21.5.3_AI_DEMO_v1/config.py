from model.function import *

##### 고정 위치 좌표들
with open("./data/robot_main/pose_cam", 'rb') as f:  # 잡기전 대기 자세
    pose_start = pickle.load(f)['qActual']
with open("./data/robot_main/pose_pre_transfer1_cable", 'rb') as f:  # 케이블 전달 전 자세
    pose_pre_transfer1_cable = pickle.load(f)['qActual']
with open("./data/robot_main/pose_transfer1_cable", 'rb') as f:  # 케이블 전달 자세
    pose_transfer1_cable = pickle.load(f)['qActual']
with open("./data/robot_main/pose_pre_transfer2_cable", 'rb') as f:  # 케이블 다시 전달받기 전 자세
    pose_pre_transfer2_cable = pickle.load(f)['qActual']
with open("./data/robot_main/pose_transfer2_cable", 'rb') as f:  # 케이블 다시 전달받기 자세
    pose_transfer2_cable = pickle.load(f)['qActual']
with open("./data/robot_main/pose_insert_hdmi", 'rb') as f:  # 케이블 허브에 조립하기 자세
    pose_insert_hdmi = pickle.load(f)['qActual']
with open("./data/robot_main/psoe_hub", 'rb') as f:  # 케이블 허브 임시 보관 위치
    pose_hub_ur = pickle.load(f)['qActual']
with open("./data/robot_sub/pose_base", 'rb') as f:  # 잡기전 대기 자세
    pose_base = pickle.load(f)['qActual']
with open("./data/robot_sub/pose_transfer1", 'rb') as f:  # 케이블 전달받기 자세
    pose_transfer1 = pickle.load(f)['qActual']
with open("./data/robot_sub/pose_transfer2", 'rb') as f:  # 케이블 전달 받은후 다시 건내주기 자세
    pose_transfer2 = pickle.load(f)['qActual']
with open("./data/robot_sub/pose_hub", 'rb') as f:  # 허브 임시 보관 위치
    pose_hub_ur2 = pickle.load(f)['qActual']
with open("./data/robot_sub/pose_insert_hdmi", 'rb') as f:  # 케이블 허브 조립 자세
    pose_hdmi = pickle.load(f)['qActual']
with open("./data/robot_sub/pose_insert_USB", 'rb') as f:  # 케이블 허브 조립 자세
    pose_USB = pickle.load(f)['qActual']
with open("./data/robot_sub/pose_complete_hdmi", 'rb') as f:  # hdmi 조립 완료 위치
    pose_complete_hdmi = pickle.load(f)['qActual']

ur_pencilcase_position = np.deg2rad([25.13, -130.05, -59.28, -80.60, 89.84, 24.94])
ur_pencilcase_lid_position = np.deg2rad([-10.48, -138.59, -68.43, -63.00, 89.87, -10.73])
ur_pencilcase_lid_back_position = np.deg2rad([25.01, -129.73, -58.47, -81.74, 89.84, 24.82])
ur2_pencilcase_start = np.deg2rad([27.51, -31.65, 67.23, -36.48, -178.26, 0.09])
ur2_pencilcase_position = np.deg2rad([12.94, -25.90, 58.10, -37.43, -178.49, -5.46])
ur_sharpener_position = np.deg2rad([15.75, -136.56, -78.05, -55.35, 89.89, 15.51])
#####
ur_sharpener_inserting_position = np.deg2rad([7.56, -117.66, -100.57, -140.82, -115.51, 0])
ur_pencilcase_inserting_position = np.deg2rad([19.64, -109.85, -90.50, -155.81, -103.42, 0.94])
ur_pencilcase_inserting_back_position = np.deg2rad([19.64, -109.85, -90.50, -155.81, -103.42, 180.94])
#####
ur_plug_waiting_position = np.deg2rad([13.93, -99.64, -70.33, -98.26, 90.00, 104.16])
ur2_plug_grap_start_position = np.deg2rad([24.35, -70.15, 77.5, -25.29, -170.34, -15.55])
ur2_plug_grap_end_position = np.deg2rad([0.67, -73.88, 89.21, -59.19, -180.93, -45.80])
ur_plug_grap_position = np.deg2rad([-10.04, -116.16, -70.19, -163.75, 170.57, 9.81])
#####


center_point = {'grasp_center_point': [525, 285],
                'silver_pencilcase_center_point': [471, 320],  # 309
                # 'red_pencilcase_center_point':[477,323]
                'pencilsharpener_center_point': [467, 297],
                'pencil_center_point': [467, 297],
                'white_plug_center_point': [463, 295],
                'black_multitap_center_point': [0, 0],  ####

                }  # 욜로 바운딩 박스 중심 좌표가 맞춰야 할 센터포인트
object_z = {'RED_PENCILCASE': 0.05957288,
            'BLUE_SHARPENER': -0.07025146,
            'WHITE_PLUG': -0.09355089,
            'BLACK_MULTITAP':-0.11067233,####

            }