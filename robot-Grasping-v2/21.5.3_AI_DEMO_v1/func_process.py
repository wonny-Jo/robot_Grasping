from torch.lib import *
from model.function import *

from device.camera.RS2 import *
from device.robot.universal_robot import *
from device.thread import *

from PIL import Image
from torchvision import transforms
from config import *


with open("./data/trajectory_approach", 'rb') as f:
    traj_approch = pickle.load(f)
img_back = Image.open("./data/back.png")


t1 = None
t2 = None
t3 = None

def set_thread():
    global t1
    global t2
    global t3

    t1= t_rs(realsense(img_size=[848, 480], frame=60))
    # t2= t_cam(0)

def start_thread():
    t1.start()
    t1.ON= True
    # t2.start()

def isDetected(target_list):
    detections, _ = YOLO(model_yolo, t1.rs)  # Calls the main function YOLO()
    cnt = 0
    for target in target_list:
        for detection in detections:
            if target == detection[0]:
                cnt += 1
    return len(target_list) == cnt

def tracking_target(ur, target_name, target_point, model_yolo, grasp_threshold = 0.01):

    acc = 1
    min_t = 0.1
    v_w = 0.25
    step = 0
    prev_target = None

    # tracking loop
    while True:
        time1 = time.time()
        t1.detections, img = YOLO(model_yolo, t1.rs)  # Calls the main function YOLO()
        target, bbox = find_target(t1.detections, target_name, target_point)

        #인식 결과 처리
        if None == target and None != prev_target :     #인식 잘 되다가 안되는 경우, 이전 정보 덮어 씌움
            print("목표를 찾을 수 없습니다.")
            target = prev_target
            # bbox = prev_bbox
            v_w = 0.1
        elif None == target and None == prev_target:
            v_w = 0.1
            target = target_point
        else:                                          # 인식 잘 되는 경우
            v_w = 0.25
            prev_target = target
            # prev_bbox = bbox

        dist = np.sqrt(((target_point[0] - target[0]) / 424) ** 2 + (-(target_point[1] - target[1]) / 240) ** 2)
        # x,y축 속도 계산 식.
        vel = [((target_point[1] - target[1]) / 480) * v_w, ((target_point[0] - target[0]) / 240) * v_w, 0, 0,
               0, 0]

        ur.ur_rob.speedx("speedl", vel, acc=acc, min_time=min_t)

        # yolo 모델 프레임 속도 맞추기 위한 대기
        while True:
            if time.time() - time1 > 0.033:
                break

        # grasping points에 근접하면 tracking 중단.
        if dist < grasp_threshold:
            tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
            ur.ur_rob.movel(tmp_pose, vel=1, acc=1, wait=False, relative=False)
            waiting_target(ur,tmp_pose,0.001)
            break

    return target

def grasping_target(model_yolo, model_vae):

    # for i in range(200):
    t1.detections, img = YOLO(model_yolo, t1.rs)  # Calls the main function YOLO()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cur_img = Image.fromarray(img)
    rad = model_vae.predict(cur_img, t1.detections[0][2])
    print ("rad : " + str(rad*180/3.14))
    return t1.img.copy()

"""
def grasping_target(ur, target_name, target_point, model_yolo, model_vae, model_rot, model_umap):
    rz_list = []
    
    for i in range(10):
        frame = cv2.cvtColor(t1.img, cv2.COLOR_BGR2RGB)

        t1.detections,img = YOLO(model_yolo, t1.rs)  # Calls the main function YOLO()
        [x,y], [w,h]= find_target(t1.detections, target_name, target_point) # b"HUB"

        if w > h:
            sz_img = w
        else:
            sz_img = h

        cur_img = Image.fromarray(frame).convert("RGBA")
        input_img = cur_img.crop((x - int(w / 2), y - int(h / 2), x + int(w / 2), y + int(h / 2)))
        img_tmp = img_back.copy()
        img_tmp.paste(input_img, (int(img_back.size[0] / 2) - int(w / 2), int(img_back.size[1] / 2) - int(h / 2)))
        input_img = img_tmp.crop(
            (int(img_back.size[0] / 2) - int(sz_img / 2), int(img_back.size[1] / 2) - int(sz_img / 2),
             int(img_back.size[0] / 2) + int(sz_img / 2), int(img_back.size[1] / 2) + int(sz_img / 2)))

        input_img = input_img.resize((256, 256))
        input_img = transforms.ToTensor()(input_img).unsqueeze(0)
        vector,_ = model_vae.get_mu_z(input_img.to('cuda'))
        vector = vector.cpu().detach().numpy()
        vector = model_umap.transform(vector)
        rz = model_rot(torch.tensor(vector).cuda(device))
        rz = rz.cpu().detach().numpy()[0,0]
        rz = rz * (360 / 108)
        if rz < 180:
            rz = (rz * np.pi) / 180
        else:
            rz = -((360 - rz) * np.pi) / 180
        rz_list.append(rz)


    t1.detections = []

    rz = np.mean(rz_list)

    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[0]+=0.055
    ur.ur_rob.movel(tmp_pose , 1, 1, wait=False, relative=False)
    waiting_target(ur,tmp_pose, 0.0005)
    # tmp_pose[2] = -0.033
    # ur.ur_rob.movel(tmp_pose, 1, 1, wait=False, relative=False)
    # waiting_target(ur, tmp_pose, 0.0005)
    if target_name == b'HUB2':
        rotation_task(ur, rz, -0.117)
    else:
        rotation_task(ur, rz, -0.112)

    if target_name == b'USB_C':
        if rz > 0:
            ur.setGripper(100)
            time.sleep(0.5)
            move_rotation(ur, rz, -0.01, -0.112)
        # else:
        #     move_rotation(ur, rz, 0.01, -0.112)

        time.sleep(0.5)
        ur.setGripper(255)
"""

def rotation_task(ur, rz,z=None):
    pose_current = ur.ur_rob.rtmon.get_all_data()['tcp']
    rot_cur = math3d.Transform(np.array([2.22, -2.22, 0]), np.array([0, 0, 0]))
    pose_next = (math3d.Orientation([0, 0, rz]) * rot_cur.orient.inverse).inverse.rotation_vector
    pose_current[3:6] = pose_next
    if z:
        pose_current[2] = z
    ur.ur_rob.movel(pose_current, 1, 1, wait=False, relative=False)
    waiting_target(ur, pose_current, 0.0002)

    #ur.setGripper(255)

def move_rotation(ur, rz, x, z):

    pose_current = ur.ur_rob.rtmon.get_all_data()['tcp']
    pose_current[3:] = [2.221, -2.221, 0]
    rot_cur = math3d.Transform(np.array(pose_current[3:6]), pose_current[:3])
    pose_next = (math3d.Orientation([0, 0, rz]) * rot_cur.orient.inverse).inverse.rotation_vector
    pose_current[3:6] = pose_next
    pose_current[2] = z

    # x1 = pose_current[0] + (-0.05) * np.cos(-np.pi * 45 / 180) - (0) * np.sin(
    #     -np.pi * 45 / 180)
    # y1 = pose_current[1] - (-0.05) * np.sin(-np.pi * 45 / 180) - (0) * np.cos(
    #     -np.pi * 45 / 180)

    x1 = pose_current[0] + (x) * np.cos(-rz) - (0) * np.sin(-rz)
    y1 = pose_current[1] - (x) * np.sin(-rz) - (0) * np.cos(-rz)

    pose_current[0] = x1
    pose_current[1] = y1
    ur.ur_rob.movel(pose_current, 1, 1, wait=False, relative=False)
    waiting_target(ur,pose_current,0.0005)


def get_img():
    return t1.img.copy()

def calc_rotation(cvt_image, vae_model, reducer, u_vector, rot_vals):
    input_img = transforms.ToTensor()(cvt_image).unsqueeze(0)
    vector = vae_model.get_mu(input_img.to(vae_model.device))
    vector = vector.cpu().detach().numpy()
    vector = reducer.transform(vector)

    norm_list = np.zeros(len(u_vector))
    for u in range(0, len(u_vector)):
        norm_list[u] = np.linalg.norm(u_vector[u] - vector[0])
    rot_res = np.where(norm_list == np.min(norm_list))[0][0]
    rz = rot_vals[rot_res]

    if rz < 180:
        rz = (rz * np.pi) / 180
    else:
        rz = -((360 - rz) * np.pi) / 180

    return rz


########################## wonny ####################################
model_yolo = init_YOLO()    # YOLO 모델 정보 읽기 ( 모델 세이브 이름 바뀌면 여기서 수정 )


def hub_cable_task(ur,ur2,target_list):
    target_hub_name, target_cable_name = target_list
    # 1. 허브 잡기(모델 사용)
    target_point = tracking_target(ur, target_hub_name, target_point=center_point['grasp_center_point'], model_yolo=model_yolo,
                                   grasp_threshold=0.005)
    time.sleep(1)
    grasping_hub(ur, target_hub_name, target_point, model_yolo, None, None, None)

    # 2.HUB 옴기기
    ur.ur_rob.movej(pose_start, acc=2, vel=2, wait=False, relative=False)
    waiting_joint(ur, pose_start, 0.005)
    ur.ur_rob.movej(pose_hub_ur, acc=2, vel=2, wait=False, relative=False)
    waiting_joint(ur, pose_hub_ur, 0.001)
    ur.setGripper(0)
    time.sleep(0.5)

    # 초기 위치로 되돌림
    ur.ur_rob.movej(pose_start, acc=2, vel=2, wait=False, relative=False)
    ur2.ur_rob.movej(pose_base, acc=2, vel=2, wait=False, relative=False)
    waiting_joint(ur, pose_start, 0.005)
    waiting_joint(ur2, pose_base, 0.005)

    # 3.케이블 잡기(모델사용)
    target_point = tracking_target(ur, target_cable_name, target_point=center_point['grasp_center_point'], model_yolo=model_yolo,
                                   grasp_threshold=0.001)
    time.sleep(1)
    grasping_cable(ur, target_cable_name, target_point, model_yolo, None, None, None)

    # 4.고쳐잡기
    # 케이블 전달 전 자세
    ur.ur_rob.movej(pose_pre_transfer1_cable, acc=2, vel=2, wait=False, relative=False)
    waiting_joint(ur, pose_pre_transfer1_cable, 0.001)

    # 케이블 전달받기 전 자세
    ur2.ur_rob.movej(pose_transfer1, acc=2, vel=2, wait=False, relative=False)
    waiting_joint(ur2, pose_transfer1, 0.001)

    # 케이블 전달 자세
    ur.ur_rob.movej(pose_transfer1_cable, acc=2, vel=2, wait=False, relative=False)
    waiting_joint(ur, pose_transfer1_cable, 0.001)
    ur2.setGripper(255)
    time.sleep(0.5)
    ur.setGripper(0)

    # 케이블 다시 가져가기 전자세
    ur.ur_rob.movej(pose_pre_transfer2_cable, acc=2, vel=2, wait=False, relative=False)
    waiting_joint(ur, pose_pre_transfer2_cable, 0.001)
    ur2.ur_rob.movej(pose_transfer2, acc=2, vel=2, wait=False, relative=False)
    waiting_joint(ur2, pose_transfer2, 0.001)

    # 케이블 다시 잡기
    ur.ur_rob.movej(pose_transfer2_cable, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur, pose_transfer2_cable, 0.001)
    ur.setGripper(255)
    time.sleep(0.5)
    ur2.setGripper(0)
    time.sleep(0.5)
    ur.ur_rob.movej(pose_pre_transfer2_cable, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur, pose_pre_transfer2_cable, 0.001)

    # 2. 허브 임시 위치에서 가져오기
    ur2.ur_rob.movej(pose_hub_ur2, acc=2, vel=2, wait=False, relative=False)
    waiting_joint(ur2, pose_hub_ur2, 0.001)

    tmp_pose = ur2.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[2] -= 0.005

    # 3.고쳐잡기
    # 허브 마다 잡는 위치 다르기 때문에 구분.
    if target_list[0] == 'USB_C':
        # 허브 잡는 위치 조정
        tmp_pose[0] += 0.02

        ur2.ur_rob.movel(tmp_pose, acc=2, vel=2, wait=False, relative=False)
        waiting_target(ur2, tmp_pose, 0.001)
        ur2.setGripper(255)
        time.sleep(1.5)

        # USB_C 조립
        ur2.ur_rob.movej(pose_USB, acc=1, vel=1, wait=False, relative=False)
        waiting_joint(ur2, pose_USB, 0.001)
        # 삽입 자세 이동
        ur.ur_rob.movej(pose_insert_hdmi, acc=1, vel=1, wait=False, relative=False)
        waiting_joint(ur, pose_insert_hdmi, 0.001)

        # 조립
        tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
        tmp_pose[:3] -= [0.04848896, -0.00659315, 0.02710033]
        ur.ur_rob.movel(tmp_pose, acc=0.1, vel=0.1, wait=False, relative=False)
        waiting_target(ur, tmp_pose, 0.0002)
    else:
        ur2.ur_rob.movel(tmp_pose, acc=1, vel=1, wait=False, relative=False)
        waiting_target(ur2, tmp_pose, 0.001)
        ur2.setGripper(255)
        time.sleep(1.5)

        # HDMI 조립
        ur2.ur_rob.movej(pose_hdmi, acc=1, vel=1, wait=False, relative=False)
        waiting_joint(ur2, pose_hdmi, 0.001)
        ur.ur_rob.movej(pose_insert_hdmi, acc=1, vel=1, wait=False, relative=False)
        waiting_joint(ur, pose_insert_hdmi, 0.001)
        # 삽입 자세 이동
        ur.ur_rob.movej(pose_insert_hdmi, acc=1, vel=1, wait=False, relative=False)
        waiting_joint(ur, pose_insert_hdmi, 0.001)

    # 4. 허브에 케이블 삽입 작업
    inserting_task(ur,ur2)

def grasping_hub(ur, target_name, target_point, model_yolo, model_vae, model_rot, model_umap):

    # while True:
    t1.detections, img = YOLO(model_yolo, t1.rs)  # Calls the main function YOLO()

    [x1, y1], [w, h] = find_target(t1.detections, target_name, target_point)  # b"HUB"
    [x2, y2], [w, h] = find_target(t1.detections, 'HUB_tail', target_point)  # b"HUB"
    y1 =480-y1
    y2 =480-y2
    x = x2 - x1
    y = y2 - y1

    # val = (x1 * y2 - y1 * x2) / (np.sqrt(x1 * x1 + y1 * y1) * np.sqrt(x2 * x2 + y2 * y2))
    # rad = np.arcsin(val)

    if target_name == 'HUB2' or target_name == 'HUB3':
        x_ = -y*np.sin(np.pi/2)
        y_ = x*np.sin(np.pi/2)
        x = x_
        y = y_

    if x == 0 and y > 0:
        rad = np.pi
    elif x == 0 and y < 0:
        rad = 0
    else:
        rad = math.atan2(y, x)

    if rad < 0:
        rad = np.pi + np.pi + rad

    print("회전각 : " + str(rad * 180 / np.pi))

    t1.detections = []
    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[0]+=0.055
    ur.ur_rob.movel(tmp_pose , 1, 1, wait=False, relative=False) # 그리퍼가 물체 중앙에 오도록 이동
    waiting_target(ur,tmp_pose, 0.0005)

    if target_name == 'HUB2':
        rotation_task(ur, rad, -0.117)
    else:
        rotation_task(ur, rad, -0.112)
    ur.setGripper(255)
    if target_name == 'USB_C':
        ur.setGripper(100)
        time.sleep(0.5)

        if rad < 3.14:
            move_rotation(ur, rad, -0.01, -0.112)
        else:
            move_rotation(ur, rad, 0.01, -0.112)

        time.sleep(0.5)
        ur.setGripper(255)

def grasping_cable(ur, target_name, target_point, model_yolo, model_vae, model_rot, model_umap):

    t1.detections, img = YOLO(model_yolo, t1.rs)  # Calls the main function YOLO()
    [x1, y1], [w, h] = find_target(t1.detections, target_name, target_point)  # b"HUB"
    if target_name == 'HDMI':
        [x2, y2], [w, h] = find_target(t1.detections, 'HDMI_tail', target_point)  # b"HUB"
    else :
        [x2, y2], [w, h] = find_target(t1.detections, 'USB_tail', target_point)  # b"HUB"
    y1 =480-y1
    y2 =480-y2
    x = x2 - x1
    y = y2 - y1

    x_ = -y*np.sin(-np.pi/2)
    y_ = x*np.sin(-np.pi/2)
    x = x_
    y = y_

    if x == 0 and y > 0:
        rad = np.pi
    elif x == 0 and y < 0:
        rad = 0
    else:
        rad = math.atan2(y, x)

    if rad < 0:
        rad = np.pi + np.pi + rad

    # print("회전각 : " + str(rad * 180 / np.pi))

    t1.detections = []
    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[0]+=0.055
    ur.ur_rob.movel(tmp_pose , 1, 1, wait=False, relative=False)
    waiting_target(ur,tmp_pose, 0.0005)

    if target_name == 'HUB2':
        rotation_task(ur, rad, -0.117)
    else:
        rotation_task(ur, rad, -0.112)
    ur.setGripper(255)
    if rad < 3.14:
        ur.setGripper(100)
        time.sleep(0.5)
        move_rotation(ur, rad, -0.007, -0.112)
        time.sleep(0.5)
        ur.setGripper(255)

def inserting_task(ur,ur2):
    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    control_params = get_inserting_control_params()
    for i in range(20):
        x_ = np.random.uniform(-0.005, 0.005)
        y_ = np.random.uniform(-0.003, 0.003)
        target_pose = tmp_pose + [x_, y_, 0, 0, 0, 0]
        fm_rob_control(ur, 1, target_pose, control_params)
        time.sleep(0.3)
        if ur.ur_rob.rtmon.get_all_data()['tcp'][2] < 0.0365:
            break
    ur.setGripper(0)

    # 조립완료물품 옮기기
    ur2.ur_rob.movej(pose_complete_hdmi, acc=2, vel=2, wait=False, relative=False)
    waiting_joint(ur2, pose_complete_hdmi, 0.001)
    ur2.setGripper(0)

    ur.ur_rob.movej(pose_start, acc=2, vel=2, wait=False, relative=False)
    waiting_joint(ur, pose_start, 0.001)
    ur2.ur_rob.movej(pose_base, acc=2, vel=2, wait=False, relative=False)
    waiting_joint(ur2, pose_base, 0.001)


def pencil_task(ur,ur2,target_list):
    target_pencil_name, target_sharpener_name,target_pencilcase_name = target_list
    # # 1. 연필통 뚜껑 열기 및 고정 위치에 위치 시키기
    (ur,ur2,target_pencilcase_name)
    # # 2. 연필깎이 고정위치로 옮기기
    placing_sharpener(ur,target_sharpener_name)
    # # 3. 연필 들어서 연필깎이로 사용
    inserting_pencil_in_sharpener(ur,target_pencil_name)
    # 4. 연필을 연필통에 넣기
    inserting_pencil_in_pencilcase(ur)
    # 5. 뚜껑 닫기
    closing_pencilcase(ur,ur2)

def placing_pencilcase(ur,ur2,target_pencilcase_name):
    if target_pencilcase_name=='SILVER_PENCILCASE':
        _ = tracking_target(ur, target_pencilcase_name, target_point=center_point['silver_pencilcase_center_point'], model_yolo=model_yolo,
                                   grasp_threshold=0.010)
    elif target_pencilcase_name=='RED_PENCILCASE':
        _ = tracking_target(ur, target_pencilcase_name, target_point=center_point['red_pencilcase_center_point'], model_yolo=model_yolo,
                                   grasp_threshold=0.010)

    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[2] -= 0.315
    ur.ur_rob.movel(tmp_pose , 1, 1, wait=False, relative=False)
    waiting_target(ur,tmp_pose, 0.0005)
    ur.setGripper(255)
    tmp_pose[2] += 0.315
    ur.ur_rob.movel(tmp_pose, 1, 1, wait=False, relative=False)
    waiting_target(ur, tmp_pose, 0.0005)
    ur.ur_rob.movej(ur_pencilcase_position, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur, ur_pencilcase_position, 0.001)
    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[2] -= 0.066
    ur.ur_rob.movel(tmp_pose, 1, 1, wait=False, relative=False)
    waiting_target(ur, tmp_pose, 0.0005)


    ##r2로 연필통 몸통잡기
    ur2.ur_rob.movej(ur2_pencilcase_start, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur2, ur2_pencilcase_start, 0.001)
    ur2.ur_rob.movej(ur2_pencilcase_position, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur2, ur2_pencilcase_position, 0.001)
    ur2.setGripper(255)
    ##r1으로 뚜꼉열기
    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[2] += 0.215
    ur.ur_rob.movel(tmp_pose, 1, 1, wait=False, relative=False)
    waiting_target(ur, tmp_pose, 0.0005)
    ur.ur_rob.movej(ur_pencilcase_lid_position, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur, ur_pencilcase_lid_position, 0.001)
    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[2] -= 0.058
    ur.ur_rob.movel(tmp_pose, 1, 1, wait=False, relative=False)
    waiting_target(ur, tmp_pose, 0.0005)

    ur.setGripper(0)
    ur2.setGripper(0)
    ur.ur_rob.movej(pose_start, acc=1, vel=1, wait=False, relative=False)
    ur2.ur_rob.movej(ur2_pencilcase_start, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur2, ur2_pencilcase_start, 0.001)
    ur2.ur_rob.movej(pose_base, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur2, pose_base, 0.001)
    waiting_joint(ur, pose_start, 0.001)

def placing_sharpener(ur,target_sharpener_name):
    _ = tracking_target(ur, target_sharpener_name, target_point=center_point['pencilsharpener_center_point'],
                        model_yolo=model_yolo,
                        grasp_threshold=0.010)
    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[2] -= 0.44
    ur.ur_rob.movel(tmp_pose, 1, 1, wait=False, relative=False)
    waiting_target(ur, tmp_pose, 0.0005)
    ur.setGripper(255)
    tmp_pose[2] += 0.44
    ur.ur_rob.movel(tmp_pose, 1, 1, wait=False, relative=False)
    waiting_target(ur, tmp_pose, 0.0005)
    ur.ur_rob.movej(ur_sharpener_position, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur, ur_sharpener_position, 0.001)
    ur.setGripper(0)
    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[2] += 0.063
    ur.ur_rob.movel(tmp_pose, 1, 1, wait=False, relative=False)
    waiting_target(ur, tmp_pose, 0.0005)
    ur.ur_rob.movej(pose_start, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur, pose_start, 0.001)

def inserting_pencil_in_sharpener(ur,target_pencil_name):
    detections, _ = YOLO(model_yolo, t1.rs)  # Calls the main function YOLO()
    target_point = tracking_target(ur, target_pencil_name, target_point=center_point['pencil_center_point'], model_yolo=model_yolo,
                                   grasp_threshold=0.010)
    ##그리퍼 회전
    [x1, y1], [w, h] = find_target(detections, target_pencil_name, target_point)  # b"HUB"
    [x2, y2], [w, h] = find_target(detections, 'ORANGE_PENCIL_TAIL', target_point)  # b"HUB"
    x_ = -(y1 - y2) * np.sin(np.pi / 2)
    y_ = (x2 - x1) * np.sin(np.pi / 2)
    if x_ == 0 and y_ > 0:
        rad = np.pi
    elif x_ == 0 and y_ < 0:
        rad = 0
    else:
        rad = math.atan2(y_, x_)
    if rad < 0:
        rad = np.pi + np.pi + rad
    rotation_task(ur, rad, -0.115)
    ur.setGripper(255)
    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[2] += 0.44  ##
    ur.ur_rob.movel(tmp_pose, 1, 1, wait=False, relative=False)
    waiting_target(ur, tmp_pose, 0.0005)
    ur.ur_rob.movej(ur_sharpener_inserting_position, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur, ur_sharpener_inserting_position, 0.001)

    ##inserting##
    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    for i in range(10):
        x_ = np.random.uniform(-0.001, 0.001)
        y_ = np.random.uniform(-0.001, 0.001)
        tmp_pose[2]-=0.01
        target_pose = tmp_pose + [x_, y_, 0, 0, 0, 0]
        ur.ur_rob.movel(target_pose,0.1,0.1, wait=False, relative=False)
    waiting_target(ur,target_pose,0.0005)
    time.sleep(2)
    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[2] = 0.3  ##
    ur.ur_rob.movel(tmp_pose, 1, 1, wait=False, relative=False)
    waiting_target(ur, tmp_pose, 0.0005)

def inserting_pencil_in_pencilcase(ur):
    ur.ur_rob.movej(ur_pencilcase_inserting_position, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur, ur_pencilcase_inserting_position, 0.001)
    ur.ur_rob.movej(ur_pencilcase_inserting_back_position, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur, ur_pencilcase_inserting_back_position, 0.001)
    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[2] = 0.12  ##
    ur.ur_rob.movel(tmp_pose, 1, 1, wait=False, relative=False)
    waiting_target(ur, tmp_pose, 0.0005)
    ur.setGripper(0)
    tmp_pose[2] += 0.1  ##
    ur.ur_rob.movel(tmp_pose, 1, 1, wait=False, relative=False)
    waiting_target(ur, tmp_pose, 0.0005)

def closing_pencilcase(ur,ur2):
    ur.ur_rob.movej(ur_pencilcase_lid_position, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur, ur_pencilcase_lid_position, 0.001)
    ur2.ur_rob.movej(ur2_pencilcase_start, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur2,ur2_pencilcase_start,0.001)
    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[2] -= 0.035
    ur.ur_rob.movel(tmp_pose, 1, 1, wait=False, relative=False)
    waiting_target(ur, tmp_pose, 0.0005)
    ur.setGripper(255)
    ur2.ur_rob.movej(ur2_pencilcase_position, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur2,ur2_pencilcase_position,0.001)
    ur2.setGripper(255)
    tmp_pose[2] += 0.3
    ur.ur_rob.movel(tmp_pose, 1, 1, wait=False, relative=False)
    waiting_target(ur, tmp_pose, 0.0005)
    ur.ur_rob.movej(ur_pencilcase_lid_back_position, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur, ur_pencilcase_lid_back_position, 0.001)
    ##inserting##
    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    for i in range(10):
        x_ = np.random.uniform(-0.001, 0.001)
        y_ = np.random.uniform(-0.001, 0.001)
        tmp_pose[2]-=0.007
        target_pose = tmp_pose + [x_, y_, 0, 0, 0, 0]
        ur.ur_rob.movel(target_pose,0.1,0.1, wait=False, relative=False)
    time.sleep(1)
    ur.setGripper(0)
    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[2] += 0.1
    ur.ur_rob.movel(tmp_pose, 1, 1, wait=False, relative=False)
    waiting_target(ur, tmp_pose, 0.0005)
    ur2.setGripper(0)
    ur.ur_rob.movej(pose_start, acc=1, vel=1, wait=False, relative=False)
    ur2.ur_rob.movej(ur2_pencilcase_start, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur2, ur2_pencilcase_start, 0.001)
    ur2.ur_rob.movej(pose_base, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur2, pose_base, 0.001)
    waiting_joint(ur, pose_start, 0.001)


def plug_multitap_task(ur,ur2,target_list):
    plug_name,multitap_name=target_list
    # 1. 멀티탭 지정 위치에 놓기
    placing_multitap(ur, multitap_name)
    # 2.plug회전해서잡기
    grasping_plug(ur,ur2,plug_name)
    # 3.삽입
    #inserting_plug(ur,multitap_name)

def placing_multitap(ur, multitap_name):
    _ = tracking_target(ur, multitap_name, target_point=center_point['pencilsharpener_center_point'],
                        model_yolo=model_yolo,
                        grasp_threshold=0.010)

    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[2]=object_z[multitap_name]
    ur.ur_rob.movel(tmp_pose, 1, 1, wait=False, relative=False)
    waiting_target(ur, tmp_pose, 0.0005)
    ur.setGripper(255)


def grasping_plug(ur,ur2,plug_name):
    detections, _ = YOLO(model_yolo, t1.rs) # Calls the main function YOLO()
    d = { l[0] : l[2] for l in detections }
    target_center = center_point['white_plug_center_point']
    target_center[0] -= (d['WHITE_PLUG'][0] - d['WHITE_PLUG_TAIL'][0]) / 2
    target_center[1] -= (d['WHITE_PLUG'][1] - d['WHITE_PLUG_TAIL'][1]) / 2

    target_point = tracking_target(ur, plug_name, target_point=target_center,
                                   model_yolo=model_yolo,
                                   grasp_threshold=0.010)

    ##그리퍼 회전
    [x1, y1], [w, h] = find_target(detections, plug_name, target_point)
    [x2, y2], [w, h] = find_target(detections, 'WHITE_PLUG_TAIL', target_point)

    x_ = -(y1 - y2) * np.sin(np.pi / 2)
    y_ = (x2 - x1) * np.sin(np.pi / 2)
    if x_ == 0 and y_ > 0:
        rad = np.pi
    elif x_ == 0 and y_ < 0:
        rad = 0
    else:
        rad = math.atan2(y_, x_)
    if rad < 0:
        rad = np.pi + np.pi + rad
    rotation_task(ur, rad)

    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[2] =object_z[plug_name]
    ur.ur_rob.movel(tmp_pose, 1, 1, wait=False, relative=False)
    waiting_target(ur, tmp_pose, 0.0005)
    ur.setGripper(255)
    ur.ur_rob.movej(ur_plug_waiting_position, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur, ur_plug_waiting_position, 0.001)

    ur2.ur_rob.movej(ur2_plug_grap_start_position, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur2, ur2_plug_grap_start_position, 0.001)
    ur2.ur_rob.movej(ur2_plug_grap_end_position, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur2, ur2_plug_grap_end_position, 0.001)
    ur2.setGripper(255)
    ur.setGripper(0)
    ur.ur_rob.movej(ur_plug_grap_position, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur, ur_plug_grap_position, 0.001)
    ur.setGripper(255)
    ur2.setGripper(0)
    ur2.ur_rob.movej(ur2_plug_grap_start_position, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur2, ur2_plug_grap_start_position, 0.001)
    ur.ur_rob.movej(pose_start, acc=1, vel=1, wait=False, relative=False)
    ur2.ur_rob.movej(pose_base, acc=1, vel=1, wait=False, relative=False)
    waiting_joint(ur2, pose_base, 0.001)
    waiting_joint(ur, pose_start, 0.001)

# def inserting_plug(ur,multitap_name):
#