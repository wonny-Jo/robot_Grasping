from torch.lib import *
from model.function import *

from device.camera.RS2 import *
from device.thread import *

from PIL import Image
from torchvision import transforms
from config import *
from math import *
import math3d as m3d

img_back = Image.open("./data/back.png")


t1 = None
t2 = None
t3 = None

def set_thread():
    global t1
    global t2
    global t3

    t1= t_rs(realsense(img_size=[1280, 720], frame=30))
    # t2= t_cam(0)

def start_thread():
    t1.start()
    t1.ON= True
    # t2.start()


def robot_control_j(rob_total_pose=None, vel=0.9, acc=0.4,
                    ur_pose=None, ur_vel=0.9, ur_acc=0.4,
                    ur2_pose=None, ur2_vel=0.9, ur2_acc=0.4):
    if rob_total_pose is not None:
        ur.ur_rob.movej(rob_total_pose, acc=acc, vel=vel, wait=False, relative=False)
        ur2.ur_rob.movej(rob_total_pose, acc=acc, vel=vel, wait=False, relative=False)
        waiting_joint(ur, rob_total_pose, 0.005)
        waiting_joint(ur2, rob_total_pose, 0.005)
    else:
        if ur_pose is not None:
            ur.ur_rob.movej(ur_pose, acc=ur_acc, vel=ur_vel, wait=False, relative=False)
        if ur2_pose is not None:
            ur2.ur_rob.movej(ur2_pose, acc=ur2_acc, vel=ur2_vel, wait=False, relative=False)
        if ur_pose is not None:
            waiting_joint(ur, ur_pose, 0.005)
        if ur2_pose is not None:
            waiting_joint(ur2, ur2_pose, 0.005)

def robot_control_l(rob_pose=None,rob=None,
                    ur_pose=None, ur_vel=0.9, ur_acc=0.4,
                    ur2_pose=None, ur2_vel=0.9, ur2_acc=0.4):
    if rob_pose is not None:
        rob.ur_rob.movel(rob_pose, acc=ur_acc, vel=ur_vel, wait=False, relative=False)
        waiting_target(rob, rob_pose, 0.005)
    else:
        if ur_pose is not None:
            ur.ur_rob.movel(ur_pose, acc=ur_acc, vel=ur_vel, wait=False, relative=False)
        if ur2_pose is not None:
            ur2.ur_rob.movel(ur2_pose, acc=ur2_acc, vel=ur2_vel, wait=False, relative=False)
        if ur_pose is not None:
            waiting_target(ur, ur_pose, 0.005)
        if ur2_pose is not None:
            waiting_target(ur2, ur2_pose, 0.005)

def robot_control_ls(ur_poses=None, ur_vel=0.8, ur_acc=0.5,
                     ur2_poses=None, ur2_vel=0.8, ur2_acc=0.5):
    ur.ur_rob.movels(ur_poses, acc=ur_acc, vel=ur_vel, radius=0.002, wait=False)
    ur2.ur_rob.movels(ur2_poses, acc=ur2_acc, vel=ur2_vel, radius=0.002, wait=False)
    waiting_target(ur,ur_poses[-1], 0.005)
    waiting_target(ur2,ur2_poses[-1], 0.005)

def solve_FK(th):
    T = np.eye(4)
    return_T = np.eye(4)
    ur5_a = [0, -0.425, -0.39225, 0, 0, 0]
    ur5_d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
    alp = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]

    for i in range(6):
        T[0, 0] = cos(th[i])
        T[0, 1] = -sin(th[i]) * cos(alp[i])
        T[0, 2] = sin(th[i]) * sin(alp[i])
        T[0, 3] = ur5_a[i] * cos(th[i])

        T[1, 0] = sin(th[i])
        T[1, 1] = cos(th[i]) * cos(alp[i])
        T[1, 2] = -cos(th[i]) * sin(alp[i])
        T[1, 3] = ur5_a[i] * sin(th[i])

        T[2, 0] = 0
        T[2, 1] = sin(alp[i])
        T[2, 2] = cos(alp[i])
        T[2, 3] = ur5_d[i]

        T[3, 0] = 0
        T[3, 1] = 0
        T[3, 2] = 0
        T[3, 3] = 1

        return_T = (return_T @ T)

    pose_vector = m3d.Transform(return_T).pose_vector
    return pose_vector


def object_grasping(rob,target_pose,target,
                    target_center_points=None,offset=[0,0],offset_after_rotation=None,not_lift=False,not_gripping=False,
                    adding_rad=0.0,adding_move=0.0,case_open=False,pushing=0.0):
    loc = rob.ur_rob.rtmon.get_all_data()['tcp']
    preloc = deepcopy(loc)
    # : 로봇의 x좌표먼저 이동
    preloc[0] = target_pose[0]
    robot_control_l(rob=rob,rob_pose=preloc)
    if case_open is True:
        temp = rob.ur_rob.rtmon.get_all_data()['tcp']
        temp[3:]=[0.01494576, -2.96527302, -0.933348]
        robot_control_l(rob=rob, rob_pose=temp)
        loc=temp
    # : 타겟 좌표로 이동
    loc[:2] = target_pose[:2]
    loc[0]+=offset[0]
    loc[1]+=offset[1]
    robot_control_l(rob=rob,rob_pose=loc)
    if target_center_points is not None:
        rad = calculate_rotating_radian_for_two_position(target_center_points)
        rad+=adding_rad
        rotation_task(rob, rad)
        if offset_after_rotation is not None:
            loc = rob.ur_rob.rtmon.get_all_data()['tcp']
            move_rad=rad+offset_after_rotation[1]
            loc[0] -= offset_after_rotation[0]*math.sin(move_rad)
            loc[1] += offset_after_rotation[0]*math.cos(move_rad)
            robot_control_l(rob=rob, rob_pose=loc)

    if not_gripping is True:
        return loc,preloc

    loc_after = rob.ur_rob.rtmon.get_all_data()['tcp']
    loc_after[2] = object_z[target]
    robot_control_l(rob=rob,rob_pose=loc_after)

    if target_center_points is not None:
        loc_after[0] += adding_move * math.cos(rad)
        loc_after[1] += adding_move * math.sin(rad)
        robot_control_l(rob=rob, rob_pose=loc_after,ur_acc=0.3,ur_vel=0.3)
        loc_after[0] -= adding_move/5 * math.cos(rad)
        loc_after[1] -= adding_move/5 * math.sin(rad)
        robot_control_l(rob=rob, rob_pose=loc_after, ur_acc=0.3, ur_vel=0.3)

    if pushing!=0.0:
        move_rad = rad -(pi/2)
        loc_after[0] -= pushing * math.sin(move_rad)
        loc_after[1] += pushing * math.cos(move_rad)
        robot_control_l(rob=rob, rob_pose=loc_after,ur_acc=0.3,ur_vel=0.3)
        rob.setGripper(190)
        time.sleep(0.5)
        loc_after[0] += pushing * math.sin(move_rad)
        loc_after[1] -= pushing * math.cos(move_rad)
        robot_control_l(rob=rob, rob_pose=loc_after,ur_acc=0.3,ur_vel=0.3)
        rob.setGripper(0)
        time.sleep(0.5)
        loc_after[0] += (pushing-0.01) * math.sin(move_rad)
        loc_after[1] -= (pushing-0.01) * math.cos(move_rad)
        robot_control_l(rob=rob, rob_pose=loc_after,ur_acc=0.3,ur_vel=0.3)

    rob.setGripper(255)
    if not_lift is False:
        robot_control_l(rob=rob,rob_pose=loc)
    return loc,preloc

def tracking_target(ur, target_name, target_point, model_yolo, obj_z = None,radian=None, task=None):
    acc = 1
    min_t = 0.1
    v_w = 0.25
    step = 0
    prev_target = None
    using_z = True
    vel_z_acc = 0.51
    # vel_z_acc=0
    # if task=='hub' or task=='plug':
    #     vel_z_acc = 0.51
    # elif task=='pencil':
    #     vel_z_acc = 0.48

    # tracking loop
    while True:
        time1 = time.time()
        detections, img = YOLO(model_yolo, t1.rs)  # Calls the main function YOLO()
        target, bbox = find_target(detections, target_name, target_point)
#target_point=[503,357]
        #인식 결과 처리
        if None == target and None != prev_target :     #인식 잘 되다가 안되는 경우, 이전 정보 덮어 씌움
            print("목표를 찾을 수 없습니다.")
            target = prev_target
            # bbox = prev_bbox
            v_w = 0.1
        elif None == target and None == prev_target:
            target=deepcopy(target_point)
            target[0]+=random.randint(-64,64)
            target[1]+=random.randint(-36,36)
            v_w = 0.1
        else:                                          # 인식 잘 되는 경우
            v_w = 0.2
            prev_target = target
            # prev_bbox = bbox

        if using_z==True and step < len(traj_approch) and None != prev_target:
            vel_z = traj_approch[step]['dtcp'][2] * vel_z_acc
            dist = 1
            step += 1
            # x,y축 속도 계산 식.
            vel = [((target_point[1] - target[1]) / 360) * v_w, ((target_point[0] - target[0]) / 640) * v_w, vel_z, 0,
                   0, 0]
        else:
            #dist = np.sqrt(((target_point[0] - target[0]) / 424) ** 2 + (-(target_point[1] - target[1]) / 240) ** 2)
            # x,y축 속도 계산 식.
            vel = [((target_point[1] - target[1]) / 360) * v_w, ((target_point[0] - target[0]) / 640) * v_w, 0, 0,
                   0, 0]

        if radian!=None:
            velx=vel[0]*math.cos(radian)-vel[1]*math.sin(radian)
            vely=vel[0]*math.sin(radian)+vel[1]*math.cos(radian)
            vel[0]=velx
            vel[1]=vely
        ur.ur_rob.speedx("speedl", vel, acc=acc, min_time=min_t)

        if using_z==True and step>=len(traj_approch):
            using_z = False
            tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
            tmp_pose[2]=-0.061103
            robot_control_l(ur_pose=tmp_pose)

        tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
        if obj_z!=None and tmp_pose[2]<=obj_z+0.05:
            using_z=False
            tmp_pose[2]=obj_z
            robot_control_l(ur_pose=tmp_pose)
            obj_z=None

        # yolo 모델 프레임 속도 맞추기 위한 대기
        while True:
            if time.time() - time1 > 0.1:#0.033:
                break

        # grasping points에 근접하면 tracking 중단.
        if (obj_z == None or tmp_pose[2]<=obj_z) and target==target_point:
            break
    return target,detections

def grasping_target(model_yolo, model_vae):
    t1.detections, img = YOLO(model_yolo, t1.rs)  # Calls the main function YOLO()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cur_img = Image.fromarray(img)
    rad = model_vae.predict(cur_img, t1.detections[0][2])
    print ("rad : " + str(rad*180/3.14))
    return t1.img.copy()


def calculate_rotating_radian_for_two_position(target_center_points):
    [x1, y1] = target_center_points[:2]
    [x2, y2] = target_center_points[2]

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

    return rad


def calculate_rotating_radian(detections,target_name,target_point):
    [x1, y1], [w, h] = find_target(detections, target_name, target_point)
    [x2, y2], [w, h] = find_target(detections, target_name + '_TAIL', target_point)

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

    return rad


def rotation_task(rob, rz,z=None):
    pose_current = rob.ur_rob.rtmon.get_all_data()['tcp']
    rot_cur = math3d.Transform(np.array([2.22, -2.22, 0]), np.array([0, 0, 0]))
    pose_next = (math3d.Orientation([0, 0, rz]) * rot_cur.orient.inverse).inverse.rotation_vector
    pose_current[3:6] = pose_next
    if z:
        pose_current[2] = z
    robot_control_l(rob=rob,rob_pose=pose_current)


def move_rotation(ur, rz, x, z):

    pose_current = ur.ur_rob.rtmon.get_all_data()['tcp']
    pose_current[3:] = [2.221, -2.221, 0]
    rot_cur = math3d.Transform(np.array(pose_current[3:6]), pose_current[:3])
    pose_next = (math3d.Orientation([0, 0, rz]) * rot_cur.orient.inverse).inverse.rotation_vector
    pose_current[3:6] = pose_next
    pose_current[2] = z

    x1 = pose_current[0] + (x) * np.cos(-rz) - (0) * np.sin(-rz)
    y1 = pose_current[1] - (x) * np.sin(-rz) - (0) * np.cos(-rz)

    pose_current[0] = x1
    pose_current[1] = y1
    robot_control_l(ur_pose=pose_current)


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

def set_obj(org_list):
    shuffled_list = copy.deepcopy(org_list)
    random.shuffle(shuffled_list)
    return shuffled_list


def task_initialize():
    ur.setGripper(0)    # 0이 오픈 / 255가 클로즈
    ur2.setGripper(0)
    robot_control_j(rob_total_pose=home)
    #ur2.rob.set_tcp([0, 0, 0.185, 0, 0, 0])
    robot_control_j(ur_pose=pose_start,ur2_pose=pose_base)
    set_thread()
    start_thread()
    ur.rob.set_tcp([0, 0, 0.153, 0, 0, 0])


def run_task(task, target,target2=None,tail=None,tail2=None):
    t1.screen_pause = True
    t1.detections.append(target)
    if target2 is not None:
        t1.detections.append(target2)
        if tail2 is not None:
            task(target,target2,tail,tail2)
        else:
            task(target, target2, tail)
    else:
        if tail is None:
            task(target)
        else:
            task(target,tail)
    t1.detections = []

def target_detection(threshold=0.4):
    # detect1,_= YOLO(model_yolo1, t1.rs, threshold=threshold) # pack in hole
    #detect3, _ = YOLO(model_yolo3, t1.rs, threshold=threshold) # picking, bottle_lid, wide_object, desk_cleaner
    # detect_bin,_=YOLO(model_bin, t1.rs, threshold=threshold)
    # detect_drawer,_ = YOLO(model_drawer, t1.rs, threshold=threshold)
    # detect_penholder,_ = YOLO(model_penholder, t1.rs, threshold=threshold)
    #detect2, _ = YOLO(model_yolo2, t1.rs, threshold=threshold)  # drawer, bin, pen holder
    #detect_hub,_=YOLO(model_hub, t1.rs, threshold=threshold)
    # detect_pencilcase,_ = YOLO(model_pencilcase, t1.rs, threshold=threshold)
    # detect_picking, _ = YOLO(model_picking, t1.rs, threshold=threshold)
    # detect_multitap,_ = YOLO(model_multitap, t1.rs, threshold=threshold)
    # detect_cwb, _ = YOLO(model_cwb, t1.rs, threshold=threshold)
    # detect_bd, _ = YOLO(model_bd, t1.rs, threshold=threshold)
    # detect_hp, _ = YOLO(model_hp, t1.rs, threshold=threshold)
    #detect_mp, _ = YOLO(model_mp, t1.rs, threshold=threshold)
    # detections=detect_bd+detect_hp+detect_cwb+detect_multitap+detect_pencilcase+detect_picking

    detect_picking, _ = YOLO(model_picking, t1.rs, threshold=threshold)
    detect_picking2, _ = YOLO(model_picking2, t1.rs, threshold=threshold)
    detect_cwb, _ = YOLO(model_cwb, t1.rs, threshold=threshold)
    detect_bd, _ = YOLO(model_bd, t1.rs, threshold=threshold)
    detect_hp, _ = YOLO(model_hp, t1.rs, threshold=threshold)
    detect_mp, _ = YOLO(model_mp, t1.rs, threshold=threshold)
    detections=detect_mp+detect_bd+detect_cwb+detect_picking+detect_picking2+detect_hp
    return detections

def task_run():
    detections=[]
    for _ in range(5):
        detections+= target_detection()
    while True:
        t1.detections=detections
        detections = []
        for _ in range(5):
            detections += target_detection()

    wide_flag=False
    while detections is not [] :
        target_lists = {}
        tail_lists = {}
        picking_lists = []
        bin_lists = []
        bin_obj_lists = []
        drawer_lists = []
        drawer_obj_lists = []
        bottle_lid_lists = []
        pen_lid_lists = []
        holder_lists = []
        wide_object_lists = []
        cleaner_lists = []
        pencil_list1s = []
        pencil_list2s = []
        multitap_list1s = []
        multitap_list2s = []

        isEnd=False
        if wide_flag is True and len(detections)<=2:
            for detection in detections:
                if detection[0] in wide_object_list:
                    isEnd=True
                    break
        if isEnd==True:
            break

        isDetected=False
        for detection in detections:
            if 'BIN' not in detection[0] and 'DRAWER' not in detection[0] and (detection[2][0]>=955 or detection[2][1]>=675):
                continue
            if ('BIN' in detection[0] or 'DRAWER' in detection[0]) and (detection[2][0]<=955 or detection[2][1]>=675):
                continue

            if detection[0] in targets:
                if detection[0] not in target_lists.keys() or float(detection[1]) > float(target_lists[detection[0]][1]):
                    target_lists[detection[0]] = detection
                if detection[0] in picking_list and detection[0] not in picking_lists:
                    picking_lists.append(detection[0])
                elif detection[0] in bin_list and detection[0] not in bin_lists:
                    bin_lists.append(detection[0])
                elif detection[0] in bin_obj_list and detection[0] not in bin_obj_lists:
                    bin_obj_lists.append(detection[0])
                elif detection[0] in drawer_list and detection[0] not in drawer_lists:
                    drawer_lists.append(detection[0])
                elif detection[0] in drawer_obj_list and detection[0] not in drawer_obj_lists:
                    drawer_obj_lists.append(detection[0])
                elif detection[0] in bottle_lid_list and detection[0] not in bottle_lid_lists:
                    bottle_lid_lists.append(detection[0])
                elif detection[0] in pen_lid_list and detection[0] not in pen_lid_lists:
                    pen_lid_lists.append(detection[0])
                elif detection[0] in holder_list and detection[0] not in holder_lists:
                    holder_lists.append(detection[0])
                elif detection[0] in wide_object_list and detection[0] not in wide_object_lists:
                    wide_object_lists.append(detection[0])
                elif detection[0] in cleaner_list and detection[0] not in cleaner_lists:
                    cleaner_lists.append(detection[0])
                elif detection[0] in pencil_list1 and detection[0] not in pencil_list1s:
                    pencil_list1s.append(detection[0])
                elif detection[0] in pencil_list2 and detection[0] not in pencil_list2s:
                    pencil_list2s.append(detection[0])
                elif detection[0] in multitap_list1 and detection[0] not in multitap_list1s:
                    multitap_list1s.append(detection[0])
                elif detection[0] in multitap_list2 and detection[0] not in multitap_list2s:
                    multitap_list2s.append(detection[0])
            elif detection[0] in tails:
                if detection[0] not in tail_lists.keys() or float(detection[1]) > float(tail_lists[detection[0]][1]):
                    tail_lists[detection[0]] = detection

        if bottle_lid_lists != []:
            run_task(bottle_lid_task, target_lists[bottle_lid_lists[0]])
            targets.remove(bottle_lid_lists[0])
            isDetected=True
            robot_control_j(rob_total_pose=home)

        if set(pencil_list1) == set(pencil_list1s):
            task_flag = True
            for obj in pencil_list1:
                obj_tail = obj + '_TAIL'
                if obj_tail in tails and obj_tail not in tail_lists.keys():
                    task_flag = False
                    break
            if task_flag == False:
                continue
            target_detections = []
            for obj in pencil_list1:
                obj_tail = obj + '_TAIL'
                t1.detections.append(target_lists[obj])
                target_detections.append(target_lists[obj])
                if obj_tail in tails:
                    target_detections.append(tail_lists[obj_tail])
            t1.screen_pause = True
            pencil_task(target_detections)
            t1.detections = []
            robot_control_j(rob_total_pose=home)
            isDetected = True

        if set(pencil_list2) == set(pencil_list2s):
            task_flag = True
            for obj in pencil_list2:
                obj_tail = obj + '_TAIL'
                if obj_tail in tails and obj_tail not in tail_lists.keys():
                    task_flag = False
                    break
            if task_flag == False:
                continue
            target_detections = []
            for obj in pencil_list2:
                obj_tail = obj + '_TAIL'
                t1.detections.append(target_lists[obj])
                target_detections.append(target_lists[obj])
                if obj_tail in tails:
                    target_detections.append(tail_lists[obj_tail])
            t1.screen_pause = True
            pencil_task(target_detections)
            t1.detections = []
            robot_control_j(rob_total_pose=home)
            isDetected = True

        if holder_lists != [] and pen_lid_lists != []:
            i = 0
            holder = holder_lists[0]
            holder_tail = holder + '_TAIL'
            if holder_tail in tails and holder_tail not in tail_lists.keys():
                continue
            while i < len(pen_lid_lists):
                obj = pen_lid_lists[i]
                obj_tail = obj + '_TAIL'
                if obj_tail in tail_lists.keys():
                    if holder_tail in tails:
                        run_task(pen_lid_holder_task,target_lists[obj],target_lists[holder_lists[0]],
                                 tail_lists[obj_tail],tail_lists[holder_tail])
                    else:
                        run_task(pen_lid_holder_task, target_lists[obj], target_lists[holder_lists[0]],
                                 tail_lists[obj_tail])
                    targets.remove(holder_lists[0])
                    targets.remove(obj)
                    isDetected = True
                i += 1
            if isDetected:
                robot_control_j(rob_total_pose=home)

        if picking_lists != []:
            i = 0
            while i < len(picking_lists):
                obj = picking_lists[i]
                obj_tail = obj + '_TAIL'
                if obj_tail not in tails:
                    run_task(picking_task, target_lists[obj])
                    targets.remove(obj)
                    isDetected = True
                elif obj_tail in tail_lists.keys():
                    run_task(picking_task, target_lists[obj],tail=tail_lists[obj_tail])
                    targets.remove(obj)
                    isDetected = True
                i += 1
            if isDetected:
                robot_control_j(rob_total_pose=home)

        if bin_lists != [] and bin_obj_lists != []:
            i = 0
            while i < len(bin_obj_lists):
                obj = bin_obj_lists[i]
                obj_tail = obj + '_TAIL'
                if obj_tail in tail_lists.keys():
                    run_task(bin_task, target_lists[obj],target_lists[bin_lists[0]],tail_lists[obj_tail])
                    targets.remove(obj)
                    targets.remove(bin_lists[0])
                    isDetected = True
                i += 1
            if isDetected:
                robot_control_j(rob_total_pose=home)

        if set(multitap_list1) == set(multitap_list1s):
            task_flag = True
            for obj in multitap_list1:
                obj_tail = obj + '_TAIL'
                if obj_tail not in tail_lists.keys():
                    task_flag = False
                    break
            if task_flag == False:
                continue
            target_detections = []
            for obj in multitap_list1:
                obj_tail = obj + '_TAIL'
                t1.detections.append(target_lists[obj])
                target_detections.append(target_lists[obj])
                target_detections.append(tail_lists[obj_tail])
            t1.screen_pause = True
            plug_multitap_task(target_detections)
            t1.detections = []
            robot_control_j(rob_total_pose=home)
            isDetected = True

        if set(multitap_list2) == set(multitap_list2s):
            task_flag = True
            for obj in multitap_list2:
                obj_tail = obj + '_TAIL'
                if obj_tail not in tail_lists.keys():
                    task_flag = False
                    break
            if task_flag == False:
                continue
            target_detections = []
            for obj in multitap_list2:
                obj_tail = obj + '_TAIL'
                t1.detections.append(target_lists[obj])
                target_detections.append(target_lists[obj])
                target_detections.append(tail_lists[obj_tail])
            t1.screen_pause = True
            plug_multitap_task(target_detections)
            t1.detections = []
            robot_control_j(rob_total_pose=home)
            isDetected = True

        if 'USB_C' in target_lists.keys() or 'HDMI' in target_lists.keys() :
            if 'USB_C' in target_lists.keys():
                t1.detections.append(target_lists['USB_C'])
                if 'USB_C_HUB' in target_lists.keys():
                    t1.detections.append(target_lists['USB_C_HUB'])
                t1.screen_pause = True
                hub_cable_task('USB_C','USB_C_HUB')
                target_lists.pop('USB_C')
            else:
                t1.detections.append(target_lists['HDMI'])
                if 'HDMI_HUB' in target_lists.keys():
                    t1.detections.append(target_lists['HDMI_HUB'])
                t1.screen_pause = True
                hub_cable_task('HDMI', 'HDMI_HUB')
                target_lists.pop('HDMI')
            t1.detections = []
            robot_control_j(rob_total_pose=home)
            isDetected = True

        if cleaner_lists != []:
            obj = cleaner_lists[0]
            obj_tail = obj + '_TAIL'
            if obj_tail in tail_lists.keys():
                run_task(desk_cleaner_task, target_lists[obj], tail=tail_lists[obj_tail])
                targets.remove(obj)
                isDetected = True
            if isDetected:
                robot_control_j(rob_total_pose=home)

        if drawer_lists != [] and drawer_obj_lists != []:
            i = 0
            while i < len(drawer_obj_lists):
                obj = drawer_obj_lists[i]
                obj_tail = obj + '_TAIL'
                if obj_tail in tail_lists.keys():
                    run_task(drawer_task, target_lists[obj], target_lists[drawer_lists[0]], tail_lists[obj_tail])
                    targets.remove(obj)
                    targets.remove(drawer_lists[0])
                    isDetected = True
                i += 1
            if isDetected:
                robot_control_j(rob_total_pose=home)

        if wide_object_lists != []:
            obj = wide_object_lists[0]
            obj_tail = obj + '_TAIL'
            if obj_tail in tail_lists.keys():
                t1.detections.append(tail_lists[obj_tail])
                run_task(wide_object_task,target_lists[obj],tail=tail_lists[obj_tail])
                targets.remove(obj)
                tails.remove(obj_tail)
                isDetected = True
                wide_flag=True
            if isDetected:
                robot_control_j(rob_total_pose=home)

        if isDetected:
            robot_control_j(ur2_pose=pose_base)
            robot_control_j(ur_pose=pose_start)
        detections= target_detection()
        t1.detections = []
        t1.screen_pause = False

def picking_task(picking_detection,picking_tail_detection=None):
    list1=['RED_CUP', 'PINK_CUP',  'GREY_CUP', 'GREEN_CUP', 'BLUE_CUP', 'PURPLE_CUP', 'SILVER_CUP',
           'GREEN_BOX', 'PINK_BOX']
    list2=['PINK_STAPLER', 'STAN_STAPLER', 'BLACK_TAPE', 'WHITE_TAPE','SMALL_BOX', 'GLUE_PEN', 'GLUE_STICK',
           'WHITE_BOX','RED_BOX']
    list3=['YELLOW_BOX','BIG_BOX']
    mean_xy = [picking_detection[2][1], picking_detection[2][0]]
    if picking_detection[0] in list1:
        obj='cup'
        offset = np.array(offsets[obj])
        offset += np.array(offsets[obj + '_left']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][2]) if mean_xy[1] < pixel_point[obj][0] else np.array(
            offsets[obj + '_right']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][3])
        offset += np.array(offsets[obj + '_up']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][4]) if mean_xy[0] < pixel_point[obj][1] else np.array(
            offsets[obj + '_down']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][5])
    elif picking_detection[0] in list2:
        obj='glue'
        offset = np.array(offsets[obj])
        offset += np.array(offsets[obj + '_left']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][2]) if mean_xy[1] < pixel_point[obj][0] else np.array(
            offsets[obj + '_right']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][3])
        offset += np.array(offsets[obj + '_up']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][4]) if mean_xy[0] < pixel_point[obj][1] else np.array(
            offsets[obj + '_down']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][5])
    elif picking_detection[0] in list3:
        obj='box'
        offset = np.array(offsets[obj])
        # offset += np.array(offsets[obj + '_left']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][2]) if mean_xy[1] < pixel_point[obj][0] else np.array(
        #     offsets[obj + '_right']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][3])
        # offset += np.array(offsets[obj + '_up']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][4]) if mean_xy[0] < pixel_point[obj][1] else np.array(
        #     offsets[obj + '_down']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][5])
    target_pose = t1.rs.pxl2xyz(mean_xy,depth_w[obj])
    offset=list(offset)

    robot_control_j(ur_pose=ur_back_pose,ur2_pose=ur2_starting_pose)
    if picking_tail_detection is not None:
        target_center_points=list(picking_detection[2][:2])
        target_center_points.append(list(picking_tail_detection[2][:2]))
        _, ur2_preloc = object_grasping(rob=ur2,target_pose=target_pose,target=picking_detection[0], target_center_points=target_center_points,
                                        offset=offset)
    else:
        _, ur2_preloc = object_grasping(rob=ur2, target_pose=target_pose,target=picking_detection[0],
                                        offset=offset)
    robot_control_l(ur2_pose=ur2_preloc)
    robot_control_j(ur2_pose=ur2_starting_pose)
    robot_control_j(ur2_pose=ur2_placing_pose)
    ur2.setGripper(0)
    robot_control_j(ur2_pose=ur2_starting_pose)


def bin_task(bin_obj_detection,bin_detection,bin_obj_tail_detection):
    mean_xy = [bin_obj_detection[2][1], bin_obj_detection[2][0]]
    if bin_obj_detection[0] == 'MILK':
        obj='box'
        offset = np.array(offsets[obj])
        offset += np.array(offsets[obj + '_left']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][2]) if mean_xy[1] < pixel_point[obj][0] else np.array(
            offsets[obj + '_right']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][3])
        offset += np.array(offsets[obj + '_up']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][4]) if mean_xy[0] < pixel_point[obj][1] else np.array(
            offsets[obj + '_down']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][5])
    elif bin_obj_detection[0] == 'YOGURT':
        obj='glue'
        offset = np.array(offsets[obj])
        offset += np.array(offsets[obj + '_left']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][2]) if mean_xy[1] < pixel_point[obj][0] else np.array(
            offsets[obj + '_right']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][3])
        offset += np.array(offsets[obj + '_up']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][4]) if mean_xy[0] < pixel_point[obj][1] else np.array(
            offsets[obj + '_down']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][5])
        offset[1]+=0.014592333
    offset=-offset+np.array(offsets['bin_ur_offset'])
    offset=list(offset)
    target_pose = t1.rs.pxl2xyz(mean_xy,depth_w[obj])
    target_pose[:2] = -target_pose[:2]
    robot_control_j(ur_pose=ur_starting_pose,ur2_pose=ur2_back_pose)
    target_center_points = list(bin_obj_detection[2][:2])
    target_center_points.append(list(bin_obj_tail_detection[2][:2]))
    ur_loc,ur_preloc = object_grasping(rob=ur, target_pose=target_pose, target=bin_obj_detection[0],offset = offset,
                    target_center_points=target_center_points)

    mean_xy = [bin_detection[2][1], bin_detection[2][0]]
    if bin_detection[0] == 'WHITE_BIN':
        obj='bottle'
        offset = np.array(offsets[obj])
        offset += np.array(offsets[obj + '_left']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][2]) if mean_xy[1] < pixel_point[obj][0] else np.array(
            offsets[obj + '_right']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][3])
        offset += np.array(offsets[obj + '_up']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][4]) if mean_xy[0] < pixel_point[obj][1] else np.array(
            offsets[obj + '_down']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][5])
    elif bin_detection[0] == 'STAN_BIN':
        obj='holder'
        offset = np.array(offsets[obj])
        offset += np.array(offsets[obj + '_left']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][2]) if mean_xy[1] < pixel_point[obj][0] else np.array(
            offsets[obj + '_right']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][3])
        offset += np.array(offsets[obj + '_up']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][4]) if mean_xy[0] < pixel_point[obj][1] else np.array(
            offsets[obj + '_down']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][5])
    offset=-offset+np.array(offsets['ur_offset'])
    offset=list(offset)
    target_pose = t1.rs.pxl2xyz(mean_xy,depth_w[obj])
    target_pose[:2] = -target_pose[:2]
    ur_loc[0] = target_pose[0]+offset[0] #0.04
    ur_loc[2] += 0.1
    robot_control_l(ur_pose=ur_loc)
    ur_loc[1] = target_pose[1] + offset[1] #0.19
    robot_control_l(ur_pose=ur_loc)
    if bin_detection[0] == 'WHITE_BIN':
        temp_deg=np.rad2deg(ur.ur_rob.getj())
        temp_deg[5]-=90
        robot_control_j(ur_pose=np.deg2rad(temp_deg))

    if bin_obj_detection[0] == 'YOGURT':
        ur.setGripper(115)
    ur.setGripper(0)
    robot_control_j(rob_total_pose=home)


def drawer_task(drawer_obj_detection, drawer_detection, drawer_obj_tail_detection):
    mean_xy = [drawer_obj_detection[2][1], drawer_obj_detection[2][0]]
    obj='glue'
    offset = np.array(offsets[obj])
    offset += np.array(offsets[obj + '_left']) * (pixel_point[obj][0] - mean_xy[1]) / (
                pixel_point[obj][0] - pixel_point[obj][2]) if mean_xy[1] < pixel_point[obj][0] else np.array(
        offsets[obj + '_right']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][3])
    offset += np.array(offsets[obj + '_up']) * (pixel_point[obj][1] - mean_xy[0]) / (
                pixel_point[obj][1] - pixel_point[obj][4]) if mean_xy[0] < pixel_point[obj][1] else np.array(
        offsets[obj + '_down']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][5])
    drawer_obj_target_pose = t1.rs.pxl2xyz(mean_xy,depth_w[obj])
    offset=list(offset)
    #grasp_placing_drawer_obj
    robot_control_j(rob_total_pose=home)
    robot_control_j(ur_pose=ur_back_pose, ur2_pose=ur2_starting_pose)
    target_center_points = list(drawer_obj_detection[2][:2])
    target_center_points.append(list(drawer_obj_tail_detection[2][:2]))
    _ = object_grasping(rob=ur2, target_pose=drawer_obj_target_pose, target=drawer_obj_detection[0],target_center_points=target_center_points,offset=offset)

    #open_drawer
    mean_xy = [drawer_detection[2][1], drawer_detection[2][0]]
    obj='holder'
    offset = np.array(offsets[obj])
    offset += np.array(offsets[obj + '_left']) * (pixel_point[obj][0] - mean_xy[1]) / (
                pixel_point[obj][0] - pixel_point[obj][2]) if mean_xy[1] < pixel_point[obj][0] else np.array(
        offsets[obj + '_right']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][3])
    offset += np.array(offsets[obj + '_up']) * (pixel_point[obj][1] - mean_xy[0]) / (
                pixel_point[obj][1] - pixel_point[obj][4]) if mean_xy[0] < pixel_point[obj][1] else np.array(
        offsets[obj + '_down']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][5])
    drawer_offset=-offset+offsets['ur_offset']
    offset=list(offset)
    drawer_offset=list(drawer_offset)
    drawer_target_pose = t1.rs.pxl2xyz(mean_xy,depth_w[obj])

    robot_control_j(rob_total_pose=home)
    robot_control_j(ur_pose=ur_starting_pose, ur2_pose=ur2_back_pose)
    ur.setGripper(80)
    robot_control_j(ur_pose=ur_move_path_j1)
    ur_loc = ur.ur_rob.rtmon.get_all_data()['tcp']
    ur_loc[0] = -drawer_target_pose[0]+drawer_offset[0]
    robot_control_l(ur_pose=ur_loc)
    ur.setGripper(255)
    ur_loc[1] = ur_loc[1] + 0.055
    robot_control_l(ur_pose=ur_loc,ur_vel=0.3,ur_acc=0.3)
    ur.setGripper(0)

    robot_control_j(ur_pose=ur_move_path_j2)
    robot_control_j(ur_pose=ur_starting_pose)

    #grasp_place_drawer_obj
    robot_control_j(rob_total_pose=home)
    robot_control_j(ur_pose=ur_back_pose, ur2_pose=ur2_starting_pose)

    ur2_loc = ur2.ur_rob.rtmon.get_all_data()['tcp']
    ur2_loc[0] = drawer_target_pose[0]+offset[0]
    robot_control_l(ur2_pose=ur2_loc)
    temp_loc = copy.deepcopy(ur2_loc)
    ur2_loc[1] = drawer_target_pose[1]+offset[1]-0.01859175
    robot_control_l(ur2_pose=ur2_loc)
    temp_deg = np.rad2deg(ur2.ur_rob.getj())
    temp_deg[5] += 90
    robot_control_j(ur2_pose=np.deg2rad(temp_deg))
    ur2.setGripper(0)
    robot_control_l(ur2_pose=temp_loc)

    #close_drawer
    robot_control_j(rob_total_pose=home)
    robot_control_j(ur_pose=ur_starting_pose, ur2_pose=ur2_back_pose)
    robot_control_j(ur_pose=ur_move_path_j)
    ur_loc = ur.ur_rob.rtmon.get_all_data()['tcp']
    ur_loc[0] = -drawer_target_pose[0]+drawer_offset[0]
    robot_control_l(ur_pose=ur_loc)
    ur_loc[1] = -drawer_target_pose[1]+drawer_offset[1] #+ 0.1975 #0.17
    robot_control_l(ur_pose=ur_loc)
    robot_control_j(ur_pose=ur_move_path_j)
    robot_control_j(ur_pose=ur_starting_pose)
    robot_control_j(rob_total_pose=home)


def bottle_lid_task(bottle_detection):
    mean_xy = [bottle_detection[2][1], bottle_detection[2][0]]
    obj='bottle'
    target_pose = t1.rs.pxl2xyz(mean_xy,depth_w[obj])
    robot_control_j(ur_pose=ur_back_pose, ur2_pose=ur2_starting_pose)
    offset=np.array(offsets[obj])
    offset += np.array(offsets[obj + '_left']) * (pixel_point[obj][0] - mean_xy[1]) / (
                pixel_point[obj][0] - pixel_point[obj][2]) if mean_xy[1] < pixel_point[obj][0] else np.array(
        offsets[obj + '_right']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][3])
    offset += np.array(offsets[obj + '_up']) * (pixel_point[obj][1] - mean_xy[0]) / (
                pixel_point[obj][1] - pixel_point[obj][4]) if mean_xy[0] < pixel_point[obj][1] else np.array(
        offsets[obj + '_down']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][5])
    offset=list(offset)
    _ = object_grasping(rob=ur2, target_pose=target_pose, target=bottle_detection[0],offset=offset)
    robot_control_j(ur2_pose=ur2_bottle_center_position)
    robot_control_j(ur_pose=ur_bottle_back_position)
    robot_control_j(ur_pose=ur_bottle_holding_position)
    ur.setGripper(255)

    robot_control_j(ur2_pose=ur2_bottle_lid_open_joint)
    ur2_loc = ur2.ur_rob.rtmon.get_all_data()['tcp']
    ur2_loc[2] += 0.05
    robot_control_l(ur2_pose=ur2_loc)
    time.sleep(3.0)
    ur2_loc[2] -= 0.05
    robot_control_l(ur2_pose=ur2_loc)

    robot_control_j(ur2_pose=ur2_bottle_lid_open_joint)
    robot_control_j(ur2_pose=ur2_bottle_center_position)
    ur.setGripper(0)
    robot_control_j(ur_pose=ur_bottle_back_position)

    robot_control_j(ur_pose=home,ur2_pose=ur2_back_path1)
    robot_control_j(ur2_pose=ur2_back_path2)
    ur2.setGripper(0)


def pen_lid_holder_task(pen_detection, holder_detection,pen_tail_detection, holder_tail_detection=None):#fixing
    # pen lid task
    mean_xy = [pen_detection[2][1], pen_detection[2][0]]
    obj='glue'
    offset=np.array(offsets[obj])
    offset += np.array(offsets[obj + '_left']) * (pixel_point[obj][0] - mean_xy[1]) / (
                pixel_point[obj][0] - pixel_point[obj][2]) if mean_xy[1] < pixel_point[obj][0] else np.array(
        offsets[obj + '_right']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][3])
    offset += np.array(offsets[obj + '_up']) * (pixel_point[obj][1] - mean_xy[0]) / (
                pixel_point[obj][1] - pixel_point[obj][4]) if mean_xy[0] < pixel_point[obj][1] else np.array(
        offsets[obj + '_down']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][5])
    offset=list(offset)
    pen_target_pose = t1.rs.pxl2xyz(mean_xy,depth_w[obj])
    robot_control_j(ur_pose=ur_back_pose,ur2_pose=ur2_starting_pose)
    target_center_points = list(pen_detection[2][:2])
    target_center_points.append(list(pen_tail_detection[2][:2]))
    _ = object_grasping(rob=ur2, target_pose=pen_target_pose, target=pen_detection[0],
                        target_center_points=target_center_points,offset=offset)

    robot_control_j(ur_pose=home)
    robot_control_j(ur_pose=ur_starting_pose,ur2_pose=ur2_pen_lid_position1)
    robot_control_j(ur_pose=ur_pen_lid_position)
    ur.setGripper(255)
    time.sleep(0.5)
    ur2.setGripper(0)
    robot_control_j(ur2_pose=ur2_pen_lid_position2)
    if 'MARKER' in pen_detection[0]:
        ur2_loc = ur2.ur_rob.rtmon.get_all_data()['tcp']
        ur2_loc[2] -= 0.01
        robot_control_l(ur2_pose=ur2_loc)
    ur2.setGripper(255)

    ur2_loc = ur2.ur_rob.rtmon.get_all_data()['tcp']
    ur2_loc[2] = ur2_loc[2] + 0.05
    robot_control_l(ur2_pose=ur2_loc)
    time.sleep(2.0)
    ur2_loc[2] = ur2_loc[2] - 0.054
    robot_control_l(ur2_pose=ur2_loc)
    ur2.setGripper(0)
    ur_loc = ur.ur_rob.rtmon.get_all_data()['tcp']
    ur_loc[2] = ur_loc[2] - 0.05
    robot_control_l(ur_pose=ur_loc)

    #pen holder task
    #holder 잡기
    mean_xy = [holder_detection[2][1], holder_detection[2][0]]
    obj='holder'
    offset=np.array(offsets[obj])
    offset += np.array(offsets[obj + '_left']) * (pixel_point[obj][0] - mean_xy[1]) / (
                pixel_point[obj][0] - pixel_point[obj][2]) if mean_xy[1] < pixel_point[obj][0] else np.array(
        offsets[obj + '_right']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][3])
    offset += np.array(offsets[obj + '_up']) * (pixel_point[obj][1] - mean_xy[0]) / (
                pixel_point[obj][1] - pixel_point[obj][4]) if mean_xy[0] < pixel_point[obj][1] else np.array(
        offsets[obj + '_down']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][5])
    offset=list(offset)
    holder_target_pose = t1.rs.pxl2xyz(mean_xy,depth_w[obj])
    robot_control_j(rob_total_pose=home)
    robot_control_j(ur_pose=ur_back_pose,ur2_pose=ur2_starting_pose)
    if holder_tail_detection is None:
        offset[0]+=0.0354234
        _ = object_grasping(rob=ur2, target_pose=holder_target_pose, target=holder_detection[0],offset=offset)
    else:
        target_center_points = list(holder_detection[2][:2])
        target_center_points.append(list(holder_tail_detection[2][:2]))
        _ = object_grasping(rob=ur2, target_pose=holder_target_pose, target=holder_detection[0],
                            target_center_points=target_center_points,offset=offset,
                            offset_after_rotation=[0.05,0],adding_move=0.05)

    robot_control_j(ur2_pose=ur2_holder_starting_pose)
    robot_control_j(ur2_pose=ur2_holder_setting_pose)
    if holder_tail_detection is None:
        robot_control_j(ur2_pose=ur2_holder_setting_pose2_without_tail)
    else:
        robot_control_j(ur2_pose=ur2_holder_setting_pose2_with_tail)
    ur2.setGripper(200)
    time.sleep(0.5)
    ur2.setGripper(150)
    ur2_loc = ur2.ur_rob.rtmon.get_all_data()['tcp']
    ur2_loc[2] += 0.1
    robot_control_l(ur2_pose=ur2_loc)
    ur2.setGripper(0)
    robot_control_j(ur2_pose=ur2_holder_setting_pose3)
    ur2_loc = ur2.ur_rob.rtmon.get_all_data()['tcp']
    ur2_loc[2] -= 0.01
    robot_control_l(ur2_pose=ur2_loc)
    robot_control_j(ur2_pose=ur2_holder_setting_pose4)
    ur2_loc = ur2.ur_rob.rtmon.get_all_data()['tcp']
    ur2_loc[1] += 0.01
    if holder_tail_detection is None:
        ur2_loc[2]-= 0.01
    robot_control_l(ur2_pose=ur2_loc)
    ur2.setGripper(255)
    ur2_loc = ur2.ur_rob.rtmon.get_all_data()['tcp']
    ur2_loc[2] += 0.15
    robot_control_l(ur2_pose=ur2_loc)
    # 펜 홀더에 넣기
    robot_control_j(ur_pose=ur_holder_insert_pose)
    robot_control_j(ur_pose=ur_holder_insert_pose2,ur2_pose=ur2_holder_insert_pose)
    ur_loc = ur.ur_rob.rtmon.get_all_data()['tcp']
    ur_loc[2] -= 0.1
    robot_control_l(ur_pose=ur_loc)
    ur.setGripper(0)
    ur_loc[2] += 0.15
    robot_control_l(ur_pose=ur_loc)
    # 홀더 위치시키기
    robot_control_j(ur_pose=home,ur2_pose=ur2_holder_back_pose)
    if holder_tail_detection is None:
        robot_control_j(ur_pose=ur_back_pose,ur2_pose=ur2_holder_back_pose2_without_tail)
    else:
        robot_control_j(ur_pose=ur_back_pose, ur2_pose=ur2_holder_back_pose2_with_tail)
    ur2.setGripper(0)
    robot_control_j(ur2_pose=ur2_holder_back_pose)
    robot_control_j(ur2_pose=ur2_starting_pose)

def wide_object_task(wide_object_detection, wide_object_tail_detection):
    target_center_points = list(wide_object_detection[2][:2])
    target_center_points.append(list(wide_object_tail_detection[2][:2]))
    rad = calculate_rotating_radian_for_two_position(target_center_points)
    if 'KEYBOARD' in wide_object_detection[0]:
        rad+=pi/2
        if rad>2*pi:
            rad-=2*pi
    mean_xy = [wide_object_detection[2][1], wide_object_detection[2][0]]
    target_pose = t1.rs.pxl2xyz(mean_xy,depth_w['glue'])
    target_pose+=[-0.001266162+0.004864936, -0.0305185568,0]#+0.004864936,-0.0230779687
    close_w=object_w[wide_object_detection[0]]
    w=close_w+0.03
    obj_pos1=target_pose+[w* cos(rad),w* sin(rad),0]
    obj_pos1_close=target_pose+[close_w* cos(rad),close_w* sin(rad),0]
    obj_pos2=target_pose-[w* cos(rad),w* sin(rad),0]
    obj_pos2_close=target_pose-[close_w* cos(rad),close_w* sin(rad),0]

    if obj_pos1[1] < obj_pos2[1]:
        obj_pos1,obj_pos2=obj_pos2,obj_pos1
        obj_pos1_close,obj_pos2_close=obj_pos2_close,obj_pos1_close

    if obj_pos1[0] < obj_pos2[0]:
        prior = 2
    else:
        prior = 1

    goal1 = np.array(obj_pos1)
    goal1[:2]=-goal1[:2]    # offset = [-0.0028, 0.021]  # -0.035,0.015
    goal1+=[-0.002841, 0.192114,0]    # offset = [0.004, 0.193]0.1953
    close1 = np.array(obj_pos1_close)
    close1[:2]=-close1[:2]
    close1 += [-0.002841, 0.192114,0]
    goal2 = np.array(obj_pos2)
    close2 = np.array(obj_pos2_close)

    robot_control_j(ur_pose=ur_initial_pose, ur2_pose=ur2_initial_pose)

    ur_loc = ur.ur_rob.rtmon.get_all_data()['tcp']
    ur_loc[:2]=goal1[:2]
    ur_loc[2]=object_z['z_lift']
    ur2_loc = ur2.ur_rob.rtmon.get_all_data()['tcp']
    ur2_loc[:2]=goal2[:2]
    ur2_loc[2]=object_z['z_lift']-0.03226

    if prior == 1:
        robot_control_l(ur_pose=ur_loc)
        robot_control_l(ur2_pose=ur2_loc)
    else:
        robot_control_l(ur2_pose=ur2_loc)
        robot_control_l(ur_pose=ur_loc)

    rotation_task(ur,rad)
    rotation_task(ur2, rad)
    ur.setGripper(80)
    ur2.setGripper(80)
    ur_loc = ur.ur_rob.rtmon.get_all_data()['tcp']
    ur2_loc = ur2.ur_rob.rtmon.get_all_data()['tcp']
    ur_loc[2] = - 0.11635
    ur2_loc[2] = - 0.14861
    robot_control_l(ur_pose=ur_loc,ur2_pose=ur2_loc)
    ur_loc[:2] = close1[:2]
    ur2_loc[:2] = close2[:2]
    # = may the Force be with you
    robot_control_l(ur_pose=ur_loc,ur_vel=0.01,ur_acc=0.005, ur2_pose=ur2_loc,ur2_vel=0.01, ur2_acc=0.005)

    # ur_loc = ur.ur_rob.rtmon.get_all_data()['tcp']
    # ur2_loc = ur2.ur_rob.rtmon.get_all_data()['tcp']
    # xy_pos1 = np.array([ur_loc[0], ur_loc[1]])
    # xy_pos2 = np.array([ur2_loc[0], ur2_loc[1]])
    # ur_loc[:2] = xy_pos1 + [cos(rad) * - 0.0014,
    #                      sin(rad) * - 0.0014]
    # ur2_loc[:2] = xy_pos2 + [cos(rad) * - 0.0014,
    #                      sin(rad) * - 0.0014]
    # robot_control_l(ur_pose=ur_loc, ur2_pose=ur2_loc)
    # ur_loc = ur.ur_rob.rtmon.get_all_data()['tcp']
    # ur2_loc = ur2.ur_rob.rtmon.get_all_data()['tcp']
    ur_loc[2] +=0.015
    ur2_loc[2] +=0.015
    robot_control_l(ur_pose=ur_loc, ur_vel=0.1, ur_acc=0.1, ur2_pose=ur2_loc, ur2_vel=0.1, ur2_acc=0.1)

    # 회전시키기
    rob_pos1 = ur.ur_rob.rtmon.get_all_data()['tcp']
    rob_pos2 = ur2.ur_rob.rtmon.get_all_data()['tcp']
    rob_pos1_j=ur.ur_rob.getj()
    rob_pos2_j=ur2.ur_rob.getj()
    rob_xy1 = [-rob_pos1[0], -rob_pos1[1] + 0.186]
    rob_xy2 = [rob_pos2[0], rob_pos2[1]]
    center = np.array([(rob_xy1[0] + rob_xy2[0]) / 2, (rob_xy1[1] + rob_xy2[1]) / 2])
    width = sqrt((rob_xy1[0] - rob_xy2[0]) ** 2 + (rob_xy1[1] - rob_xy2[1]) ** 2)

    #         위 0
    #  -90 왼 ㅇ 오 90
    #         아
    obj_angle=np.rad2deg(rad)
    if obj_angle<180:
        obj_angle+=180
    if 270 <= obj_angle:
        # angle_list = [angle for angle in range(int(obj_angle), - 90, -1)]
        angle_list = [angle for angle in range(int(obj_angle), 270, -1)]
    else:
        # angle_list = [angle for angle in range(int(obj_angle), - 90, 1)]
        angle_list = [angle for angle in range(int(obj_angle), 270, 1)]
    angle_list.append(270)

    move_list1 = []
    move_list2 = []
    for angle in angle_list:
        rob_pos1_mod = rob_pos1.copy()
        rob_pos2_mod = rob_pos2.copy()
        angle_rad = np.deg2rad(angle)
        rot_xy1 = center - [cos(angle_rad) * (width / 2),
                            sin(angle_rad) * (width / 2)]
        rot_xy1[:2]=-rot_xy1[:2]
        rot_xy1[1]+=0.186
        rot_xy2 = center + [cos(angle_rad) * (width / 2),
                            sin(angle_rad) * (width / 2)]
        rob_pos1_mod[:2] = rot_xy1
        rob_pos2_mod[:2] = rot_xy2

        rot = np.deg2rad(obj_angle - angle)
        rot1 = np.append(rob_pos1_j[:-1], rob_pos1_j[-1] + rot)
        rot2 = np.append(rob_pos2_j[:-1], rob_pos2_j[-1] + rot)
        rot_pose1 = solve_FK(rot1)
        rot_pose2 = solve_FK(rot2)
        rob_pos1_mod[3:] = rot_pose1[3:]
        rob_pos2_mod[3:] = rot_pose2[3:]
        #robot_control_l(ur_pose=rob_pos1_mod,ur_acc=0.5,ur_vel=0.8,ur2_pose=rob_pos2_mod,ur2_acc=0.5,ur2_vel=0.8)
        move_list1.append(rob_pos1_mod)
        move_list2.append(rob_pos2_mod)

    robot_control_ls(ur_poses=move_list1,ur2_poses=move_list2)
    time.sleep(0.5)

    rob_pos1 = ur.ur_rob.rtmon.get_all_data()['tcp']
    rob_pos2 = ur2.ur_rob.rtmon.get_all_data()['tcp']
    goal_position=np.array(goal_positions[wide_object_detection[0]])
    goal_angle_rad = np.deg2rad(270.0)

    rot_xy1 = goal_position - [cos(goal_angle_rad) * (width / 2),
                               sin(goal_angle_rad) * (width / 2)]
    rot_xy1[:2] = -rot_xy1[:2]
    rot_xy1[1] += 0.186
    rot_xy2 = goal_position + [cos(goal_angle_rad) * (width / 2),
                               sin(goal_angle_rad) * (width / 2)]
    rob_pos1[:2] = rot_xy1
    rob_pos2[:2] = rot_xy2
    robot_control_l(ur_pose=rob_pos1,ur_vel=0.1, ur_acc=0.1, ur2_pose=rob_pos2, ur2_vel=0.1, ur2_acc=0.1)

    rob_pos1[2] = - 0.11635
    rob_pos2[2] =  - 0.14
    robot_control_l(ur_pose=rob_pos1, ur2_pose=rob_pos2)

    rot_xy1_open = goal_position - [cos(goal_angle_rad) * (width / 2 + 0.02),
                                    sin(goal_angle_rad) * (width / 2 + 0.02)]
    rot_xy1_open[:2] = -rot_xy1_open[:2]
    rot_xy1_open[1] += 0.186
    rot_xy2_open = goal_position + [cos(goal_angle_rad) * (width / 2 + 0.02),
                                    sin(goal_angle_rad) * (width / 2 + 0.02)]
    rob_pos1[:2] = rot_xy1_open
    rob_pos2[:2] = rot_xy2_open
    robot_control_l(ur_pose=rob_pos1, ur2_pose=rob_pos2)

    rob_pos1[2] = object_z['z_lift']
    rob_pos2[2] = object_z['z_lift']
    robot_control_l(ur_pose=rob_pos1, ur2_pose=rob_pos2)

    ur.setGripper(0)
    ur2.setGripper(0)
    robot_control_j(rob_total_pose=home)


def desk_cleaner_task(cleaner_detection, cleaner_tail_detection):
    mean_xy = [cleaner_detection[2][1], cleaner_detection[2][0]]
    if cleaner_detection[0]=='GREEN_DESK_CLEANER':
        obj='cup'
        offset = np.array(offsets[obj])
        offset += np.array(offsets[obj + '_left']) * (pixel_point[obj][0] - mean_xy[1]) / (
                pixel_point[obj][0] - pixel_point[obj][2]) if mean_xy[1] < pixel_point[obj][0] else np.array(
            offsets[obj + '_right']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][3])
        offset += np.array(offsets[obj + '_up']) * (pixel_point[obj][1] - mean_xy[0]) / (
                pixel_point[obj][1] - pixel_point[obj][4]) if mean_xy[0] < pixel_point[obj][1] else np.array(
            offsets[obj + '_down']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][5])
    else:
        obj='multitap'
        offset = np.array(offsets[obj])
        offset += np.array(offsets[obj + '_left']) * (pixel_point[obj][0] - mean_xy[1]) / (
                pixel_point[obj][0] - pixel_point[obj][2]) if mean_xy[1] < pixel_point[obj][0] else np.array(
            offsets[obj + '_right']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][3])
        offset += np.array(offsets[obj + '_up']) * (pixel_point[obj][1] - mean_xy[0]) / (
                pixel_point[obj][1] - pixel_point[obj][4]) if mean_xy[0] < pixel_point[obj][1] else np.array(
            offsets[obj + '_down']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][5])
    target_pose = t1.rs.pxl2xyz(mean_xy,depth_w[obj])
    offset=list(offset)
    robot_control_j(ur_pose=ur_back_pose,ur2_pose=ur2_starting_pose)
    target_center_points = list(cleaner_detection[2][:2])
    target_center_points.append(list(cleaner_tail_detection[2][:2]))
    _ = object_grasping(rob=ur2, target_pose=target_pose, target=cleaner_detection[0],
                        target_center_points=target_center_points,offset=offset)
    if cleaner_detection[0]=='GREEN_DESK_CLEANER':
        robot_control_j(ur2_pose=ur2_green_desk_cleaner_center_joint)
    else:
        robot_control_j(ur2_pose=ur2_blue_desk_cleaner_center_joint)
    pos_to_clean = ur2.ur_rob.rtmon.get_all_data()['tcp']
    #cleaning_pos
    for i in range(15):
        x_ = np.random.uniform(-0.03, 0.03)
        y_ = np.random.uniform(-0.03, 0.03)
        temp_loc = pos_to_clean + [x_, y_, 0, 0, 0, 0]
        robot_control_l(ur2_pose=temp_loc,ur2_vel=0.1,ur2_acc=0.1)
    time.sleep(1.5)
    cur_pos = ur2.ur_rob.rtmon.get_all_data()['tcp']
    cur_pos[2] += 0.1
    robot_control_l(ur2_pose=cur_pos)
    robot_control_j(ur2_pose=ur2_starting_pose)
    robot_control_j(ur2_pose=ur2_placing_pose)
    ur2.setGripper(0)
    robot_control_j(ur2_pose=ur2_starting_pose)
    robot_control_j(rob_total_pose=home)


#######################################
# def hub_cable_task(target_detection):
#     hub_detection, hub_tail_detection, cable_detection, cable_tail_detection=target_detection
#
#     #cable잡기
#     mean_xy = [cable_detection[2][1], cable_detection[2][0]]
#     obj='cable'
#     offset = np.array(offsets[obj])
#     # offset += np.array(offsets[obj + '_left']) * (pixel_point[obj][0] - mean_xy[1]) / (
#     #         pixel_point[obj][0] - pixel_point[obj][2]) if mean_xy[1] < pixel_point[obj][0] else np.array(
#     #     offsets[obj + '_right']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][3])
#     # offset += np.array(offsets[obj + '_up']) * (pixel_point[obj][1] - mean_xy[0]) / (
#     #         pixel_point[obj][1] - pixel_point[obj][4]) if mean_xy[0] < pixel_point[obj][1] else np.array(
#     #     offsets[obj + '_down']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][5])
#     offset=-offset+offsets['ur_offset']
#     offset=list(offset)
#     target_pose = t1.rs.pxl2xyz(mean_xy,depth_w[obj])
#     robot_control_j(ur_pose=ur_starting_pose, ur2_pose=ur2_back_pose)
#     target_pose[:2]=-target_pose[:2]
#     target_center_points = list(cable_detection[2][:2])
#     target_center_points.append(list(cable_tail_detection[2][:2]))
#     _ = object_grasping(rob=ur, target_pose=target_pose, target=cable_detection[0],offset = offset,
#                     target_center_points=target_center_points),#offset_after_rotation=[0.005,-pi/2])
#     # cable 고쳐잡기
#     robot_control_j(ur_pose=pose_pre_transfer1_cable)
#     robot_control_j(ur_pose=pose_transfer1_cable,ur2_pose=pose_transfer1)
#     ur2.setGripper(255)
#     time.sleep(0.5)
#     ur.setGripper(0)
#     robot_control_j(ur_pose=pose_pre_transfer2_cable)
#     robot_control_j(ur_pose=pose_transfer2_cable, ur2_pose=pose_transfer2)
#     ur.setGripper(255)
#     time.sleep(0.5)
#     ur2.setGripper(0)
#     time.sleep(0.5)
#
#     # hub잡기
#     mean_xy = [hub_detection[2][1], hub_detection[2][0]]
#     offset = np.array(offsets[obj])
#     offset += np.array(offsets[obj + '_left']) * (pixel_point[obj][0] - mean_xy[1]) / (
#             pixel_point[obj][0] - pixel_point[obj][2]) if mean_xy[1] < pixel_point[obj][0] else np.array(
#         offsets[obj + '_right']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][3])
#     offset += np.array(offsets[obj + '_up']) * (pixel_point[obj][1] - mean_xy[0]) / (
#             pixel_point[obj][1] - pixel_point[obj][4]) if mean_xy[0] < pixel_point[obj][1] else np.array(
#         offsets[obj + '_down']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][5])
#     offset=list(offset)
#     target_pose = t1.rs.pxl2xyz(mean_xy,depth_w[obj])
#     robot_control_j(ur_pose=ur_back_pose, ur2_pose=ur2_starting_pose)
#     target_center_points = list(hub_detection[2][:2])
#     target_center_points.append(list(hub_tail_detection[2][:2]))
#     if hub_detection[0] == 'USB_HUB':
#         _ = object_grasping(rob=ur2, target_pose=target_pose, target=hub_detection[0],offset=offset,
#                         target_center_points=target_center_points,adding_rad=pi/2,),#offset_after_rotation=[0.025,pi/2])
#     else:
#         _ = object_grasping(rob=ur2, target_pose=target_pose, target=hub_detection[0],offset=offset,
#                         target_center_points=target_center_points)
#     if cable_detection[0] == 'USB_C':
#         robot_control_j(ur2_pose=pose_USB)
#         robot_control_j(ur_pose=pose_insert_usb)
#         z_threshold=0.28
#     else:
#         robot_control_j(ur2_pose=pose_hdmi)
#         robot_control_j(ur_pose=pose_insert_hdmi)
#         z_threshold=0.296
#
#     # inserting
#     tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
#     control_params = get_inserting_control_params()
#     for i in range(20):
#         x_ = np.random.uniform(-0.005, 0.005)
#         y_ = np.random.uniform(-0.003, 0.003)
#         target_pose = tmp_pose + [x_, y_, 0, 0, 0, 0]
#         fm_rob_control(ur, 1, target_pose, control_params)
#         time.sleep(0.3)
#         if ur.ur_rob.rtmon.get_all_data()['tcp'][2] < z_threshold:
#             break
#
#     # 조립완료물품 옮기기
#     ur.setGripper(0)
#     robot_control_j(ur2_pose=pose_complete_hdmi) ## 물품 바깥으로 처리해야할듯
#     ur2.setGripper(0)
#     robot_control_j(ur_pose=pose_start,ur2_pose=pose_base)

def hub_cable_task(target_cable_name,target_hub_name):
    #target_cable_name,target_hub_name  = target_list
    # 1. 허브 잡기(모델 사용)
    c=grasping_hub(target_hub_name, model_hp)
    if c==False:
        return
    # 2.케이블 잡기(모델사용)
    c=grasping_cable(target_cable_name, model_hp)
    if c==False:
        return

    # 3.고쳐잡기
    # 케이블 전달 전 자세
    robot_control_j(ur_pose=pose_pre_transfer1_cable,ur2_pose=pose_transfer1)
    # 케이블 전달 자세
    robot_control_j(ur_pose=pose_transfer1_cable)
    ur2.setGripper(255)
    time.sleep(0.5)
    ur.setGripper(0)
    # 케이블 다시 가져가기 전자세
    robot_control_j(ur_pose=pose_pre_transfer2_cable)
    robot_control_j(ur2_pose=pose_transfer2)
    # 케이블 다시 잡기
    robot_control_j(ur_pose=pose_transfer2_cable)
    ur.setGripper(255)
    time.sleep(0.5)
    ur2.setGripper(0)
    time.sleep(0.5)
    # 허브 임시 위치에서 가져오기
    robot_control_j(ur_pose=pose_pre_transfer2_cable,ur2_pose=pose_hub_ur2)

    tmp_pose = ur2.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[2] -= 0.005

    # 허브 마다 잡는 위치 다르기 때문에 구분.
    if target_cable_name == 'USB_C':
        # 허브 잡는 위치 조정
        tmp_pose[0] += 0.035
        robot_control_l(ur2_pose=tmp_pose)
        time.sleep(0.5)
        ur2.setGripper(255)
        time.sleep(0.5)

        # USB_C 조립, 삽입 자세 이동
        robot_control_j(ur_pose=pose_insert_usb, ur2_pose=pose_USB)
        inserting_task(0.28)
    else:
        robot_control_l(ur2_pose=tmp_pose)
        time.sleep(0.5)
        ur2.setGripper(255)
        time.sleep(0.5)

        # HDMI 조립, 삽입 자세 이동
        robot_control_j(ur_pose=pose_insert_hdmi, ur2_pose=pose_hdmi)
        tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
        tmp_pose[0]-=0.00596
        robot_control_l(ur_pose=tmp_pose)
        inserting_task(0.293)

def grasping_hub(target_name, model_yolo):
    robot_control_j(ur_pose=pose_hub_start)
    target_point,detections = tracking_target(ur, target_name, target_point=center_point['DEFAULT'], model_yolo=model_yolo)
    if detections == None:
        return False
    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[:2] += [0.086543526,0.050097434]#0.08855255,0.05369648
    robot_control_l(ur_pose=tmp_pose)# 그리퍼가 물체 중앙에 오도록 이동

    rad = calculate_rotating_radian(detections, target_name, target_point)
    if target_name=='USB_C_HUB':
        rad-=pi/2
    rotation_task(ur, rad)
    pose_current = ur.ur_rob.rtmon.get_all_data()['tcp']
    pose_current[2]=object_z[target_name]
    robot_control_l(ur_pose=pose_current)
    if target_name=='USB_C_HUB':
        move_rotation(ur, rad, 0.015,-0.11504811)
    ur.setGripper(255)

    # HUB 옴기기
    robot_control_j(ur_pose=pose_hub_start)
    robot_control_j(ur_pose=pose_hub_ur)
    time.sleep(0.5)
    ur.setGripper(0)
    time.sleep(0.5)

    # 초기 위치로 되돌림
    robot_control_j(ur_pose=pose_hub_start)
    return True

def grasping_cable(target_name, model_yolo):
    robot_control_j(ur_pose=pose_hub_start)
    target_point,detections = tracking_target(ur, target_name, target_point=center_point['DEFAULT'], model_yolo=model_yolo)
    if target_point == None:
        return False
    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[:2] += [0.086543526,0.050837577]
    robot_control_l(ur_pose=tmp_pose)

    rad = calculate_rotating_radian(detections, target_name, target_point)
    rotation_task(ur, rad)
    pose_current = ur.ur_rob.rtmon.get_all_data()['tcp']
    pose_current[2]=object_z[target_name]
    robot_control_l(ur_pose=pose_current)
    if target_name=='USB_C':
        move_rotation(ur, rad, 0.0105, -0.108579240)  #0.01
    else:
        move_rotation(ur, rad, 0.01, -0.108579240)
    ur.setGripper(255)
    return True

def inserting_task(z_threshold):
    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    control_params = get_inserting_control_params()
    for i in range(20):
        x_ = np.random.uniform(-0.005, 0.005)
        y_ = np.random.uniform(-0.003, 0.003)
        target_pose = tmp_pose + [x_, y_, 0, 0, 0, 0]
        fm_rob_control(ur, 1, target_pose, control_params)
        time.sleep(0.3)
        if ur.ur_rob.rtmon.get_all_data()['tcp'][2] < z_threshold:
            break
    ur.setGripper(0)

    # 조립완료물품 옮기기
    robot_control_j(ur2_pose=pose_complete_hdmi)
    ur2.setGripper(0)
    robot_control_j(ur_pose=pose_start,ur2_pose=pose_base)



def pencil_task(target_detection):
    pencil_detection, pencil_tail_detection, sharpener_detection, sharpener_tail_detection,pencilcase_detection=target_detection
    # 1. ur 기준 연필통 및 뚜껑 고정위치에 놓기
    ur_loc,ur_loc_j=placing_pencilcase(pencilcase_detection)
    #2. ur2 기준 연필깎이 고정위치로 옮기기
    placing_sharpener(sharpener_detection,sharpener_tail_detection)
    # 3. ur 기준(ur2도 사용) 연필 들어서 연필깎이로 사용
    inserting_pencil_in_sharpener(pencil_detection, pencil_tail_detection)
    # 4. 연필을 연필통에 넣기
    inserting_pencil_in_pencilcase(ur_loc)
    # 5. 뚜껑 닫기
    closing_pencilcase(ur_loc_j,pencilcase_detection)

def placing_pencilcase(pencilcase_detection):
    #ur로 몸통 잡고 ur2로 뚜껑열어서 옆에 놓기
    mean_xy = [pencilcase_detection[2][1], pencilcase_detection[2][0]]
    obj='pencilcase'
    target_pose = t1.rs.pxl2xyz(mean_xy,depth_w[obj])

    offset=np.array(offsets[obj])
    offset += np.array(offsets[obj + '_left']) * (pixel_point[obj][0] - mean_xy[1]) / (
                pixel_point[obj][0] - pixel_point[obj][2]) if mean_xy[1] < pixel_point[obj][0] else np.array(
        offsets[obj + '_right']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][3])
    offset += np.array(offsets[obj + '_up']) * (pixel_point[obj][1] - mean_xy[0]) / (
                pixel_point[obj][1] - pixel_point[obj][4]) if mean_xy[0] < pixel_point[obj][1] else np.array(
        offsets[obj + '_down']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][5])

    ur_offset=-offset+np.array(offsets['pencilcase_ur_offset'])
    ur_offset=list(ur_offset)
    ur_target_pose=[-target_pose[0],-target_pose[1],target_pose[2:]]
    robot_control_j(ur_pose=ur_starting_pose, ur2_pose=ur2_back_pose)
    _ = object_grasping(rob=ur, target_pose=ur_target_pose, target=pencilcase_detection[0],not_gripping=True,offset=ur_offset)
    #180도 돌리기
    ur_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    ur_pose[2]+=0.1
    robot_control_l(ur_pose=ur_pose)
    temp_deg=np.rad2deg(ur.ur_rob.getj())
    temp_deg[5]-=180
    robot_control_j(ur_pose=np.deg2rad(temp_deg))
    ur_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    ur_pose[3:] =[-1.93498903, 0.00960883, -0.02889001]
    robot_control_l(ur_pose=ur_pose,ur_vel=0.5,ur_acc=0.5)
    ur_pose_j = ur.ur_rob.getj()
    ur_pose[2]=-0.02152355
    robot_control_l(ur_pose=ur_pose,ur_vel=0.5,ur_acc=0.5)
    ur.setGripper(255)

    robot_control_j(ur2_pose=ur2_starting_pose)
    offset=list(offset)
    _ = object_grasping(rob=ur2, target_pose=target_pose, target=pencilcase_detection[0],case_open=True, offset=offset,not_lift=True)
    tmp_pose = ur2.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[2] += 0.215
    robot_control_l(ur2_pose=tmp_pose)
    robot_control_j(ur2_pose=ur2_pencilcase_lid_position1)
    robot_control_j(ur2_pose=ur2_pencilcase_lid_position2,ur2_vel=0.2,ur2_acc=0.2)

    ur.setGripper(0)
    ur2.setGripper(60)
    time.sleep(0.5)

    tmp2_pose = ur2.ur_rob.rtmon.get_all_data()['tcp']
    tmp2_pose[1]-=0.045
    robot_control_l(ur2_pose=tmp2_pose)
    ur2.setGripper(120)
    tmp2_pose[1] += 0.02
    robot_control_l(ur2_pose=tmp2_pose,ur2_acc=0.1,ur2_vel=0.1)
    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[2]+=0.15
    tmp2_pose = ur2.ur_rob.rtmon.get_all_data()['tcp']
    tmp2_pose[2]+=0.15
    robot_control_l(ur_pose=tmp_pose,ur2_pose=tmp2_pose)
    ur2.setGripper(0)
    robot_control_j(ur_pose=pose_start,ur2_pose=pose_base)
    return ur_pose,ur_pose_j

def placing_sharpener(sharpener_detection,sharpener_tail_detection):
    mean_xy = [sharpener_detection[2][1], sharpener_detection[2][0]]
    obj='cup'
    offset=np.array(offsets[obj])
    offset += np.array(offsets[obj + '_left']) * (pixel_point[obj][0] - mean_xy[1]) / (
                pixel_point[obj][0] - pixel_point[obj][2]) if mean_xy[1] < pixel_point[obj][0] else np.array(
        offsets[obj + '_right']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][3])
    offset += np.array(offsets[obj + '_up']) * (pixel_point[obj][1] - mean_xy[0]) / (
                pixel_point[obj][1] - pixel_point[obj][4]) if mean_xy[0] < pixel_point[obj][1] else np.array(
        offsets[obj + '_down']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][5])
    offset[1]+=0.02
    offset=list(offset)
    target_pose = t1.rs.pxl2xyz(mean_xy,depth_w[obj])
    robot_control_j(ur_pose=ur_back_pose, ur2_pose=ur2_starting_pose)
    target_center_points = list(sharpener_detection[2][:2])
    target_center_points.append(list(sharpener_tail_detection[2][:2]))
    loc,preloc = object_grasping(rob=ur2, target_pose=target_pose, target=sharpener_detection[0],
                        target_center_points=target_center_points,offset = offset,adding_rad=pi)
    loc[2]+=0.1
    robot_control_l(ur2_pose=loc)
    preloc[2]+=0.1
    robot_control_l(ur2_pose=preloc)
    robot_control_j(ur2_pose=ur2_sharpener_setting_position)
    tmp_pose = ur2.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[2] = object_z[sharpener_detection[0]]
    robot_control_l(ur2_pose=tmp_pose)
    ur2.setGripper(0)
    tmp_pose[3:]=[0.0372148, 2.65224217, 1.66969745]
    robot_control_l(ur2_pose=tmp_pose)
    tmp_pose[1]+=0.065
    robot_control_l(ur2_pose=tmp_pose,ur2_vel=0.2,ur2_acc=0.2)
    ur2.setGripper(255)
    tmp_pose[2] += 0.24
    robot_control_l(ur2_pose=tmp_pose)
    robot_control_j(ur2_pose=ur2_sharpener_waiting_position)

def inserting_pencil_in_sharpener(pencil_detection, pencil_tail_detection):
    mean_xy = [pencil_detection[2][1], pencil_detection[2][0]]
    obj='glue'
    offset=np.array(offsets[obj])
    offset += np.array(offsets[obj + '_left']) * (pixel_point[obj][0] - mean_xy[1]) / (
                pixel_point[obj][0] - pixel_point[obj][2]) if mean_xy[1] < pixel_point[obj][0] else np.array(
        offsets[obj + '_right']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][3])
    offset += np.array(offsets[obj + '_up']) * (pixel_point[obj][1] - mean_xy[0]) / (
                pixel_point[obj][1] - pixel_point[obj][4]) if mean_xy[0] < pixel_point[obj][1] else np.array(
        offsets[obj + '_down']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][5])
    offset=-offset+np.array(offsets['ur_offset'])
    offset=list(offset)
    target_pose = t1.rs.pxl2xyz(mean_xy,depth_w[obj])
    robot_control_j(ur_pose=ur_starting_pose)
    target_pose[:2] = -target_pose[:2]
    target_center_points = list(pencil_detection[2][:2])
    target_center_points.append(list(pencil_tail_detection[2][:2]))
    _ = object_grasping(rob=ur, target_pose=target_pose, target=pencil_detection[0],offset=offset,
                        target_center_points=target_center_points)

    if pencil_detection[0]=="ORANGE_PENCIL":
        robot_control_j(ur_pose=ur_orange_sharpener_inserting_position,ur2_pose=ur2_sharpener_position)
    elif pencil_detection[0]=='BLUE_PENCIL':
        robot_control_j(ur_pose=ur_blue_sharpener_inserting_position,ur2_pose=ur2_sharpener_position)


    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    ##inserting##
    while True:
        x_ = np.random.uniform(-0.0001, 0.0001)
        y_ = np.random.uniform(-0.0001, 0.0001)
        tmp_pose[2]-=0.0005
        target_pose = tmp_pose + [x_, y_, 0, 0, 0, 0]
        ur.ur_rob.movel(target_pose,0.1,0.1, wait=False, relative=False)
        #목표지점에 도달하면 멈추는 코드 추가
        if tmp_pose[2]<object_z[pencil_detection[0]+'_INSERT']: #높이 변경해야하나
            break
    time.sleep(3)
    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[2] += 0.1
    robot_control_l(ur_pose=tmp_pose)
    robot_control_j(ur2_pose=ur2_sharpener_waiting_position)
    robot_control_j(ur_pose=ur_back_pose)
    robot_control_j(ur2_pose=ur2_placing_pose)
    ur2.setGripper(0)
    robot_control_j(rob_total_pose=home)
    robot_control_j(ur_pose=ur_pencil_insering_pose,ur2_pose=ur2_back_pose)


def inserting_pencil_in_pencilcase(loc):
    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[:2]=loc[:2]
    robot_control_l(ur_pose=tmp_pose)
    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[2] = 0.20 #0.12
    robot_control_l(ur_pose=tmp_pose)
    ur.setGripper(0)

def closing_pencilcase(ur_loc_j,pencilcase_detection):
    robot_control_j(ur_pose=ur_loc_j)
    ur_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    ur_pose[2]=-0.06152355
    robot_control_l(ur_pose=ur_pose,ur_vel=0.3,ur_acc=0.3)
    ur_pose[1]+=0.038 #0.033
    ur_pose[2]-=0.02
    robot_control_l(ur_pose=ur_pose, ur_vel=0.1, ur_acc=0.1)
    ur_pose[1] -= 0.02
    ur_pose[2] += 0.06
    robot_control_l(ur_pose=ur_pose, ur_vel=0.1, ur_acc=0.1)
    robot_control_j(ur2_pose=ur2_starting_pose)
    robot_control_j(ur2_pose=ur2_pencilcase_lid_position1)
    robot_control_j(ur2_pose=ur2_pencilcase_lid_position2,ur2_vel=0.5,ur2_acc=0.5)
    ur.setGripper(255)
    ur2.setGripper(255)
    tmp_pose = ur2.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[2]+=0.4
    robot_control_l(ur2_pose=tmp_pose)
    robot_control_j(ur_pose=ur_pencilcase_inserting_pose, ur2_pose=ur2_pencilcase_inserting_pose)
    tmp_pose = ur2.ur_rob.rtmon.get_all_data()['tcp']
    while True:
        #x_ = np.random.uniform(-0.0005, 0.0005)
        y_ = np.random.uniform(-0.002, 0.002)
        tmp_pose[2]-=0.001
        target_pose = tmp_pose + [0, y_, 0, 0, 0, 0]
        robot_control_l(ur2_pose=target_pose,ur2_vel=0.1,ur2_acc=0.1)
        #목표지점 도달하면 멈추는걸로 추가
        if tmp_pose[2]< 0.21531261:# 높이정보 수정
            break
    time.sleep(1)
    ur.setGripper(0)
    tmp_pose = ur2.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[2] += 0.1
    robot_control_l(ur2_pose=tmp_pose)
    robot_control_j(ur2_pose=ur2_starting_pose)
    robot_control_j(ur2_pose=ur2_pencilcase_placing_pose)
    ur2.setGripper(0)
    robot_control_j(rob_total_pose=home)
    robot_control_j(ur_pose=pose_start, ur2_pose=pose_base)


def plug_multitap_task(target_detection):
    plug_detection, plug_tail_detection, multitap_detection, multitap_tail_detection= target_detection
    # 1.plug회전해서잡기
    grasping_plug(plug_detection, plug_tail_detection)
    # 2. 멀티탭 지정 위치에 놓기
    placing_multitap(multitap_detection, multitap_tail_detection)
    # 3.삽입
    inserting_plug(plug_detection)


def grasping_plug(plug_detection, plug_tail_detection):
    mean_xy = [plug_detection[2][1], plug_detection[2][0]]
    robot_control_j(ur_pose=ur_starting_pose, ur2_pose=ur2_back_pose)
    obj = 'multitap'
    offset = np.array(offsets[obj])
    offset += np.array(offsets[obj + '_left']) * (pixel_point[obj][0] - mean_xy[1]) / (
                pixel_point[obj][0] - pixel_point[obj][2]) if mean_xy[1] < pixel_point[obj][0] else np.array(
        offsets[obj + '_right']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][3])
    offset += np.array(offsets[obj + '_up']) * (pixel_point[obj][1] - mean_xy[0]) / (
                pixel_point[obj][1] - pixel_point[obj][4]) if mean_xy[0] < pixel_point[obj][1] else np.array(
        offsets[obj + '_down']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][5])
    offset=-offset+offsets['ur_offset']
    offset=list(offset)
    target_pose = t1.rs.pxl2xyz(mean_xy, depth_w[obj])
    target_pose[:2] = -target_pose[:2]
    target_center_points = list(plug_detection[2][:2])
    target_center_points.append(list(plug_tail_detection[2][:2]))

    # if plug_detection[0]=='WHITE_PLUG':
    _ = object_grasping(rob=ur, target_pose=target_pose, target=plug_detection[0],offset = offset,
                    target_center_points=target_center_points,pushing=0.055),#offset_after_rotation=[0.01,pi/2])
    # elif plug_detection[0]=='BLACK_PLUG':
    #     _ = object_grasping(rob=ur, target_pose=target_pose, target=plug_detection[0],offset = offset,
    #                         target_center_points=target_center_points,offset_after_rotation=[0.01,pi/2])#,adding_rad=pi)
    #돌려잡기
    robot_control_j(ur_pose=pose_pre_transfer1_cable)
    robot_control_j(ur_pose=pose_transfer1_cable,ur2_pose=pose_transfer1)
    ur2.setGripper(255)
    time.sleep(0.5)
    ur.setGripper(0)
    robot_control_j(ur_pose=pose_pre_transfer2_cable)
    robot_control_j(ur2_pose=ur2_plug_pose)
    robot_control_j(ur_pose=ur_plug_pose)
    ur.setGripper(255)
    time.sleep(0.5)
    ur2.setGripper(0)
    time.sleep(0.5)

def placing_multitap(multitap_detection, multitap_tail_detection):
    robot_control_j(ur_pose=ur_back_pose, ur2_pose=ur2_starting_pose)
    mean_xy = [multitap_detection[2][1], multitap_detection[2][0]]
    obj = 'multitap'
    offset = np.array(offsets[obj])
    offset += np.array(offsets[obj + '_left']) * (pixel_point[obj][0] - mean_xy[1]) / (
                pixel_point[obj][0] - pixel_point[obj][2]) if mean_xy[1] < pixel_point[obj][0] else np.array(
        offsets[obj + '_right']) * (pixel_point[obj][0] - mean_xy[1]) / (pixel_point[obj][0] - pixel_point[obj][3])
    offset += np.array(offsets[obj + '_up']) * (pixel_point[obj][1] - mean_xy[0]) / (
                pixel_point[obj][1] - pixel_point[obj][4]) if mean_xy[0] < pixel_point[obj][1] else np.array(
        offsets[obj + '_down']) * (pixel_point[obj][1] - mean_xy[0]) / (pixel_point[obj][1] - pixel_point[obj][5])
    offset=list(offset)
    target_pose = t1.rs.pxl2xyz(mean_xy,depth_w[obj])
    target_center_points = list(multitap_detection[2][:2])
    target_center_points.append(list(multitap_tail_detection[2][:2]))
    _ = object_grasping(rob=ur2, target_pose=target_pose, target=multitap_detection[0],offset = offset,
                        target_center_points=target_center_points)
    if multitap_detection[0] == 'GREEN_MULTITAP':
        robot_control_j(ur2_pose=ur2_greenmultitap_position1)
        robot_control_j(ur2_pose=ur2_greenmultitap_position2)
        ur2.setGripper(100)
        time.sleep(0.5)
        ur2.setGripper(0)
        tmp_pose = ur2.ur_rob.rtmon.get_all_data()['tcp']
        tmp_pose[1]-=0.12
        tmp_pose[2]+=0.026
        robot_control_l(ur2_pose=tmp_pose)
        ur2.setGripper(180)
        time.sleep(0.5)
        tmp_pose[1] += 0.053
        robot_control_l(ur2_pose=tmp_pose,ur2_vel=0.1,ur2_acc=0.1)
        time.sleep(0.5)
        tmp_pose[1] -= 0.01
        tmp_pose[2]+=0.07
        robot_control_l(ur2_pose=tmp_pose)
        ur2.setGripper(0)
        robot_control_j(ur2_pose=ur2_multitap_pose)
        tmp_pose = ur2.ur_rob.rtmon.get_all_data()['tcp']
        tmp_pose[1]+=0.006
        tmp_pose[2] -= 0.048
        robot_control_l(ur2_pose=tmp_pose,ur2_acc=0.5,ur2_vel=0.5)
        time.sleep(0.5)
        ur2.setGripper(255)
    elif multitap_detection[0] == 'BLACK_MULTITAP':
        robot_control_j(ur2_pose=ur2_blackmultitap_position1)
        robot_control_j(ur2_pose=ur2_blackmultitap_position2)
        ur2.setGripper(30)
        time.sleep(0.5)
        ur2.setGripper(0)
        tmp_pose = ur2.ur_rob.rtmon.get_all_data()['tcp']
        tmp_pose[1] -= 0.1
        tmp_pose[2] += 0.0135
        robot_control_l(ur2_pose=tmp_pose)
        ur2.setGripper(150)
        tmp_pose[1] += 0.05
        robot_control_l(ur2_pose=tmp_pose, ur2_vel=0.1, ur2_acc=0.1)
        time.sleep(0.5)
        tmp_pose[1] -= 0.01
        tmp_pose[2]+=0.07
        robot_control_l(ur2_pose=tmp_pose)
        ur2.setGripper(0)
        robot_control_j(ur2_pose=ur2_multitap_pose)
        tmp_pose = ur2.ur_rob.rtmon.get_all_data()['tcp']
        tmp_pose[1]+=0.028
        tmp_pose[2] -= 0.052
        robot_control_l(ur2_pose=tmp_pose,ur2_acc=0.5,ur2_vel=0.5)
        time.sleep(0.5)
        ur2.setGripper(255)

def inserting_plug(plug_detection):
    robot_control_j(ur_pose=ur_starting_pose)
    target_pose=ur2.ur_rob.rtmon.get_all_data()['tcp']
    offset = 0.04

    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[0]=-target_pose[0]+offset
    robot_control_l(ur_pose=tmp_pose)
    if plug_detection[0] == 'WHITE_PLUG':
        robot_control_j(ur_pose=ur_whiteplug_inserting_position)
    elif plug_detection[0] == 'BLACK_PLUG':
        robot_control_j(ur_pose=ur_blackplug_inserting_position)
    #inserting
    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    while True:
        x_ = np.random.uniform(-0.0005, 0.0005)
        y_ = np.random.uniform(-0.0005, 0.0005)
        tmp_pose[2] -= 0.0005
        target_pose = tmp_pose + [x_, y_, 0, 0, 0, 0]
        robot_control_l(ur_pose=target_pose,ur_vel=0.1,ur_acc=0.1)
        # 목표지점 도달하면 멈추는걸로 추가
        if tmp_pose[2] < object_z[plug_detection[0]+'_INSERT']:# 도달 지점 변경
            break
    time.sleep(1)
    ur.setGripper(0)
    tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[2] += 0.1
    robot_control_l(ur_pose=tmp_pose)
    robot_control_j(ur_pose=home)
    tmp_pose = ur2.ur_rob.rtmon.get_all_data()['tcp']
    tmp_pose[2] += 0.1
    robot_control_l(ur2_pose=tmp_pose)
    robot_control_j(ur2_pose=ur2_multitap_back_position1)
    robot_control_j(ur2_pose=ur2_multitap_back_position2)
    ur2.setGripper(0)
    robot_control_j(ur2_pose=home)
    #ur2로 바깥으로 빼서 정리


# def target_task(target_list):
#     if target_list[0] == 'ORANGE_PENCIL' or target_list[0] == 'BLUE_PENCIL':
#         pencil_task(target_list)  # case뚜껑열때 지탱해주는 손이 먼저잡는거. 뚜꼉위치 변경, 위치변경이 필요한지 wide object랑 비교할것. 작업이후 case 빼기
#     elif target_list[0] == 'WHITE_PLUG' or target_list[0] == 'BLACK_PLUG':
#         plug_multitap_task(target_list)  # 멀티탭 위치 재조정. 마지막 제거까지
#     elif target_list[0] == 'HDMI_HUB' or target_list[0] == 'USB_HUB':
#         hub_cable_task(target_list)




# def pencil_task(target_list):
#     target_pencil_name, target_sharpener_name,target_pencilcase_name = target_list
#     # 1. 연필통 및 뚜껑 고정위치에 놓기
#     placing_pencilcase(target_pencilcase_name,model_yolo1)
#     # 2. 연필깎이 고정위치로 옮기기
#     placing_sharpener(target_sharpener_name,model_yolo1)
#     # 3. 연필 들어서 연필깎이로 사용
#     inserting_pencil_in_sharpener(target_pencil_name,model_yolo1)
#     # 4. 연필을 연필통에 넣기
#     inserting_pencil_in_pencilcase()
#     # 5. 뚜껑 닫기
#     closing_pencilcase()

# def placing_pencilcase(target_pencilcase_name,model_yolo):
#     _ = tracking_target(ur, target_pencilcase_name, target_point=center_point['DEFAULT'], model_yolo=model_yolo,
#                                    obj_z=object_z[target_pencilcase_name],task='pencil')
#
#     tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
#     tmp_pose[0] += 0.055
#     robot_control_l(ur_pose=tmp_pose)
#     ur.setGripper(255)
#     tmp_pose[2] += 0.315
#     robot_control_l(ur_pose=tmp_pose)
#     robot_control_j(ur_pose=ur_pencilcase_position)
#     tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
#     tmp_pose[2] -= 0.066
#     robot_control_l(ur_pose=tmp_pose)
#
#     ##ur2로 연필통 몸통잡기
#     robot_control_j(ur2_pose=ur2_pencilcase_start)
#     robot_control_j(ur2_pose=ur2_pencilcase_position)
#     ur.setGripper(0)
#     time.sleep(0.5)
#     ur2.setGripper(255)
#     time.sleep(0.5)
#     ur.setGripper(255)
#
#     ##ur로 뚜꼉열기
#     tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
#     tmp_pose[2] += 0.215
#     robot_control_l(ur_pose=tmp_pose)
#     robot_control_j(ur_pose=ur_pencilcase_lid_position)
#     tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
#     tmp_pose[2] -= 0.058
#     robot_control_l(ur_pose=tmp_pose)
#
#     ur.setGripper(0)
#     ur2.setGripper(0)
#     tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
#     tmp_pose[2] += 0.05
#     robot_control_l(ur_pose=tmp_pose)
#     robot_control_j(ur_pose=pose_start,ur2_pose=ur2_pencilcase_start)
#     robot_control_j(ur2_pose=pose_base)
#
# def placing_sharpener(target_sharpener_name,model_yolo):
#     if target_sharpener_name=="ORANGE_SHARPENER":
#         _ = tracking_target(ur, target_sharpener_name, target_point=center_point['DEFAULT'],
#                             model_yolo=model_yolo, obj_z=object_z[target_sharpener_name],task='pencil')
#     elif target_sharpener_name=="BLUE_SHARPENER":
#         target_point,detections = tracking_target(ur, target_sharpener_name, target_point=center_point['CAMERA'],
#                             model_yolo=model_yolo, obj_z=object_z['DEFAULT'],task='pencil')
#         rad = calculate_rotating_radian(detections, target_sharpener_name, target_point)
#         rotation_task(ur, rad)
#         _ = tracking_target(ur, target_sharpener_name, target_point=center_point[target_sharpener_name],
#                             model_yolo=model_yolo, obj_z=object_z['DEFAULT'], radian=rad, task='pencil')
#
#     tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
#     if target_sharpener_name=="ORANGE_SHARPENER":
#         tmp_pose[0] += 0.055
#     elif target_sharpener_name=="BLUE_SHARPENER":
#         tmp_pose[2]=object_z[target_sharpener_name]
#     robot_control_l(ur_pose=tmp_pose)
#     ur.setGripper(255)
#     tmp_pose[2] += 0.44
#     robot_control_l(ur_pose=tmp_pose)
#     robot_control_j(ur_pose=ur_sharpener_position)
#     tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
#     tmp_pose[2]=object_z[target_sharpener_name]
#     robot_control_l(ur_pose=tmp_pose)
#     ur.setGripper(0)
#     tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
#     tmp_pose[2] += 0.063
#     robot_control_l(ur_pose=tmp_pose)
#     robot_control_j(ur_pose=pose_start)
#
# def inserting_pencil_in_sharpener(target_pencil_name,model_yolo):
#     target_point,detections = tracking_target(ur, target_pencil_name, target_point=center_point['CAMERA'],
#                                               model_yolo=model_yolo, obj_z=object_z['DEFAULT'], task='pencil')
#     ##그리퍼 회전
#     rad=calculate_rotating_radian(detections,target_pencil_name,target_point)
#     rotation_task(ur, rad)
#
#     _ = tracking_target(ur, target_pencil_name, target_point=center_point[target_pencil_name], model_yolo=model_yolo,
#                                    obj_z=object_z['DEFAULT'],radian=rad, task='pencil')
#
#     tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
#     tmp_pose[2]=object_z[target_pencil_name]
#     robot_control_l(ur_pose=tmp_pose)
#     ur.setGripper(255)
#     tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
#     tmp_pose[2] += 0.44  ##
#     robot_control_l(ur_pose=tmp_pose)
#
#     robot_control_j(ur2_pose=ur2_sharpener_start)
#     if target_pencil_name=="ORANGE_PENCIL":
#         robot_control_j(ur_pose=ur_orange_sharpener_inserting_position,ur2_pose=ur2_sharpener_position)
#     elif target_pencil_name=='BLUE_PENCIL':
#         robot_control_j(ur_pose=ur_blue_sharpener_inserting_position,ur2_pose=ur2_sharpener_position)
#     ur2.setGripper(255)
#
#     ##inserting##
#     tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
#     tmp_pose[2]=0.0575
#     robot_control_l(ur_pose=tmp_pose)
#
#     while True:
#         x_ = np.random.uniform(-0.001, 0.001)
#         y_ = np.random.uniform(-0.001, 0.001)
#         tmp_pose[2]-=0.005
#         target_pose = tmp_pose + [x_, y_, 0, 0, 0, 0]
#         ur.ur_rob.movel(target_pose,0.1,0.1, wait=False, relative=False)
#         #목표지점에 도달하면 멈추는 코드 추가
#         if tmp_pose[2]<object_z[target_pencil_name+'_INSERT']:
#             break
#     time.sleep(4)
#     tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
#     tmp_pose[2] = 0.3
#     robot_control_l(ur_pose=tmp_pose)
#
# def inserting_pencil_in_pencilcase():
#     robot_control_j(ur_pose=ur_pencilcase_inserting_position)
#     robot_control_j(ur2_pose=pose_complete_hdmi)
#     ur2.setGripper(0)
#
#     tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
#     tmp_pose[2] = 0.12  ##
#     robot_control_l(ur_pose=tmp_pose)
#     ur.setGripper(0)
#     tmp_pose[2] += 0.1  ##
#     robot_control_l(ur_pose=tmp_pose)
#
# def closing_pencilcase():
#     robot_control_j(ur_pose=ur_pencilcase_lid_position,ur2_pose=ur2_pencilcase_start)
#     tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
#     tmp_pose[2] -= 0.035
#     robot_control_l(ur_pose=tmp_pose)
#     ur.setGripper(255)
#     robot_control_j(ur2_pose=ur2_pencilcase_position)
#     ur2.setGripper(255)
#     tmp_pose[2] += 0.3
#     robot_control_l(ur_pose=tmp_pose)
#     robot_control_j(ur2_pose=ur_pencilcase_lid_back_position)
#
#     ##inserting##제대로 들어가도록 수정할것
#     tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
#     tmp_pose[2]=0.117
#     robot_control_l(ur_pose=tmp_pose)
#     while True:
#         x_ = np.random.uniform(-0.001, 0.001)
#         y_ = np.random.uniform(-0.001, 0.001)
#         tmp_pose[2]-=0.003
#         target_pose = tmp_pose + [x_, y_, 0, 0, 0, 0]
#         ur.ur_rob.movel(target_pose,0.1,0.1, wait=False, relative=False)
#         #목표지점 도달하면 멈추는걸로 추가
#         if tmp_pose[2]<0.0775:
#             break
#     time.sleep(1)
#     ur.setGripper(0)
#     tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
#     tmp_pose[2] += 0.1
#     robot_control_l(ur_pose=tmp_pose)
#     robot_control_j(ur2_pose=ur2_pencilcase_start)
#     robot_control_j(ur2_pose=pose_complete_hdmi)
#     ur2.setGripper(0)
#     robot_control_j(ur_pose=pose_start, ur2_pose=pose_base)


# def plug_multitap_task(target_list):
#     plug_name,multitap_name=target_list
#     # 1. 멀티탭 지정 위치에 놓기
#     placing_multitap(multitap_name,model_yolo1)
#     # 2.plug회전해서잡기
#     grasping_plug(plug_name,model_yolo1)
#     # 3.삽입
#     inserting_plug(plug_name)
#
# def placing_multitap(multitap_name,model_yolo):
#     target_point, detections = tracking_target(ur, multitap_name, target_point=center_point['CAMERA'],
#                         model_yolo=model_yolo, obj_z=object_z['MULTITAP'], task='plug')
#
#     rad=calculate_rotating_radian(detections,multitap_name,target_point)
#     rotation_task(ur, rad)
#     _ = tracking_target(ur, multitap_name, target_point=center_point[multitap_name],
#                         model_yolo=model_yolo, obj_z=object_z['MULTITAP'],radian=rad, task='plug')
#
#     tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
#     tmp_pose[2]=object_z[multitap_name]
#     robot_control_l(ur_pose=tmp_pose)
#     ur.setGripper(255)
#
#     tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
#     tmp_pose[2] += 0.3
#     robot_control_l(ur_pose=tmp_pose)
#     robot_control_j(ur_pose=ur_multitap_position)
#     ur.setGripper(0)
#     tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
#     tmp_pose[2] += 0.05
#     robot_control_l(ur_pose=tmp_pose)
#     robot_control_j(ur_pose=pose_start)
#
# def grasping_plug(plug_name,model_yolo):
#     target_point,detections = tracking_target(ur, plug_name, target_point=center_point['CAMERA'],
#                                    model_yolo=model_yolo,obj_z=object_z['MULTITAP'], task='plug')
#
#     rad=calculate_rotating_radian(detections,plug_name,target_point)
#     rotation_task(ur, rad)
#     _ = tracking_target(ur, plug_name, target_point=center_point[plug_name],
#                         model_yolo=model_yolo,obj_z=object_z['MULTITAP'],radian=rad, task='plug')
#
#     tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
#     tmp_pose[2] = object_z[plug_name]
#     robot_control_l(ur_pose=tmp_pose)
#     ur.setGripper(255)
#     tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
#     tmp_pose[2] += 0.1
#     robot_control_l(ur_pose=tmp_pose)
#
# def inserting_plug(plug_name):
#     robot_control_j(ur_pose=ur_plug_inserting_position)
#     #조금씩 움직여서 넣기
#     tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
#     while True:
#         x_ = np.random.uniform(-0.001, 0.001)
#         y_ = np.random.uniform(-0.001, 0.001)
#         tmp_pose[2] -= 0.001
#         target_pose = tmp_pose + [x_, y_, 0, 0, 0, 0]
#         ur.ur_rob.movel(target_pose, 0.1, 0.1, wait=False, relative=False)
#         # 목표지점 도달하면 멈추는걸로 추가
#         if tmp_pose[2] < object_z[plug_name+'_INSERT']:# 도달 지점 변경
#             break
#     time.sleep(1)
#     ur.setGripper(0)
#     tmp_pose = ur.ur_rob.rtmon.get_all_data()['tcp']
#     tmp_pose[2] += 0.1
#     robot_control_l(ur_pose=tmp_pose)
#     robot_control_j(ur_pose=pose_start)
#     robot_control_j(ur2_pose=ur2_multitap_position)
#     tmp_pose = ur2.ur_rob.rtmon.get_all_data()['tcp']
#     tmp_pose[2]=-0.131
#     robot_control_l(ur2_pose=tmp_pose)
#     ur2.setGripper(255)
#     robot_control_j(ur2_pose=pose_complete_hdmi)
#     ur2.setGripper(0)
#     robot_control_j(ur2_pose=pose_base)

