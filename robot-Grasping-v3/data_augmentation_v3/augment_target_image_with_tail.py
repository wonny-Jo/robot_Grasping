import os
import numpy as np
import multiprocessing as mp

from PIL import Image, ImageDraw
from tqdm import tqdm

def progress_bar(s_path, total):
    p_bar = tqdm(total=total)
    num_out, pre_out = 0, 0
    while num_out < int(p_bar.total):
        num_out = len(os.listdir(s_path))
        p_bar.update(num_out-pre_out)
        pre_out = num_out

def augment_images(b_img, objects, xyr, bn, tn, num_length, draw_on = False):
    bbox_point = {
        #새로 구한 꼬리 좌표
        # 좌상  우하  우상  좌하
        # hub & penholder
        # hub
        # "0_0.png": [0, 0, -32, 0, -32, 0, 0, 0],
        # "1_0.png": [25, 0, 0, 0, 0, 0, 25, 0],
        # "2_0.png": [0, 0, -99, 0, -99, 0, 0, 0],
        # "3_0.png": [0, 55, 0, 0, 0, 55, 0, 0],
        # "0_1.png": [0, 0, -105, 0, -105, 0, 0, 0],
        # "1_1.png": [80, 0, 0, 0, 0, 0, 80, 0],
        # "2_1.png": [0, 0, -300, 0, -300, 0, 0, 0],
        # "3_1.png": [0, 167, 0, 0, 0, 167, 0, 0],
        # penholder
        # "4.png": [90, 0, 0, 0, 0, 0, 90, 0],
        # "5.png": [90, 0, 0, 0, 0, 0, 90, 0],
        # "6.png": [90, 0, 0, 0, 0, 0, 90, 0],
        # "7.png": [90, 0, 0, 0, 0, 0, 90, 0],
        # "8.png": [0, 70, 0, 0, 0, 70, 0, 0],

        # multitap & pencil case
        # "0.png": [0, 0, 0, -40, 0, 0, 0, -40],
        # "1.png": [0, 0, 0, -54, 0, 0, 0, -54],
        # "2.png": [0, 140, 0, 0, 0, 140, 0, 0],
        # "3.png": [213, 0, 0, 0, 0, 0, 213, 0],
        # "4.png": [0, 175, 0, 0, 0, 175, 0, 0],
        # "5.png": [0, 155, 0, 0, 0, 155, 0, 0],
        # "6.png": [0, 0, 0, -59, 0, 0, 0, -59],
        # "7.png": [30, 32, -38, -29, -38, 32, 30, -29],
        # bin & drawer
        # "0.png": [0, 90, 0, 0, 0, 90, 0, 0],
        # "1.png": [0, 50, 0, 0, 0, 50, 0, 0],
        # "2.png": [30, 0, 0, 0, 0, 0, 30, 0],
        # "3.png": [50, 0, 0, 0, 0, 0, 50, 0],
        # "4.png": [0, 0, -40, 0, -40, 0, 0, 0],
        # "5.png": [25, 0, 0, 0, 0, 0, 25, 0],
        #picking
        # "0.png": [0, 45, 0, 0, 0, 45, 0, 0],
        # "1.png": [0, 45, 0, 0, 0, 45, 0, 0],
        # "2.png": [0, 40, 0, 0, 0, 40, 0, 0],
        # "3.png": [0, 50, 0, 0, 0, 50, 0, 0],
        # "4.png": [0, 0, 0, -45, 0, 0, 0, -45],
        # "5.png": [0, 0, 0, -70, 0, 0, 0, -70],
        # "6.png": [0, 0, 0, -90, 0, 0, 0, -90],
        # "7.png": [0, 80, 0, 0, 0, 80, 0, 0],
        #cleaner &  wide object & bottle lid
        # "0.png": [0, 0, -90, 0, -90, 0, 0, 0],
        # "1.png": [0, 0, -50, 0, -50, 0, 0, 0],
        # "2.png": [0, 0, -235, 0, -235, 0, 0, 0],
        # "3.png": [240, 0, 0, 0, 0, 0, 240, 0],
        # "4.png": [0, 0, 0, -300, 0, 0, 0, -300],
        # "5.png": [0, 0, 0, -300, 0, 0, 0, -300],
        # "6.png": [0, 110, 0, 0, 0, 110, 0, 0],
        # "7.png": [0, 115, 0, 0, 0, 115, 0, 0],
        # picking2
        "0.png": [0, 50, 0, 0, 0, 50, 0, 0],
        "1.png": [0, 60, 0, 0, 0, 60, 0, 0],
        "2.png": [0, 60, 0, 0, 0, 60, 0, 0],
        "3.png": [0, 70, 0, 0, 0, 70, 0, 0],
        "4.png": [0, 55, 0, 0, 0, 55, 0, 0],
        "5.png": [50, 0, 0, 0, 0, 0, 50, 0],
        "6.png": [50, 0, 0, 0, 0, 0, 50, 0],
        "7.png": [0, 45, 0, 0, 0, 45, 0, 0],
    }

    # 작업 목표 이미지 경로 정의
    # if bn < 3:
    #     target_file = "target/" + str(tn) + "_0.png"
    # else:
    #     target_file = "target/" + str(tn) + "_1.png"

    target_file = "target/" + str(tn) + ".png"
    target_img = Image.open(root + target_file)

    # 해당 타겟이미지의 시작 번호 계산
    cnt = len(xyr[0]) * len(xyr[1]) * len(xyr[2]) * bn

    # 작업 목표 물체의 폭,넓이(w_, h_)를 계산하기 위한 계산, 꼬리 부분을 상대적인 거리로 계산하기 위해.
    t_img = target_img
    t_img = t_img.rotate(0, center=(target_center_x, target_center_y), resample=Image.BICUBIC)
    trans_img = Image.new("RGBA", t_img.size)
    trans_img.paste(t_img, (0, 0))
    mask = trans_img.split()[3]
    ann = mask.getbbox()

    w_ = int((ann[2] - ann[0]) / 2)
    h_ = int((ann[3] - ann[1]) / 2)

    for px in xyr[0]:
        for py in xyr[1]:
            for rz in xyr[2]:
                # 타겟이미지(작업대상 이미지) 세팅----------------------------------------------------
                t_img = target_img
                t_img = t_img.rotate(rz, center=(target_center_x, target_center_y), resample=Image.BICUBIC)
                trans_img = Image.new("RGBA", t_img.size)
                trans_img.paste(t_img, (px, py))
                mask = trans_img.split()[3]
                ann = mask.getbbox()

                path_img = images_path + str(tn) + "_" + str(cnt).zfill(num_length) + ".jpg"
                path_txt = images_path + str(tn) + "_" + str(cnt).zfill(num_length) + ".txt"

                # 저장할 대상 경로 정의
                f = open(path_txt, 'w')
                label = [tn, ((ann[0] + ann[2]) / 2) / target_img.size[0], ((ann[1] + ann[3]) / 2) / target_img.size[1],
                         (ann[2] - ann[0]) / target_img.size[0], (ann[3] - ann[1]) / target_img.size[1]]

                f.write(str(tn) + " " + str(label[1]) + " " + str(label[2]) + " " + str(label[3]) + " " + str(
                    label[4]) + "\n")

                if tn<n_tail:
                    #꼬리부분 저장
                    [x1_w, x1_h] = bbox_point[target_file[7:]][:2]
                    [x2_w, x2_h] = bbox_point[target_file[7:]][2:4]
                    [x3_w, x3_h] = bbox_point[target_file[7:]][4:6]
                    [x4_w, x4_h] = bbox_point[target_file[7:]][6:]

                    # 꼬리 부분 좌표 계산.
                    box_x1 = target_center_x + int((-w_ + x1_w) * np.cos(-np.pi * rz / 180) - (-h_ + x1_h) * np.sin(
                        -np.pi * rz / 180)) + px
                    box_y1 = target_center_y + int((-w_ + x1_w) * np.sin(-np.pi * rz / 180) + (-h_ + x1_h) * np.cos(
                        -np.pi * rz / 180)) + py
                    box_x2 = target_center_x + int((w_ + x2_w) * np.cos(-np.pi * rz / 180) - (h_ + x2_h) * np.sin(
                        -np.pi * rz / 180)) + px
                    box_y2 = target_center_y + int((w_ + x2_w) * np.sin(-np.pi * rz / 180) + (h_ + x2_h) * np.cos(
                        -np.pi * rz / 180)) + py
                    box_x3 = target_center_x + int((w_ + x3_w) * np.cos(-np.pi * rz / 180) - (-h_ + x3_h) * np.sin(
                        -np.pi * rz / 180)) + px
                    box_y3 = target_center_y + int((w_ + x3_w) * np.sin(-np.pi * rz / 180) + (-h_ + x3_h) * np.cos(
                        -np.pi * rz / 180)) + py
                    box_x4 = target_center_x + int((x4_w - w_) * np.cos(-np.pi * rz / 180) - (h_ + x4_h) * np.sin(
                        -np.pi * rz / 180)) + px
                    box_y4 = target_center_y + int((x4_w - w_) * np.sin(-np.pi * rz / 180) + (h_ + x4_h) * np.cos(
                        -np.pi * rz / 180)) + py

                    # 회전된 꼬리부분의 바운딩 박스에서 x,y 중 제일 큰것을 계산
                    if abs(box_x2 - box_x1) < abs(box_x4 - box_x3):
                        box_x1, box_x2 = box_x3, box_x4
                    if abs(box_y2 - box_y1) < abs(box_y4 - box_y3):
                        box_y1, box_y2 = box_y3, box_y4

                    # 화면 밖으로 나가는 부분 처리
                    box_x1 = 0 if box_x1 < 0 else box_x1
                    box_x2 = 0 if box_x2 < 0 else box_x2
                    box_y1 = 0 if box_y1 < 0 else box_y1
                    box_y2 = 0 if box_y2 < 0 else box_y2
                    n = tn + n_target
                    tail_ann = (box_x1, box_y1, box_x2, box_y2)
                    label = [n, ((tail_ann[0] + tail_ann[2]) / 2) / target_img.size[0],
                             ((tail_ann[1] + tail_ann[3]) / 2) / target_img.size[1],
                             abs(tail_ann[2] - tail_ann[0]) / target_img.size[0], abs(tail_ann[3] - tail_ann[1]) / target_img.size[1]]
                    f.write(
                        str(n) + " " + str(label[1]) + " " + str(label[2]) + " " + str(label[3]) + " " + str(label[4]))
                f.close()

                # 바운딩 박스 테스트를 위한 시각화
                if(draw_on):
                    draw_img = ImageDraw.Draw(trans_img)
                    if tn<n_tail:
                        draw_img.rectangle(tail_ann, outline='blue')
                    draw_img.rectangle(ann, outline='red')

                # 배경 이미지 합성 ----------------------------------------------------------------
                output_img = Image.new("RGBA", t_img.size)
                output_img = Image.alpha_composite(output_img, b_img)

                # no target 오브젝트 합성 ----------------------------------------------------------------
                objects_ = objects.copy()
                np.random.shuffle(objects_)
                objects_ = objects_[:int(np.random.uniform(8, min(10,len(objects))))]

                for obj_ in objects_:
                    obj_ = obj_.rotate(int(np.random.uniform(0, 360)), center=(target_center_x, target_center_y),
                                       resample=Image.BICUBIC)

                    x_ = int(np.random.uniform(xyr[0][0], xyr[0][-1]))
                    y_ = int(np.random.uniform(xyr[1][0], xyr[1][-1]))
                    obj_img = Image.new("RGBA", t_img.size)
                    obj_img.paste(obj_, (x_, y_))
                    output_img = Image.alpha_composite(output_img, obj_img)

                # 타겟이미지 합성
                output_img = Image.alpha_composite(output_img, trans_img)
                output_img = output_img.convert("RGB")

                test_img = Image.new("RGBA", t_img.size)
                test_img.paste(output_img)
                test_img.convert('RGB').save(path_img, "JPEG", quality=100)

                cnt += 1


root = "./data/"                            # 이미지 리소스들이 담긴 폴더
path_background = root + "/background/"     # 배경이미지 경로
path_object = root + "/no_target/"          # 작업 외 이미지 경로
                                            # 작업 대상 물체의 경로는 augment_images 함수 내에서 정의함

data_path = "./dataset/"                    # 데이타셋 경로
images_path = data_path + "image_picking2/"          # 학습이미지 경로

if not os.path.exists(images_path):
    os.makedirs(images_path)

target_center_x = 640
target_center_y = 360


# 합성 영역 내 갯수 조절 변수
x_size = 11         # x 축 방향 갯수
y_size = 5          # y축 방향 갯수
r_size = 19         # 회전 갯수 + 1
n_target = 10       # 작업 대상 갯수
n_bg = 3            # 배경 갯수
n_tail=8

xyr_num = x_size * y_size * (r_size - 1)
padding=[200,200]
#최대 생성 갯수 (이미지 파일 + bbox 정보 텍스트파일)
total_num = xyr_num * n_target * n_bg * 2

if __name__ == '__main__':
    print("Start")

    # 배경이미지와 작업 외 물체 이미지 로드----------------------------------------------------
    back_list = os.listdir(path_background)
    background = []
    for b in range(0, len(back_list)):
        background.append(Image.open(path_background + str(b) + ".png").convert('RGBA'))

    obj_list = os.listdir(path_object)
    objects = []
    for b in range(0, len(obj_list)):
        objects.append(Image.open(path_object + str(b) + ".png").convert('RGBA'))
    # ----------------------------------------------------------------------------------------

    prog_bar = mp.Process(target=progress_bar, args=(images_path, total_num))
    all_process = []
    for b in range(0, 3):
        x_range = [-target_center_x + padding[0], target_center_x - padding[0], x_size]  # 30
        y_range = [-target_center_y + padding[1], target_center_y - padding[1], y_size]  # 15
        r_range = [0, 360, r_size]

        # 영역 내의 합성 좌표 리스트 생성
        x_list = np.linspace(*x_range).astype(np.int)
        y_list = np.linspace(*y_range).astype(np.int)
        r_list = np.linspace(*r_range).astype(np.int)[0:-1]
        xyr_list = [x_list, y_list, r_list]

        #augment_images(background[0],objects, xyr_list, 0, 35, len(str(total_num)))
        for t in range(0, n_target):
            processes = [mp.Process(target=augment_images, args=(background[b], objects, xyr_list, b, t, len(str(total_num*4*2)),  False))]
            for p in processes:
                all_process.append(p)


    prog_bar.start()
    for poc in all_process:
        poc.start()

    for poc in all_process:
        poc.join()
    #
    # all_process = []
    # for b in range(0, 3):
    #     x_range = [-target_center_x + padding[0], target_center_x - padding[0], x_size]  # 30
    #     y_range = [-target_center_y + padding[1], target_center_y - padding[1], y_size]  # 15
    #     r_range = [0, 360, r_size]
    #
    #     # 영역 내의 합성 좌표 리스트 생성
    #     x_list = np.linspace(*x_range).astype(np.int)
    #     y_list = np.linspace(*y_range).astype(np.int)
    #     r_list = np.linspace(*r_range).astype(np.int)[0:-1]
    #     xyr_list = [x_list, y_list, r_list]
    #
    #     for t in range(n_target//2, n_target):
    #         processes = [mp.Process(target=augment_images, args=(background[b], objects, xyr_list, b, t, len(str(total_num*4*2)),  False))]
    #         for p in processes:
    #             all_process.append(p)
    #
    # for poc in all_process:
    #     poc.start()
    #
    # for poc in all_process:
    #     poc.join()

    # yolo 학습에 필요한
    f = open("./dataset/train_picking2.txt", 'w')
    for num in range(n_target):
        cnt = 0
        dir_list = []
        for i in range(xyr_num * n_bg):
            path_img = os.path.abspath(".") + images_path[1:] + str(num) + "_" + str(cnt).zfill(len(str(total_num))+1) + ".jpg"
            dir_list.append(path_img)
            cnt += 1
        for d in dir_list:
            f.write(d + "\n")
    f.close()
    print("End")