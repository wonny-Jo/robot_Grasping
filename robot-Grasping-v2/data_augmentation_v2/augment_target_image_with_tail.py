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

        # x 축 좌, y축 상, x축 우, y축 하 ||  x 축 상, y축 상, x축 좌, y축 하
    bbox_point = {
        # 좌상  우하  우상  좌하
        #[3, 0], [-3, -33], [-3, 0], [3, -33]
        "0_0.png": [3,0,3,0,-3,-33,-3,-33,-3,0,-3,0,3,-33,3,-33],
        "0_1.png": [6,0,6,0,-6,-95,-6,-95,-6,0,-6,0,6,-95,6,-95],
        "1_0.png": [3,0,3,0,-3,-22,-3,-22,-4,0,-4,0,3,-22,3,-22],
        "1_1.png": [12,0,12,0,-12,-67,-12,-67,-12,0,-12,0,12,-67,12,-67],
        "2_0.png": [40,25,40,25,0,-25,0,-25,0,25,0,25,40,-25,40,-25],
        "2_1.png": [140,90,140,90,0,-90,0,-90,0,90,0,90,140,-90, 140,-90],
        "3_0.png": [0,80,0,80,0,0,0,0,0,80,0,80,0,0,0,0],
        "3_1.png": [0,240,0,240,0,0,0,0,0,240,0,240,0,0,0,0],
        "4_0.png": [0,165,0,165,0,0,0,0,0,165,0,165,0,0,0,0],
        "4_1.png": [368,10,368,10,0,0,0,0,0,10,0,10,368,0,368,0],
        "5_0.png": [9,146,9,146,0,0,0,0,0,146,0,146,9,0,9,0],
        "5_1.png": [344,25,344,25,0,0,0,0,0,25,0,25,344,0,344,0],
        "6_0.png": [0,30,0,30,-52,-25,-52,-25,-52,30,-52,30,0,-25,0,-25],
        "6_1.png": [0,80,0,80,-45,-78,-45,-78,-45,80,-45,80,0,-78,0,-78],
        "7_0.png": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "7_1.png": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "8_0.png": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "8_1.png": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "9_0.png": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "9_1.png": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "10_0.png": [0,9,0,9,-47,-4,-47,-4,-47,9,-47,9,0,-4,0,-4],
        "10_1.png": [152,20,152,20,0,-25,0,-25,0,20,0,20,152,-25,152,-25],
        "11_0.png": [0,14,0,14,-49,-9,-49,-9,-49,14,-49,14,0,-9,0,-9],
        "11_1.png": [161,45,161,45,0,-47,0,-47,0,45,0,45,161,-47,161,-47],
        "12_0.png": [47,128,47,128,-12,0,-12,0,-12,128,-12,128,47,0,47,0],
        "12_1.png": [257,38,257,38,0,-65,0,-65,0,38,0,38,257,-65,257,-65],
        "13_0.png": [15,186,15,186,-29,0,-29,0,-29,186,-29,186,15,0,15,0],
        "13_1.png": [344,35,344,35,0,-32,0,-32,0,35,0,35,344,-32,344,-32],
    }

    # 작업 목표 이미지 경로 정의
    if bn < 3:
        target_file = "target/" + str(tn) + "_0.png"
    else:
        target_file = "target/" + str(tn) + "_1.png"

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

                # 저장할 대상 경로 정의
                path_img = images_path + str(tn) + "_" + str(cnt).zfill(num_length) + ".jpg"
                path_txt = images_path + str(tn) + "_" + str(cnt).zfill(num_length) + ".txt"

                # 타겟이미지(작업대상 이미지) 세팅----------------------------------------------------
                t_img = target_img
                t_img = t_img.rotate(rz, center=(target_center_x, target_center_y), resample=Image.BICUBIC)
                trans_img = Image.new("RGBA", t_img.size)
                trans_img.paste(t_img, (px, py))
                mask = trans_img.split()[3]
                ann = mask.getbbox()

                [x1_w, x1_h, y1_w, y1_h] = bbox_point[target_file[7:]][:4]
                [x2_w, x2_h, y2_w, y2_h] = bbox_point[target_file[7:]][4:8]
                [x3_w, x3_h, y3_w, y3_h] = bbox_point[target_file[7:]][8:12]
                [x4_w, x4_h, y4_w, y4_h] = bbox_point[target_file[7:]][12:]

                # 꼬리 부분 좌표 계산.
                box_x1 = 424 + int((-w_ + x1_w) * np.cos(-np.pi * rz / 180) - (-h_ + x1_h) * np.sin(
                    -np.pi * rz / 180)) + px
                box_y1 = 240 + int((-w_ + y1_w) * np.sin(-np.pi * rz / 180) + (-h_ + y1_h) * np.cos(
                    -np.pi * rz / 180)) + py
                box_x2 = 424 + int((w_ + x2_w) * np.cos(-np.pi * rz / 180) - (h_ + x2_h) * np.sin(
                    -np.pi * rz / 180)) + px
                box_y2 = 240 + int((w_ + y2_w) * np.sin(-np.pi * rz / 180) + (h_ + y2_h) * np.cos(
                    -np.pi * rz / 180)) + py
                box_x3 = 424 + int((w_ + x3_w) * np.cos(-np.pi * rz / 180) - (-h_ + x3_h) * np.sin(
                    -np.pi * rz / 180)) + px
                box_y3 = 240 + int((w_ + y3_w) * np.sin(-np.pi * rz / 180) + (-h_ + y3_h) * np.cos(
                    -np.pi * rz / 180)) + py
                box_x4 = 424 + int((x4_w - w_) * np.cos(-np.pi * rz / 180) - (h_ + x4_h) * np.sin(
                    -np.pi * rz / 180)) + px
                box_y4 = 240 + int((y4_w - w_) * np.sin(-np.pi * rz / 180) + (h_ + y4_h) * np.cos(
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

                # 바운딩 박스 테스트를 위한 시각화
                if(draw_on):
                    draw_img = ImageDraw.Draw(trans_img)
                    draw_img.rectangle((box_x1, box_y1, box_x2, box_y2), outline='blue')
                    draw_img.rectangle(ann, outline='red')
                # trans_img.show()

                # 배경 이미지 합성 ----------------------------------------------------------------
                output_img = Image.new("RGBA", t_img.size)
                output_img = Image.alpha_composite(output_img, b_img)

                # 오브젝트 합성 ----------------------------------------------------------------
                objects_ = objects.copy()
                np.random.shuffle(objects_)
                objects_ = objects_[:int(np.random.uniform(7, len(objects)))]

                for obj_ in objects_:
                    size_ = int(np.random.uniform(1, 3))
                    # size_ = 1

                    obj_ = obj_.rotate(int(np.random.uniform(0, 360)), center=(target_center_x, target_center_y),
                                       resample=Image.BICUBIC)
                    obj_ = obj_.resize((obj_.size[0] * size_, obj_.size[1] * size_))

                    # 랜덤 위치 합성
                    x_ = int(np.random.uniform(xyr[0][0]/size_, xyr[0][-1]/size_))
                    y_ = int(np.random.uniform(xyr[1][0]/size_, xyr[1][-1]/size_))
                    obj_img = Image.new("RGBA", t_img.size)

                    # 비작업대상 이미지를 확대 시켰다면 x,y 좌표 조절을 해야함. 기존의 848,480 크기 이미지랑 중심 좌표값이 달라지기 때문.
                    if size_ > 1:
                        obj_img.paste(obj_, (
                        -int(obj_.size[0] / 2 - 424) + x_ * size_, -int(obj_.size[1] / 2 - 240) + y_ * size_))
                    else:
                        obj_img.paste(obj_, (x_, y_))
                    output_img = Image.alpha_composite(output_img, obj_img)

                # 타겟이미지 합성
                output_img = Image.alpha_composite(output_img, trans_img)
                output_img = output_img.convert("RGB")

                test_img = Image.new("RGBA", t_img.size)
                test_img.paste(output_img)
                test_img.convert('RGB').save(path_img, "JPEG", quality=100)

                f = open(path_txt, 'w')
                label = [tn, ((ann[0] + ann[2]) / 2) / target_img.size[0], ((ann[1] + ann[3]) / 2) / target_img.size[1],
                         (ann[2] - ann[0]) / target_img.size[0], (ann[3] - ann[1]) / target_img.size[1]]

                f.write(str(tn) + " " + str(label[1]) + " " + str(label[2]) + " " + str(label[3]) + " " + str(
                    label[4]) + "\n")

                ann = (box_x1, box_y1, box_x2, box_y2)
                label = [tn, ((ann[0] + ann[2]) / 2) / target_img.size[0], ((ann[1] + ann[3]) / 2) / target_img.size[1],
                         abs(ann[2] - ann[0]) / target_img.size[0], abs(ann[3] - ann[1]) / target_img.size[1]]


                #꼬리 부분 라벨 지정
                if tn == 2 or tn == 3:
                    n = 16
                else:
                    n = tn+14

                f.write(str(n) + " " + str(label[1]) + " " + str(label[2]) + " " + str(label[3]) + " " + str(label[4]))

                f.close()
                cnt += 1



root = "./data/"                            # 이미지 리소스들이 담긴 폴더
path_background = root + "/background/"     # 배경이미지 경로
path_object = root + "/no_target/"          # 작업 외 이미지 경로
                                            # 작업 대상 물체의 경로는 augment_images 함수 내에서 정의함

data_path = "./dataset/"                    # 데이타셋 경로
images_path = data_path + "image/"          # 학습이미지 경로

if not os.path.exists(images_path):
    os.makedirs(images_path)

target_center_x = 424
target_center_y = 240


# 합성 영역 내 갯수 조절 변수
x_size = 20          # x 축 방향 갯수
y_size = 12          # y축 방향 갯수
r_size = 36         # 회전 갯수 + 1
n_target = 14           # 작업 대상 갯수
n_bg = 6               # 배경 갯수

xyr_num = x_size * y_size * (r_size - 1)

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
        objects.append(Image.open(path_object + "object" + str(b) + ".png").convert('RGBA'))
    # ----------------------------------------------------------------------------------------
    # x_range = [-424 + 150, 424 - 140, x_size]  # 40 584-424 = 160
    # y_range = [-240 + 150, 240 - 150, y_size]  # 50 416-240 = 176
    x_range = [-424 + 190, 424 - 190, x_size]  # 40 584-424 = 160
    y_range = [-240 + 190, 240 - 190, y_size]  # 50 416-240 = 176
    r_range = [0, 360, r_size]

    # # 영역 내의 합성 좌표 리스트 생성
    # x_list = np.linspace(*x_range).astype(np.int)
    # y_list = np.linspace(*y_range).astype(np.int)
    # r_list = np.linspace(*r_range).astype(np.int)[0:-1]
    # xyr_list = [x_list, y_list, r_list]


    prog_bar = mp.Process(target=progress_bar, args=(images_path, total_num))
    all_process = []
    for b in range(0, 6):
        # 0~2 은 작은 타겟 이미지, 3~5은 큰 타겟 이미지

        # x,y,r 길이에 따른 좌표 값 생성
        if b < 3:
            # x_range = [-424 + 40, 424 - 40, x_size]  # 30
            # y_range = [-240 + 50, 240 - 50, y_size]  # 15
            x_range = [-424 + 80, 424 - 80, x_size]  # 30
            y_range = [-240 + 80, 240 - 80, y_size]  # 15
        else:
            # x_range = [-424+150, 424-140, x_size] # 40 584-424 = 160
            # y_range = [-240+150, 240-150, y_size] # 50 416-240 = 176
            x_range = [-424+190, 424-190, x_size] # 40 584-424 = 160
            y_range = [-240+190, 240-190, y_size] # 50 416-240 = 176
        r_range = [0, 360, r_size]

        # 영역 내의 합성 좌표 리스트 생성
        x_list = np.linspace(*x_range).astype(np.int)
        y_list = np.linspace(*y_range).astype(np.int)
        r_list = np.linspace(*r_range).astype(np.int)[0:-1]
        xyr_list = [x_list, y_list, r_list]


        # augment_images(background[0],objects, xyr_list, 0, 0, len(str(total_num)))
        for t in range(0, n_target):
            processes = [mp.Process(target=augment_images, args=(background[b], objects, xyr_list, b, t, len(str(total_num*4*2)),True))]
            for p in processes:
                all_process.append(p)


    prog_bar.start()
    for poc in all_process:
        poc.start()

    for poc in all_process:
        poc.join()


    # yolo 학습에 필요한
    f = open("./dataset/train.txt", 'w')
    for num in range(n_target):
        cnt = 0
        dir_list = []
        for i in range(xyr_num * n_bg):
            path_img = os.path.abspath(".") + images_path[1:] + str(num) + "_" + str(cnt).zfill((len(str(total_num))+1)) + ".jpg"
            dir_list.append(path_img)
            cnt += 1
        for d in dir_list:
            f.write(d + "\n")
    f.close()
    print("End")