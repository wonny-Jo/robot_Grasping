import os
import numpy as np
import multiprocessing as mp

from PIL import Image
from tqdm import tqdm

def progress_bar(s_path, total):
    p_bar = tqdm(total=total)
    num_out, pre_out = 0, 0
    while num_out < int(p_bar.total):
        num_out = len(os.listdir(s_path))
        p_bar.update(num_out-pre_out)
        pre_out = num_out

# def labeling_mask(mask, label):
#     data = mask.load()
#     for y in range(mask.size[1]):
#         for x in range(mask.size[0]):
#             if(data[x, y] != 0):
#                 data[x, y] = label
#             else:
#                 data[x, y] = 0

def augment_images(b_img,objects, xyr, bn, tn, num_length):

    # 작업 목표 이미지 경로 정의
    if bn < 3:
        target_file = "target/" + str(tn) + "_0.png"
    else:
        target_file = "target/" + str(tn) + "_1.png"

    target_img = Image.open(root + target_file)


    #해당 타겟이미지의 시작 번호 계산
    cnt = len(xyr[0]) * len(xyr[1]) * len(xyr[2]) * bn

    for px in xyr[0]:
        for py in xyr[1]:
            for rz in xyr[2]:

                # 저장할 대상 경로 정의
                path_img = images_path + str(tn) + "_" + str(cnt).zfill(num_length) + ".jpg"
                path_txt = images_path + str(tn) + "_" + str(cnt).zfill(num_length) + ".txt"


                # 타겟이미지(작업대상 이미지) 세팅----------------------------------------------------
                t_img = target_img
                t_img = t_img.rotate(rz, center=(target_center_x, target_center_y), resample=Image.BICUBIC) # 타겟 이미지 회전

                trans_img = Image.new("RGBA", t_img.size)   # 4차원 이미지 작업공간 생성
                trans_img.paste(t_img, (px, py))            # 타겟 이미지를 원하는 x,y 좌표에 붙임
                mask = trans_img.split()[3]                 # 타겟 이미지의 mask 채널 분리
                ann = mask.getbbox()                        # 분리된 마스크 채널에서 바운딩박스 계산.

                # 배경 이미지 합성 ----------------------------------------------------------------
                output_img = Image.new("RGBA", t_img.size)
                output_img = Image.alpha_composite(output_img, b_img)

                # 오브젝트 합성 ----------------------------------------------------------------
                objects_ = objects.copy()
                np.random.shuffle(objects_)
                objects_ = objects_[:int(np.random.uniform(7, len(objects)))]   # 비작업대상 물체 랜덤 추출

                for obj_ in objects_:
                    size_ = int(np.random.uniform(1, 3)) # 비작업대상 이미지 확대 배수 (1~3배 크기)

                    obj_ = obj_.rotate(int(np.random.uniform(0, 360)), center=(target_center_x, target_center_y), resample=Image.BICUBIC)   # 랜덤 회전
                    obj_ = obj_.resize((obj_.size[0]*size_ , obj_.size[1]*size_))   # 이미지 크기 조절 (확대)

                    # 랜덤 위치 합성
                    x_ = int(np.random.uniform(xyr[0][0]/size_, xyr[0][-1]/size_))
                    y_ = int(np.random.uniform(xyr[1][0]/size_, xyr[1][-1]/size_))
                    obj_img = Image.new("RGBA", t_img.size)

                    # 비작업대상 이미지를 확대 시켰다면 x,y 좌표 조절을 해야함. 기존의 848,480 크기 이미지랑 중심 좌표값이 달라지기 때문.
                    if size_ > 1:
                        obj_img.paste(obj_, (-int(obj_.size[0] / 2 - 424)+x_*size_, -int(obj_.size[1] / 2 - 240)+y_*size_))
                    else:
                        obj_img.paste(obj_, ( x_,  y_))
                    output_img = Image.alpha_composite(output_img, obj_img)

                # 타겟이미지 합성
                output_img = Image.alpha_composite(output_img, trans_img)
                output_img = output_img.convert("RGB")


                # 최종 결과물 저장.
                test_img = Image.new("RGBA", t_img.size)
                test_img.paste(output_img)
                test_img.convert('RGB').save(path_img, "JPEG", quality=100)

                # 학습시 필요한 라벨과 바운딩 박스 저장.
                f = open(path_txt, 'w')
                label = [tn, ((ann[0] + ann[2]) / 2) / target_img.size[0], ((ann[1] + ann[3]) / 2) / target_img.size[1],
                         (ann[2] - ann[0]) / target_img.size[0], (ann[3] - ann[1]) / target_img.size[1]]

                f.write(str(tn) + " " + str(label[1]) + " " + str(label[2]) + " " + str(label[3]) + " " + str(label[4]))
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
x_size = 10#7          # x 축 방향 갯수
y_size = 6#4          # y축 방향 갯수
r_size = 36#4          # 회전 갯수 + 1
n_target = 14           # 작업 대상 갯수
n_bg = 6               # 배경 갯수

xyr_num = x_size * y_size * (r_size - 1)

#최대 생성 갯수 (이미지 파일 + bbox 정보 텍스트파일)
total_num = xyr_num * n_target * n_bg * 2

if __name__ == '__main__':

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


    prog_bar = mp.Process(target=progress_bar, args=(images_path, total_num))
    all_process = []

    for b in range(0, 6):
        # 0~1 은 작은 타겟 이미지, 2~3은 큰 타겟 이미지

        # x,y,r 길이에 따른 좌표 값 생성
        # 물체 이미지가 화면 밖으로 안나가도록 +- 조절을 잘 해야함.
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
            processes = [mp.Process(target=augment_images, args=(background[b], objects, xyr_list, b, t, len(str(total_num*4*2))))]
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
            path_img = os.path.abspath(".") + images_path[1:] + str(num) + "_" + str(
                cnt).zfill(len(str(total_num)) + 1) + ".jpg"
            dir_list.append(path_img)
            cnt += 1
        for d in dir_list:
            f.write(d + "\n")
    f.close()
    print("Process End")