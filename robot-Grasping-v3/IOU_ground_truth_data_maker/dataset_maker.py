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

def augment_images(b_img, objects,cnt):
    object_name = {
        "0.png": 'HDMI',
        "1.png": 'USB_C',
        "2.png": 'HDMI_HUB',
        "3.png": 'USB_C_HUB',
        "4.png": 'WHITE_PLUG',
        "5.png": 'BLACK_PLUG',
        "6.png": 'BLACK_MULTITAP',
        "7.png": 'GREEN_MULTITAP',
        "8.png": 'ORANGE_PENCIL',
        "9.png": 'BLUE_PENCIL',
        "10.png": 'BLUE_SHARPENER',
        "11.png": 'ORANGE_SHARPENER',
        "12.png": 'GREEN_DESK_CLEANER',
        "13.png": 'BLUE_DESK_CLEANER',
        "14.png": 'RED_CUP',
        "15.png": 'PINK_CUP',
        "16.png": 'SMALL_BOX',
        "17.png": 'BIG_BOX',
        "18.png": 'PINK_STAPLER',
        "19.png": 'STAN_STAPLER',
        "20.png": 'GLUE_PEN',
        "21.png": 'GLUE_STICK',
        "22.png": 'BLACK_MARKER',
        "23.png": 'RED_MARKER',
        "24.png": 'BLACK_NAMEPEN',
        "25.png": 'SILVER_NAMEPEN',
        "26.png": 'MILK',
        "27.png": 'YOGURT',
        "28.png": 'SMALL_USB',
        "29.png": 'BIG_USB',
        "30.png": 'SMALL_ERASER',
        "31.png": 'BIG_ERASER',
        "32.png": 'GREEN_BOOK',
        "33.png": 'BLUE_BOOK',
        "34.png": 'BLACK_FILE_HOLDER',
        "35.png": 'PINK_FILE_HOLDER',
        "36.png": 'BLACK_KEYBOARD',
        "37.png": 'PINK_KEYBOARD',
        "38.png": 'GREEN_HOLDER',
        "39.png": 'APRICOT_BOTTLE',
        "40.png": 'SILVER_PENCILCASE',
        "41.png": 'RED_PENCILCASE',
        "42.png": 'WHITE_BIN',
        "43.png": 'STAN_BIN',
        "44.png": 'BLACK_TAPE',
        "45.png": 'WHITE_TAPE',
        "46.png": 'GREY_BOTTLE',
        "47.png": 'BLACK_HOLDER',
        "48.png": 'LIGHT_DRAWER',
        "49.png": 'DARK_DRAWER',
        "50.png": 'GREY_CUP',
        "51.png": 'GREEN_CUP',
        "52.png": 'BLUE_CUP',
        "53.png": 'PURPLE_CUP',
        "54.png": 'SILVER_CUP',
        "55.png": 'WHITE_BOX',
        "56.png": 'RED_BOX',
        "57.png": 'YELLOW_BOX',
        "58.png": 'GREEN_BOX',
        "59.png": 'PINK_BOX',
    }
    path_txt = images_path + str(cnt) + ".txt"
    f = open(path_txt, 'w')
    checkPos=[]
    # 배경 이미지 합성 ----------------------------------------------------------------
    output_img = Image.new("RGBA", b_img.size)
    output_img = Image.alpha_composite(output_img, b_img)
    for object_num in objects:
        target_file = "target/" + str(object_num) + ".png"
        target_img = Image.open(root + target_file)
        #Image.open(path_object + str(b) + ".png").convert('RGBA')

        while 1:
            xNum=np.random.randint(0,x_size)
            yNum=np.random.randint(0,y_size)
            if [xNum,yNum] not in checkPos:
                checkPos.append([xNum,yNum])
                break
        rNum=np.random.randint(0,r_size)
        x=x_list[xNum]
        y=y_list[yNum]
        r=r_list[rNum]

        # 타겟이미지(작업대상 이미지) 세팅----------------------------------------------------
        t_img = target_img
        t_img = t_img.rotate(r, center=(target_center_x, target_center_y), resample=Image.BICUBIC)
        trans_img = Image.new("RGBA", t_img.size)
        trans_img.paste(t_img, (x, y))
        mask = trans_img.split()[3]
        ann = mask.getbbox()
        label = [object_num, ann[0],ann[1],ann[2],ann[3]]
        f.write(str(label[0])+ " " + str(label[1])+ " " + str(label[2])+ " " + str(label[3])+ " " + str(label[4])+ "\n")
        output_img = Image.alpha_composite(output_img, trans_img)
        # 바운딩 박스 테스트를 위한 시각화
    f.close()
    output_img = output_img.convert("RGB")
    path_img = images_path + str(cnt) + ".jpg"
    test_img = Image.new("RGBA", t_img.size)
    test_img.paste(output_img)
    test_img.convert('RGB').save(path_img, "JPEG", quality=100)


root = "./data/"                            # 이미지 리소스들이 담긴 폴더
path_background = root + "/background/"     # 배경이미지 경로
data_path = "./dataset/"                    # 데이타셋 경로
images_path = data_path + "image/"          # 학습이미지 경로
target_path=root +"/target/"

if not os.path.exists(images_path):
    os.makedirs(images_path)

target_center_x = 640
target_center_y = 360


# 합성 영역 내 갯수 조절 변수
x_size = 7         # x 축 방향 갯수
y_size = 5          # y축 방향 갯수
r_size = 19         # 회전 갯수 + 1
n_target = 60       # 작업 대상 갯수
n_bg = 3            # 배경 갯수

padding=[200,200]
#최대 생성 갯수 (이미지 파일 + bbox 정보 텍스트파일)
total_num = 200

x_range = [-target_center_x + padding[0], target_center_x - padding[0], x_size]  # 30
y_range = [-target_center_y + padding[1], target_center_y - padding[1], y_size]  # 15
r_range = [0, 360, r_size]
x_list = np.linspace(*x_range).astype(np.int)
y_list = np.linspace(*y_range).astype(np.int)
r_list = np.linspace(*r_range).astype(np.int)#[0:-1]

if __name__ == '__main__':
    print("Start")

    # 배경이미지와 작업 외 물체 이미지 로드----------------------------------------------------
    back_list = os.listdir(path_background)
    background = []
    for b in range(len(back_list)):
        background.append(Image.open(path_background + str(b) + ".png").convert('RGBA'))

    obj_list = os.listdir(target_path)
    objects = []
    for b in range(n_target):
        objects.append(str(b))

    # prog_bar = mp.Process(target=progress_bar, args=(images_path, total_num))
    # all_process = []

    for i in range(total_num):
        objects_ = objects.copy()
        np.random.shuffle(objects_)
        objects_ = objects_[:int(np.random.uniform(8, min(10, len(objects))))]

        bNum=int(np.random.uniform(0,3))
        augment_images(background[bNum],objects_,i)
        #augment_images(background[0],objects, xyr_list, 0, 35, len(str(total_num)))
        # processes = [mp.Process(target=augment_images, args=(background[bNum], objects_, i, True))]
        # for p in processes:
        #     all_process.append(p)


    # prog_bar.start()
    # for poc in all_process:
    #     poc.start()
    #
    # for poc in all_process:
    #     poc.join()

    print("End")