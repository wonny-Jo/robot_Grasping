import os
import numpy as np
from PIL import Image, ImageDraw

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

data_path = "./dataset/"                    # 데이타셋 경로
images_path = data_path + "image/"          # 학습이미지 경로
output_path=data_path+"GT_image/"
if not os.path.exists(images_path):
    os.makedirs(images_path)

if __name__ == '__main__':
    print("Start")

    image_list = os.listdir(images_path)
    for i in range(len(image_list)//2):
        target_file = images_path + str(i) + ".jpg"
        target_img = Image.open(target_file).convert('RGBA')
        output_img = Image.new("RGBA", target_img.size)
        output_img = Image.alpha_composite(output_img, target_img)
        path_txt = images_path + str(i) + ".txt"
        f = open(path_txt, 'r')
        while True:
            temp = f.readline().split()
            if not temp:
                break
            target_n=int(temp[0])
            ann=[]
            for a in temp[1:]:
                ann.append(int(a))
            ann=tuple(ann)
            draw_img = ImageDraw.Draw(output_img)
            draw_img.rectangle(ann, outline='red')
        output_img = output_img.convert("RGB")
        path_img = output_path + str(i)+"_GT" + ".jpg"
        test_img = Image.new("RGBA", output_img.size)
        test_img.paste(output_img)
        test_img.convert('RGB').save(path_img, "JPEG", quality=100)

    print("End")