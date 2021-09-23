from torch.lib import *
from model.function import *

from PIL import Image, ImageDraw
from torchvision import transforms
from config import *
from math import *
import math3d as m3d


data_path = "./dataset/"                    # 데이타셋 경로
images_path = data_path + "image/"          # 학습이미지 경로
output_path=data_path+"detection_image/"

def target_detection(output_img,threshold=0.7):
    detect_picking, _ = YOLO(model_picking, output_img, threshold=threshold)
    detect_picking2, _ = YOLO(model_picking2, output_img, threshold=threshold)
    detect_cwb, _ = YOLO(model_cwb, output_img, threshold=threshold)
    detect_bd, _ = YOLO(model_bd, output_img, threshold=threshold)
    detect_hp, _ = YOLO(model_hp, output_img, threshold=threshold)
    detect_mp, _ = YOLO(model_mp, output_img, threshold=threshold)
    detections=detect_mp+detect_bd+detect_cwb+detect_picking+detect_picking2+detect_hp
    return detections

def task_run():
    image_list = os.listdir(images_path)
    for i in range(len(image_list)//2):
        target_file = images_path + str(i) + ".jpg"
        target_img = Image.open(target_file).convert('RGBA')
        output_img = Image.new("RGBA", target_img.size)
        output_img = Image.alpha_composite(output_img, target_img)

        detections= target_detection(output_img)
        for detection in detections:
            if '_TAIL' in detection[0]:
                continue
            temp=detection[2][:4]
            w,h=temp[2]//2,temp[3]//2
            ann=tuple([temp[0]-w,temp[1]-h,temp[0]+w,temp[1]+h])
            draw_img = ImageDraw.Draw(output_img)
            draw_img.rectangle(ann, outline='blue')
        output_img = output_img.convert("RGB")
        path_img = output_path + str(i)  + ".jpg"
        test_img = Image.new("RGBA", output_img.size)
        test_img.paste(output_img)
        test_img.convert('RGB').save(path_img, "JPEG", quality=100)

