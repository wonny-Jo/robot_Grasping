import cv2
from glob import glob
import os
import argparse
import random
import numpy as np
from xml.etree.ElementTree import Element, ElementTree
from datetime import datetime
import obj_configs
from time import sleep
import copy

parser = argparse.ArgumentParser(description='')

parser.add_argument('--back_root',default='C:/Users/Park/Desktop/mask_rcnn/OBJDetection_DataCollector_Mask/background',help='background root path')
parser.add_argument('--obj_root',default='C:/Users/Park/Desktop/mask_rcnn/OBJDetection_DataCollector_Mask/obj_data',help='object root path')
parser.add_argument('--obj_add_root',default='C:/Users/incorl/Desktop/OBJDetection_DataCollector_Mask/object_seg',help='object root path')
parser.add_argument('--save_img_dir',default='C:/Users/Park/Desktop/mask_rcnn/OBJDetection_DataCollector_Mask/result/img',help='save directory')
parser.add_argument('--save_xml_dir',default='C:/Users/Park/Desktop/mask_rcnn/OBJDetection_DataCollector_Mask/result/xml',help='save directory')
parser.add_argument('--save_seg_dir',default='C:/Users/Park/Desktop/mask_rcnn/OBJDetection_DataCollector_Mask/result/seg',help='save directory')
parser.add_argument('--min_num_of_object',default=5,help='min number of objects per image')
parser.add_argument('--max_num_of_object',default=10,help='max number of objects per image')
args = parser.parse_args()

back_root = args.back_root
back_list = glob(back_root+"/*")
obj_root = args.obj_root
save_img_dir = args.save_img_dir
save_xml_dir = args.save_xml_dir
save_seg_dir = args.save_seg_dir
min_num_of_obj = args.min_num_of_object
max_num_of_obj = args.max_num_of_object

def LoadBackground(back_list):

    image = np.array(cv2.imread(back_list[random.randrange(0,len(back_list))]))
    width = len(image[0])
    height = len(image)
    image_info = [image,width,height]

    return image_info

def PasteObject_add(back_info, obj_list, seg, obj_pos):

    obj_size_ratio = 1
    obj_idx = os.path.basename(obj_list)
    obj_name = obj_configs.ChangeName(obj_idx)
    if obj_name == None:
        print('Name error happened.')
        while True:
            sleep(1)
    obj_image_list = glob(obj_list + "/*")
    random.shuffle(obj_image_list)
    file_name = os.path.basename(obj_image_list[0])

    obj_image = cv2.imread(obj_image_list[0], cv2.IMREAD_UNCHANGED)
    obj_width = len(obj_image[0])
    obj_height = len(obj_image)

    obj_seg = cv2.imread(args.obj_add_root + '/' + obj_idx + '/' + file_name, cv2.IMREAD_GRAYSCALE)

    image_ch = random.randint(0, 5)
    if image_ch == 0:
        obj_image = cv2.flip(obj_image, 1)
        obj_seg = cv2.flip(obj_seg, 1)
    elif image_ch == 1:
        obj_image = cv2.flip(obj_image, 2)
        obj_seg = cv2.flip(obj_seg, 2)
    elif image_ch == 2:
        obj_image = cv2.flip(obj_image, 1)
        obj_seg = cv2.flip(obj_seg, 1)
        obj_image = cv2.flip(obj_image, 2)
        obj_seg = cv2.flip(obj_seg, 2)

    back_ratio = min(back_info[1], back_info[2])
    obj_ratio = min(obj_width, obj_height)

    obj_re_width = int(obj_width * (back_ratio / obj_ratio) * obj_size_ratio)
    obj_re_height = int(obj_height * (back_ratio / obj_ratio) * obj_size_ratio)

    obj_image = cv2.resize(obj_image, (obj_re_width, obj_re_height))

    a_info = np.array(obj_image)[0:, 0:, 3]

    location = np.where(a_info == 255)
    location = list(location)
    obj_image = cv2.cvtColor(obj_image, cv2.COLOR_RGBA2RGB)

    obj_center = [int((max(location[0]) + min(location[0])) / 2), int((max(location[1]) + min(location[1])) / 2)]

    location_back_y = location[0] - obj_center[0] + obj_pos[0]
    location_back_x = location[1] - obj_center[1] + obj_pos[1]

    location_back = [location_back_y, location_back_x]

    width = back_info[1]
    height = back_info[2]
    ymin = 0
    ymax = height
    xmin = 0
    xmax = width

    delete = []
    for i in range(0, len(location_back[0])):
        if location_back[0][i] >= ymax or location_back[0][i] < ymin or location_back[1][i] >= xmax or location_back[1][
            i] < xmin:
            delete.append(i)
    location_back[0] = np.delete(location_back[0], delete)
    location_back[1] = np.delete(location_back[1], delete)
    location[0] = np.delete(location[0], delete)
    location[1] = np.delete(location[1], delete)

    image = back_info[0]
    image[location_back] = obj_image[location]

    seg[location_back] = obj_seg[location]
    obj_idx = int(obj_idx)

    return image, seg, obj_name, obj_idx

def PasteObject(back_info, obj_list, seg, obj_pos):

    obj_size_ratio = 1
    obj_idx = os.path.basename(obj_list)
    obj_name = obj_configs.ChangeName(obj_idx)
    if obj_name == None:
        print('Name error happened.')
        while True:
            sleep(1)
    obj_image_list = glob(obj_list + "/*")
    random.shuffle(obj_image_list)

    obj_image = cv2.imread(obj_image_list[0],cv2.IMREAD_UNCHANGED)
    obj_width = len(obj_image[0])
    obj_height = len(obj_image)

    image_ch = random.randint(0,5)
    if image_ch == 0:
        obj_image = cv2.flip(obj_image, 1)
    elif image_ch == 1:
        obj_image = cv2.flip(obj_image, 2)
    elif image_ch == 2:
        obj_image = cv2.flip(obj_image, 1)
        obj_image = cv2.flip(obj_image, 2)

    back_ratio = min(back_info[1], back_info[2])
    obj_ratio = min(obj_width, obj_height)

    obj_re_width = int(obj_width * (back_ratio / obj_ratio) * obj_size_ratio)
    obj_re_height = int(obj_height * (back_ratio / obj_ratio) * obj_size_ratio)

    obj_image = cv2.resize(obj_image,(obj_re_width,obj_re_height))

    a_info = np.array(obj_image)[0:,0:,3]

    location = np.where(a_info == 255)
    location = list(location)
    obj_image = cv2.cvtColor(obj_image, cv2.COLOR_RGBA2RGB)

    obj_center = [int((max(location[0]) + min(location[0])) / 2), int((max(location[1]) + min(location[1])) / 2)]

    location_back_y = location[0] - obj_center[0] + obj_pos[0]
    location_back_x = location[1] - obj_center[1] + obj_pos[1]

    location_back = [location_back_y, location_back_x]

    width = back_info[1]
    height = back_info[2]
    ymin = 0
    ymax = height
    xmin = 0
    xmax = width

    delete = []
    for i in range(0, len(location_back[0])):
        if location_back[0][i] >= ymax or location_back[0][i] < ymin or location_back[1][i] >= xmax or location_back[1][
            i] < xmin:
            delete.append(i)
    location_back[0] = np.delete(location_back[0], delete)
    location_back[1] = np.delete(location_back[1], delete)
    location[0] = np.delete(location[0], delete)
    location[1] = np.delete(location[1], delete)

    image = back_info[0]
    image[location_back] = obj_image[location]

    seg[location_back] = obj_idx
    obj_idx = int(obj_idx)

    return image, seg, obj_name, obj_idx

def Make_obj_pose():
    obj_pos_y = random.randint(97,636)
    obj_pos_x = random.randint(79,896)
    obj_pos = [obj_pos_y,obj_pos_x]
    return obj_pos

def MakeInfo(filename, seg, info):
    width = len(seg[0])
    height = len(seg)
    data = []
    for id in info:
        mask = np.where(seg == id[0])
        if len(mask[0]) != 0:
            bbox = [min(mask[1]), min(mask[0]), max(mask[1]), max(mask[0])]
            name = id[1]
            data.append([os.path.join(filename, '.png'), width, height, name, id[0], bbox])
    return data

def save_to_xml(data, savedir, filename, width, height):

    root = Element("annotation")
    node1 = Element("folder")
    node1.text = 'train'
    root.append(node1)

    node2 = Element("filename")
    node2.text = filename+'.png'
    root.append(node2)

    node3 = Element("path")
    node3.text = savedir+'/'+filename+'.png'
    root.append(node3)

    node4 = Element("source")
    node4_1 = Element("database")
    node4_1.text = 'Unknown'
    node4.append(node4_1)
    root.append(node4)

    node5 = Element("size")
    node5_1 = Element("width")
    node5_1.text = str(width)
    node5_2 = Element("height")
    node5_2.text = str(height)
    node5_3 = Element("depth")
    node5_3.text = str(3)
    node5.append(node5_1)
    node5.append(node5_2)
    node5.append(node5_3)
    root.append(node5)

    node6 = Element("segmented")
    node6.text = str(0)
    root.append(node6)

    for i in range(len(data)):

        name = data[i][3]
        bbox = data[i][5]

        node7 = Element("object")
        node7_1 = Element("name")
        node7_1.text = name
        node7_2 = Element("pose")
        node7_2.text = 'Unspecified'
        node7_3 = Element("truncated")
        node7_3.text = str(0)
        node7_4 = Element("difficult")
        node7_4.text = str(0)
        node7_5 = Element("bndbox")
        node7_5_1 = Element("xmin")
        node7_5_1.text = str(bbox[0])
        node7_5_2 = Element("ymin")
        node7_5_2.text = str(bbox[1])
        node7_5_3 = Element("xmax")
        node7_5_3.text = str(bbox[2])
        node7_5_4 = Element("ymax")
        node7_5_4.text = str(bbox[3])
        node7_5.append(node7_5_1)
        node7_5.append(node7_5_2)
        node7_5.append(node7_5_3)
        node7_5.append(node7_5_4)
        node7.append(node7_1)
        node7.append(node7_2)
        node7.append(node7_3)
        node7.append(node7_4)
        node7.append(node7_5)
        root.append(node7)

    def indent(elem, level=0):  # 자료 출처 https://goo.gl/J8VoDK
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    indent(root)

    save_name = savedir + '/' + filename+'.xml'
    ElementTree(root).write(save_name)

def main():

    _pid = os.getpid()
    count = 0

    while True:

        count += 1
        nowtime = datetime.now()
        strNowTime = nowtime.strftime("%y%m%d_%H%M%S")

        # read image

        back_info = LoadBackground(back_list)
        seg = np.zeros([back_info[2], back_info[1]],dtype=np.uint8)

        # append object image

        num_of_obj = random.randrange(min_num_of_obj,max_num_of_obj+1)
        info = []

        filename = '%d_%s_%d' % (_pid, strNowTime, count)

        obj_list = glob(obj_root + "/*")
        random.shuffle(obj_list)

        for i in range(num_of_obj):

            obj_pos = Make_obj_pose()
            # if os.path.basename(obj_list[i]) == '6':
            #     image, seg, name, idx = PasteObject_add(back_info, obj_list[i], seg, obj_pos)
            #     # info.append([106, name])
            # elif os.path.basename(obj_list[i]) == '9':
            #     image, seg, name, idx = PasteObject_add(back_info, obj_list[i], seg, obj_pos)
            #     # info.append([109, name])
            # elif os.path.basename(obj_list[i]) == '10':
            #     image, seg, name, idx = PasteObject_add(back_info, obj_list[i], seg, obj_pos)
            #     # info.append([110, name])
            # elif os.path.basename(obj_list[i]) == '12':
            #     image, seg, name, idx = PasteObject_add(back_info, obj_list[i], seg, obj_pos)
            #     # info.append([112, name])
            # else:
            #     image, seg, name ,idx= PasteObject(back_info, obj_list[i], seg, obj_pos)
            # random.shuffle(obj_list)
            image, seg, name, idx = PasteObject(back_info, obj_list[i], seg, obj_pos)
            info.append([idx,name])

        data = MakeInfo(filename, seg, info)

        if not os.path.exists(save_xml_dir):
            os.makedirs(save_xml_dir)
        if not os.path.exists(save_img_dir):
            os.makedirs(save_img_dir)
        if not os.path.exists(save_seg_dir):
            os.makedirs(save_seg_dir)

        save_to_xml(data, save_xml_dir, filename, back_info[1], back_info[2])
        cv2.imwrite(save_img_dir + '/' + filename + '.jpg', image)
        cv2.imwrite(save_seg_dir + '/' + filename + '.png', seg)

        # seg_vis = seg * int(255/120)

        image_mod = copy.deepcopy(image)
        for i in range(len(data)):
            image_mod = cv2.rectangle(image_mod, (data[i][5][0], data[i][5][1]), (data[i][5][2], data[i][5][3]), (0, 0, 255), 3)
            image_mod = cv2.putText(image_mod, data[i][3], (data[i][5][0], data[i][5][1] - 2), cv2.FONT_ITALIC, 0.7, (0, 0, 255),2)

        if not os.path.exists(save_img_dir + '_ans_image'):
            os.makedirs(save_img_dir + '_ans_image')
        cv2.imwrite(save_img_dir + '_ans_image/' + filename + '_rectangled.jpg', image_mod)

        print(filename + ' is saved.')

        if count == 50000:
            break

    print('finished')

if __name__=='__main__':
    main()