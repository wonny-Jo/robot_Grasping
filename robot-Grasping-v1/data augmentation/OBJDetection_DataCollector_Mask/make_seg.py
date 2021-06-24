import cv2
from glob import glob
import os
import numpy as np

path1 = 'C:/Users/incorl_robot/Desktop/seg_data/object_data/6'
path2 = 'C:/Users/incorl_robot/Desktop/seg_data/object_data/9'
path3 = 'C:/Users/incorl_robot/Desktop/seg_data/object_data/10'
path4 = 'C:/Users/incorl_robot/Desktop/seg_data/object_data/12'
add_path = 'C:/Users/incorl_robot/Desktop/seg_data/object_data_add'
save_path = 'C:/Users/incorl_robot/Desktop/seg_data/object_data_mod'
save_seg_path = 'C:/Users/incorl_robot/Desktop/seg_data/object_seg'

img_list1 = glob(path1 + '/*')
img_list2 = glob(path2 + '/*')
img_list3 = glob(path3 + '/*')
img_list4 = glob(path4 + '/*')

for file in img_list1:
    idx = os.path.basename(path1)
    file_name = os.path.basename(file)
    img1 = cv2.imread(file,cv2.IMREAD_UNCHANGED)
    seg = np.zeros([len(img1),len(img1[0])],dtype = np.uint8)
    a_info = np.array(img1)[0:, 0:, 3]
    location = np.where(a_info == 255)
    seg[location] = idx
    if os.path.exists(add_path + '/%s' % idx + '/' + file_name):
        img_add = cv2.imread(add_path + '/%s' % idx + '/' + file_name, cv2.IMREAD_UNCHANGED)
        a_info = np.array(img_add)[0:, 0:, 3]
        location = np.where(a_info == 255)
        seg[location] = 106
        img1[location] = img_add[location]
    if not os.path.exists(save_path + '/%s' % idx):
        os.makedirs(save_path + '/%s' % idx)
    cv2.imwrite(save_path + '/%s' % idx + '/' + file_name,img1)
    if not os.path.exists(save_seg_path + '/%s' % idx):
        os.makedirs(save_seg_path + '/%s' % idx)
    cv2.imwrite(save_seg_path + '/%s' % idx + '/' + file_name,seg)
    print('%s is saved.'%file_name)

for file in img_list2:
    idx = os.path.basename(path2)
    file_name = os.path.basename(file)
    img = cv2.imread(file,cv2.IMREAD_UNCHANGED)
    seg = np.zeros([len(img),len(img[0])],dtype = np.uint8)
    a_info = np.array(img)[0:, 0:, 3]
    location = np.where(a_info == 255)
    seg[location] = idx
    if os.path.exists(add_path + '/%s' % idx + '/' + file_name):
        img_add = cv2.imread(add_path + '/%s' % idx + '/' + file_name, cv2.IMREAD_UNCHANGED)
        a_info = np.array(img_add)[0:, 0:, 3]
        location = np.where(a_info == 255)
        seg[location] = 109
        img[location] = img_add[location]
    if not os.path.exists(save_path + '/%s' % idx):
        os.makedirs(save_path + '/%s' % idx)
    cv2.imwrite(save_path + '/%s' % idx + '/' + file_name,img)
    if not os.path.exists(save_seg_path + '/%s' % idx):
        os.makedirs(save_seg_path + '/%s' % idx)
    cv2.imwrite(save_seg_path + '/%s' % idx + '/' + file_name,seg)
    print('%s is saved.' % file_name)

for file in img_list3:
    idx = os.path.basename(path3)
    file_name = os.path.basename(file)
    img = cv2.imread(file,cv2.IMREAD_UNCHANGED)
    seg = np.zeros([len(img),len(img[0])],dtype = np.uint8)
    a_info = np.array(img)[0:, 0:, 3]
    location = np.where(a_info == 255)
    seg[location] = idx
    if os.path.exists(add_path + '/%s' % idx + '/' + file_name):
        img_add = cv2.imread(add_path + '/%s' % idx + '/' + file_name, cv2.IMREAD_UNCHANGED)
        a_info = np.array(img_add)[0:, 0:, 3]
        location = np.where(a_info == 255)
        seg[location] = 110
        img[location] = img_add[location]
    if not os.path.exists(save_path + '/%s' % idx):
        os.makedirs(save_path + '/%s' % idx)
    cv2.imwrite(save_path + '/%s' % idx + '/' + file_name,img)
    if not os.path.exists(save_seg_path + '/%s' % idx):
        os.makedirs(save_seg_path + '/%s' % idx)
    cv2.imwrite(save_seg_path + '/%s' % idx + '/' + file_name,seg)
    print('%s is saved.' % file_name)

for file in img_list4:
    idx = os.path.basename(path4)
    file_name = os.path.basename(file)
    img = cv2.imread(file,cv2.IMREAD_UNCHANGED)
    seg = np.zeros([len(img),len(img[0])],dtype = np.uint8)
    a_info = np.array(img)[0:, 0:, 3]
    location = np.where(a_info == 255)
    seg[location] = idx
    if os.path.exists(add_path + '/%s' % idx + '/' + file_name):
        img_add = cv2.imread(add_path + '/%s' % idx + '/' + file_name, cv2.IMREAD_UNCHANGED)
        a_info = np.array(img_add)[0:, 0:, 3]
        location = np.where(a_info == 255)
        seg[location] = 112
        img[location] = img_add[location]
    if not os.path.exists(save_path + '/%s' % idx):
        os.makedirs(save_path + '/%s' % idx)
    cv2.imwrite(save_path + '/%s' % idx + '/' + file_name,img)
    if not os.path.exists(save_seg_path + '/%s' % idx):
        os.makedirs(save_seg_path + '/%s' % idx)
    cv2.imwrite(save_seg_path + '/%s' % idx + '/' + file_name,seg)
    print('%s is saved.' % file_name)