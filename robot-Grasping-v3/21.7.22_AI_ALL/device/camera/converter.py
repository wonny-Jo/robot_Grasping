from PIL import Image
import cv2
import numpy as np

def rgb2bin(image, ratio):
    #dir = '.\ImageData\img0.png'
    #im = cv2.imread(dir, cv2.IMREAD_COLOR)
    im = cv2.resize(image, (0,0), fx=ratio, fy=ratio)
    #cv2.imshow("image",im)
    #return im/255
'''
def rgb2bin(dir,width, height):
    #입력 세팅
    dir = '.\ImageData\img0.png'
    im = Image.open(dir)

    re_im = im.resize((width, height))
    re_im_frame = np.array(re_im.getdata())

    #바이너리 변환 ---------------------------------------------
    image_bin = []

    for rgb in range(3):
        tmp_data = []
        for h in range(height):
            tmp_list = []
            for w in range(width):
                tmp_bin = re_im_frame[width*h+w][rgb]/255
                #tmp_bin = "{0:0>8b}".format(re_im_frame[width*h+w][rgb])
                tmp_list.append(tmp_bin)
            tmp_data.append(tmp_list)
        image_bin.append(tmp_data)

    return np.array(image_bin)
'''

def pose2bin(pose):
    pose = pose * 10000
    pose_bin = []

    for i in range(6):
        tmp_bin = "{0:0>16b}".format(abs(int(pose[i])))
        if (pose[i] < 0):
            tmp_bin = str(1) + tmp_bin
        else :
            tmp_bin = str(0) + tmp_bin
        pose_bin.append(tmp_bin)

    return np.array(pose_bin)

def bin2rgb(image_bin):
    width = np.size(image_bin, 2)
    height = np.size(image_bin, 1)

    return image_bin*255

    '''
    for rgb in range(3):
        tmp_data =[]
        for i in range(image_len):
            tmp_data.append(int(image_bin[rgb][i],2))
        image.append(tmp_data)
    '''
    #return np.transpose(np.array(image))

def bin2pose(pose_bin):
    pose = []

    for i in range(6):
        tmp_data = int(pose_bin[i][1:],2)
        if pose_bin[i][0] == '1':  # pose의 값이 음수라면
            tmp_data = -tmp_data

        pose.append(tmp_data)

    return np.array(pose)/10000


#int("{0:b}".format(10), 2)