import cv2
import os
import argparse
import numpy as np
from glob import glob

parser = argparse.ArgumentParser(description='')

parser.add_argument('--image_root',default='C:/Users/Park/Desktop/mask_rcnn/Labelmetool/image',help='image root path')
parser.add_argument('--seg_root',default='C:/Users/Park/Desktop/mask_rcnn/Labelmetool/seg',help='segmentation root path')
parser.add_argument('--save_dir',default='C:/Users/Park/Desktop/mask_rcnn/OBJDetection_DataCollector_Mask/obj_data',help='save directory')
args = parser.parse_args()

def LoadData(seg):
    name = os.path.basename(seg)
    image = cv2.imread(args.image_root + '/' + name, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2RGBA)
    seg_image = cv2.imread(seg,cv2.IMREAD_GRAYSCALE)

    width = len(seg_image[0])
    height = len(seg_image)
    pad = np.zeros([height,width,4],dtype=np.uint8)

    return image, seg_image, pad, name

def MakeData(img, seg, pad, name):
    obj_list = np.unique(seg)
    obj_list = np.delete(obj_list, np.where(obj_list == 0)[0])

    for idx in obj_list:
        pad_mod = pad.copy()

        pix = np.where(seg == idx)
        pad_mod[pix] = img[pix]

        cv2.imshow('pad',pad_mod)
        cv2.waitKey(1)

        savepath = args.save_dir + '/%d' % idx
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        savefile = savepath + '/%s' % name
        cv2.imwrite(savefile, pad_mod)

def main():
    seg_list = glob(args.seg_root + '/*')

    for seg in seg_list:
        img, seg, pad, name = LoadData(seg)
        MakeData(img,seg,pad,name)
        print('%s is saved.' % name)

if __name__ == '__main__':
    main()