import os, time, cv2, threading
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

class t_ur(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.state = False
        self.ur = None
        self.tempList = []

    def run(self):
        self.state = True
        try:
            while self.state == True:
                temp = self.ur.ur_rob.rtmon.get_all_data()['tcp']
                self.tempList.append(temp)
                self.tempList = self.tempList[-1000:]
                # plt.plot(self.tempList)
                # plt.pause(0.1)
                time.sleep(0.1)
        except:
            raise

# class t_rs(threading.Thread):
#     def __init__(self):
#         threading.Thread.__init__(self)
#         self.state = False
#         self.rs = None
#         self.img = []
#
#
#     def run(self):
#         self.state = True
#         try:
#             while self.state == True:
#                 t1 = time.time()
#                 tmp_img = self.rs.get_img("", "rgb")
#                 self.img = tmp_img.copy()
#                 cv2.imshow("image", tmp_img)
#                 cv2.waitKey(1)
#         except:
#             raise


class t_rs(threading.Thread):
    def __init__(self, rs_):
        threading.Thread.__init__(self)
        self.state = False
        self.rs = rs_           # 리얼센스정보
        self.img = []           # 실시간 이미지 저장 공간
        self.detections = []  # yolo 인식 결과를 보여주기 위한 변수
        self.img_yolo = None    # 실시간이미지+yolo인식결과 박스 적용된 이미지
        self.ON = False         # 윈도우 창 출력 버튼
        self.screen_pause = False
        self.targets= [
            'RED_CUP', 'PINK_CUP','SMALL_BOX','BIG_BOX','PINK_STAPLER','STAN_STAPLER','GLUE_PEN','GLUE_STICK',
            'BLACK_TAPE','WHITE_TAPE',
            'GREY_CUP','GREEN_CUP','BLUE_CUP','PURPLE_CUP','SILVER_CUP','WHITE_BOX','RED_BOX','YELLOW_BOX',
            'GREEN_BOX','PINK_BOX',
            'WHITE_BIN','STAN_BIN','MILK','YOGURT',
            'LIGHT_DRAWER','DARK_DRAWER', 'SMALL_USB','BIG_USB','SMALL_ERASER', 'BIG_ERASER',
            'APRICOT_BOTTLE','GREY_BOTTLE',
            'GREEN_HOLDER','BLACK_HOLDER', 'BLACK_MARKER','RED_MARKER','BLACK_NAMEPEN','SILVER_NAMEPEN',
            'GREEN_BOOK','BLUE_BOOK','BLACK_FILE_HOLDER','PINK_FILE_HOLDER','BLACK_KEYBOARD','PINK_KEYBOARD',
            'GREEN_DESK_CLEANER', 'BLUE_DESK_CLEANER',
            'ORANGE_PENCIL', 'ORANGE_SHARPENER', 'SILVER_PENCILCASE',
            'BLUE_PENCIL', 'BLUE_SHARPENER', 'RED_PENCILCASE',
            'BLACK_PLUG', 'BLACK_MULTITAP',
            'WHITE_PLUG', 'GREEN_MULTITAP',
            'HDMI_HUB', 'HDMI',
            'USB_C_HUB', 'USB_C',
            'HDMI_TAIL', 'USB_C_TAIL', 'USB_C_HUB_TAIL', 'HDMI_HUB_TAIL', 'WHITE_PLUG_TAIL', 'BLACK_PLUG_TAIL',
            'BLACK_MULTITAP_TAIL',
            'GREEN_MULTITAP_TAIL', 'ORANGE_PENCIL_TAIL', 'BLUE_PENCIL_TAIL', 'BLUE_SHARPENER_TAIL',
            'ORANGE_SHARPENER_TAIL',
            'GREEN_DESK_CLEANER_TAIL', 'BLUE_DESK_CLEANER_TAIL', 'RED_CUP_TAIL', 'PINK_CUP_TAIL', 'SMALL_BOX_TAIL',
            'BIG_BOX_TAIL', 'PINK_STAPLER_TAIL', 'STAN_STAPLER_TAIL', 'GLUE_PEN_TAIL', 'GLUE_STICK_TAIL',
            'BLACK_MARKER_TAIL', 'RED_MARKER_TAIL', 'BLACK_NAMEPEN_TAIL', 'SILVER_NAMEPEN_TAIL', 'MILK_TAIL',
            'YOGURT_TAIL',
            'SMALL_USB_TAIL', 'BIG_USB_TAIL', 'SMALL_ERASER_TAIL', 'BIG_ERASER_TAIL', 'GREEN_BOOK_TAIL',
            'BLUE_BOOK_TAIL',
            'BLACK_FILE_HOLDER_TAIL', 'PINK_FILE_HOLDER_TAIL', 'BLACK_KEYBOARD_TAIL', 'PINK_KEYBOARD_TAIL',
            'GREEN_HOLDER_TAIL',
            'GREY_CUP_TAIL','GREEN_CUP_TAIL','BLUE_CUP_TAIL','PURPLE_CUP_TAIL','SILVER_CUP_TAIL','WHITE_BOX_TAIL',
            'RED_BOX_TAIL','YELLOW_BOX_TAIL'
        ]

        self.color_dict = {
            'HDMI': [0, 0, 255], 'USB_C': [0, 0, 255],'HDMI_HUB': [0, 255, 0], 'USB_C_HUB': [0, 255, 0],
            'ORANGE_PENCIL':[255,255,0],'BLUE_PENCIL':[255,255,0],'BLUE_SHARPENER':[255,0,255],
            'ORANGE_SHARPENER': [255,0,255],'SILVER_PENCILCASE':[0,255,255],'RED_PENCILCASE':[0,255,255],
            'WHITE_PLUG': [125, 0, 0],'BLACK_PLUG':[125, 0, 0],
            'BLACK_MULTITAP':[0,125,0],'GREEN_MULTITAP':[0,125,0],
            'GREY_CUP': [255, 0, 0], 'GREEN_CUP': [255, 0, 0], 'BLUE_CUP': [255, 0, 0], 'PURPLE_CUP': [255, 0, 0],
            'SILVER_CUP': [255, 0, 0], 'WHITE_BOX': [255, 0, 0], 'RED_BOX': [255, 0, 0], 'YELLOW_BOX': [255, 0, 0],
            'GREEN_BOX': [255, 0, 0], 'PINK_BOX': [255, 0, 0],
            'RED_CUP':[0,0,125], 'PINK_CUP':[0,0,125], 'SMALL_BOX':[0,0,125], 'BIG_BOX':[0,0,125], 'PINK_STAPLER':[0,0,125],
            'STAN_STAPLER':[0,0,125], 'GLUE_PEN':[0,0,125], 'GLUE_STICK':[0,0,125],
            'BLACK_TAPE':[0,0,125], 'WHITE_TAPE':[0,0,125],
            'WHITE_BIN':[125,125,0], 'STAN_BIN':[125,125,0], 'MILK':[125,0,125], 'YOGURT':[125,0,125],
            'LIGHT_DRAWER':[200,0,200], 'DARK_DRAWER':[200,0,200], 'SMALL_USB':[0,200,100], 'BIG_USB':[0,200,100],
            'SMALL_ERASER':[0,200,100], 'BIG_ERASER':[0,200,100],
            'APRICOT_BOTTLE':[100,100,255], 'GREY_BOTTLE':[100,100,255],
            'GREEN_HOLDER':[200,100,0], 'BLACK_HOLDER':[200,200,0], 'BLACK_MARKER':[150,0,0], 'RED_MARKER':[150,0,0],
            'BLACK_NAMEPEN':[150,0,0], 'SILVER_NAMEPEN':[150,0,0],
            'GREEN_BOOK':[50,0,0], 'BLUE_BOOK':[50,0,0], 'BLACK_FILE_HOLDER':[50,0,0], 'PINK_FILE_HOLDER':[50,0,0],
            'BLACK_KEYBOARD':[50,0,0], 'PINK_KEYBOARD':[50,0,0],
            'GREEN_DESK_CLEANER':[0,50,0], 'BLUE_DESK_CLEANER':[0,0,50],

            #############
            'USB_C_HUB_TAIL':[0,50,0],'USB_C_TAIL':[0,50,0],
            'HDMI_TAIL': [0,50,0], 'HDMI_HUB_TAIL':[0,50,0],
            'WHITE_PLUG_TAIL':[0,50,0], 'BLACK_PLUG_TAIL':[0,50,0],
            'BLACK_MULTITAP_TAIL':[0,50,0], 'GREEN_MULTITAP_TAIL':[0,50,0],
            'BLACK_KEYBOARD_TAIL': [0, 50, 0],'BIG_BOX_TAIL':[0,50,0],
            'GREY_CUP_TAIL': [0, 50, 0], 'GREEN_CUP_TAIL': [0, 50, 0], 'BLUE_CUP_TAIL': [0, 50, 0],
            'PURPLE_CUP_TAIL': [0, 50, 0], 'SILVER_CUP_TAIL': [0, 50, 0], 'WHITE_BOX_TAIL': [0, 50, 0],
            'RED_BOX_TAIL': [0, 50, 0], 'YELLOW_BOX_TAIL': [0, 50, 0],
            'ORANGE_PENCIL_TAIL': [0, 50, 0], 'BLUE_PENCIL_TAIL': [0, 50, 0], 'BLUE_SHARPENER_TAIL': [0, 50, 0],
            'ORANGE_SHARPENER_TAIL': [0, 50, 0], 'GREEN_DESK_CLEANER_TAIL': [0, 50, 0], 'BLUE_DESK_CLEANER_TAIL': [0, 50, 0],
            'RED_CUP_TAIL': [0, 50, 0], 'PINK_CUP_TAIL': [0, 50, 0], 'SMALL_BOX_TAIL': [0, 50, 0],
            'PINK_STAPLER_TAIL': [0, 50, 0], 'STAN_STAPLER_TAIL': [0, 50, 0], 'GLUE_PEN_TAIL': [0, 50, 0],
            'GLUE_STICK_TAIL': [0, 50, 0], 'BLACK_MARKER_TAIL': [0, 50, 0], 'RED_MARKER_TAIL': [0, 50, 0],
            'BLACK_NAMEPEN_TAIL': [0, 50, 0], 'SILVER_NAMEPEN_TAIL': [0, 50, 0], 'MILK_TAIL': [0, 50, 0],
            'YOGURT_TAIL': [0, 50, 0], 'SMALL_USB_TAIL': [0, 50, 0], 'BIG_USB_TAIL': [0, 50, 0],
            'SMALL_ERASER_TAIL': [0, 50, 0], 'BIG_ERASER_TAIL': [0, 50, 0], 'GREEN_BOOK_TAIL': [0, 50, 0],
            'BLUE_BOOK_TAIL': [0, 50, 0], 'BLACK_FILE_HOLDER_TAIL': [0, 50, 0],
            'PINK_FILE_HOLDER_TAIL': [0, 50, 0], 'PINK_KEYBOARD_TAIL': [0, 50, 0], 'GREEN_HOLDER_TAIL': [0, 50, 0],
        }

    def run(self):
        self.state = True
        try:
            while self.state == True:
                if self.screen_pause:
                    tmp_img=deepcopy(self.img)
                else:
                    tmp_img = self.rs.get_img("rgb")
                    self.img = deepcopy(tmp_img)

                #for target in self.targets:
                for detection in self.detections:
                    if detection[0] in self.targets:
                        target=detection[0]
                        x, y, w, h = detection[2][0], \
                                     detection[2][1], \
                                     detection[2][2], \
                                     detection[2][3]

                        if target in self.color_dict.keys():
                            color = self.color_dict[target]
                            xmin, ymin, xmax, ymax = convertBack(
                                float(x), float(y), float(w), float(h))
                            pt1 = (xmin, ymin)
                            pt2 = (xmax, ymax)
                            cv2.rectangle(tmp_img, pt1, pt2, color, 1)
                            cv2.putText(tmp_img,
                                        #detection[0].decode() +
                                        detection[0] +
                                        " [" + str(round(float(detection[1]), 2)) + "]",
                                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        color, 2)
                if self.ON:
                    cv2.imshow("view_gripper", tmp_img)
                    cv2.waitKey(1)
                self.img_yolo = tmp_img.copy()
        except:
            raise

class t_cam(threading.Thread):
    def __init__(self, device = 1):
        threading.Thread.__init__(self)
        self.capture = cv2.VideoCapture(device)
        if not (self.capture.isOpened()):
            self.capture.open(device)


    def run(self):
        while True:
            _, image = self.capture.read()
            cv2.imshow("External viewpoint",image)
            cv2.waitKey(1)
            self.img = image.copy()

class t_visual(threading.Thread):
    def __init__(self, t1_, t2_):
        threading.Thread.__init__(self)
        import pickle
        with open("./model_outputs/usb-hub_rotation/umap/save-ep104_vector", 'rb') as f:
            self.vectors = pickle.load(f)

        self.t1 = t1_
        self.t2 = t2_
        self.empty_image = np.zeros([480 ,848 ,3]).astype(np.uint8)
        self.target_prev = self.empty_image
        self.img_fig = None
        self.state_detection = False
        self.state_rotation = False
        self.state_fig = False
        self.pos_target = None
        self.bbox_target = None
        self.vec_tar = None

    def run(self):
        fig = plt.figure(num=0, figsize=(4, 4))
        ax = fig.add_subplot(1, 1, 1)
        color = ["Hub", "USB-white", "USB-red", "USB-black"]
        for i in range(4):
            ax.scatter(self.vectors[324 * i:324 * (i + 1), 0], self.vectors[324 * i:324 * (i + 1), 1],
                       label=color[i])
        ax.legend()
        while True:
            self.img2 = cv2.resize(self.t2.img, dsize=(1280,960), interpolation=cv2.INTER_AREA)
            self.img1 = self.t1.img_yolo

            # if self.state_detection == False:
            #         self.img1 = self.target_prev
            # else:
            #     if self.bbox_target[0] > self.bbox_target[1]:
            #         sz_img = int(self.bbox_target[0]/2)
            #     else:
            #         sz_img = int(self.bbox_target[1]/2)
            #
            #     y1 = np.clip(self.pos_target[1] - sz_img, a_min=0, a_max=480)
            #     y2 = np.clip(self.pos_target[1] + sz_img, a_min=0, a_max=480)
            #     x1 = np.clip(self.pos_target[0] - sz_img, a_min=0, a_max=848)
            #     x2 = np.clip(self.pos_target[0] + sz_img, a_min=0, a_max=848)
            #
            #     target_img = self.t1.img.copy()[y1:y2,x1:x2, :]
            #     target_img = cv2.resize(target_img, dsize=(240, 240), interpolation=cv2.INTER_AREA)
            #     self.img1 = target_img.astype(np.uint8)
            #     self.target_prev = self.img1


            if self.state_rotation == False:
                self.img_fig = self.empty_image
                # self.img_fig = self.img_fig
            else:
                if self.state_fig == True:
                    # color = ["Hub", "USB-white", "USB-red", "USB-black"]
                    ax.scatter(self.vec_tar[0], self.vec_tar[1], s=200, color='black', marker='x', label="Target")

                    fig.canvas.draw()
                    # convert canvas to image
                    self.img_fig = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    # fig.clf()
                    self.img_fig = self.img_fig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    self.img_fig = cv2.resize(self.img_fig , dsize=(480, 480), interpolation=cv2.INTER_AREA)
                    tmp = np.zeros([480,184,3]).astype(np.uint8)
                    self.img_fig = cv2.hconcat([tmp, self.img_fig, tmp])
                    self.state_fig = False

            img = cv2.vconcat([self.img1, self.img_fig])
            img = cv2.hconcat([img, self.img2])
            cv2.imshow('Vizuallization', img)
            cv2.waitKey(1)