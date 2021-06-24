import os, time, cv2, threading
import numpy as np
import matplotlib.pyplot as plt

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
        self.detections = None  # yolo 인식 결과를 보여주기 위한 변수
        self.img_yolo = None    # 실시간이미지+yolo인식결과 박스 적용된 이미지
        self.ON = False         # 윈도우 창 출력 버튼

        self.color_dict = {
            'HDMI': [0, 0, 255], 'USB_C': [0, 0, 0], 'HUB1': [255, 0, 0], 'HUB2': [0, 255, 0]
        }

    def run(self):
        self.state = True
        try:
            while self.state == True:
                tmp_img = self.rs.get_img("rgb")
                self.img = tmp_img.copy()
                if self.detections != None:

                    for detection in self.detections:
                        x, y, w, h = detection[2][0], \
                                     detection[2][1], \
                                     detection[2][2], \
                                     detection[2][3]
                        name_tag = detection[0]
                        #name_tag = str(detection[0].decode())

                        for name_key, color_val in self.color_dict.items():
                            if name_key == name_tag:
                                color = color_val
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

                self.img_yolo = tmp_img.copy()

                if self.ON:
                    cv2.imshow("view_gripper", tmp_img)
                    cv2.waitKey(1)
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