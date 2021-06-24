from torch.lib import *
import darknet

#from .vae_model import *
#from .FC_model import *

import pickle, os, cv2
import numpy as np

device = "cuda:0"
netMain = None
metaMain = None
altNames = None

# def load_model(path_save):
#     model_vae = VAE(True, 1, device)
#     model_fc = FC(2, 1)
#
#     model_vae = model_vae.cuda()
#     model_vae.to(device)
#     model_vae.load_state_dict(torch.load(path_save[0]))
#     model_vae.eval()
#
#     model_fc = model_fc.cuda()
#     model_fc.to(device)
#     model_fc.load_state_dict(torch.load(path_save[1]))
#     model_fc.eval()
#
#     with open(path_save[2], 'rb') as f:
#         model_umap = pickle.load(f)
#
#     return model_vae, model_fc, model_umap

def init_YOLO():
    netMain = None
    metaMain = None
    altNames = None

    # global metaMain, netMain, altNames
    '''
    configPath = "./model/yolov4-new_train.cfg"  # Path to cfg
    weightPath = "./model/yolov4-new_train.weights"  # Path to weights
    metaPath = "./model/new_train.data"  # Path to meta data
    '''
    configPath = "./model/yolov4-210428.cfg"  # Path to cfg
    weightPath = "./model/yolov4-210428.weights"  # Path to weights
    metaPath = "./model/210428.data"  # Path to meta data

    if not os.path.exists(configPath):  # Checks whether file exists otherwise return ValueError
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath) + "`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath) + "`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath) + "`")
    if netMain is None:  # Checks the metaMain, NetMain and altNames. Loads it in script
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    #network, class_names, class_colors= darknet.load_network(configPath,metaPath,weightPath,batch_size=1)
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    frame_width = int(848)  # Returns the width and height of capture video
    frame_height = int(480)

    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(frame_width, frame_height,
                                       3)  # Create image according darknet for compatibility of network
    #return network, class_names, darknet_image
    network, class_names, class_colors= darknet.load_network(configPath,metaPath,weightPath,batch_size=1)
    return netMain, class_names, darknet_image
    #return netMain, metaMain, darknet_image

def YOLO(model_yolo, rs):
    netMain, metaMain, darknet_image = model_yolo
    # Load the input frame and write output frame.
    frame_read = np.array(rs.get_img("rgb"))
    frame_rgb = cv2.cvtColor(frame_read,
                             cv2.COLOR_BGR2RGB)  # Convert frame into RGB from BGR and resize accordingly

    darknet.copy_image_from_bytes(darknet_image, frame_rgb.tobytes())  # Copy that frame bytes to darknet_image

    # detections = darknet.detect_image(netMain, class_names, darknet_image,
    #                                   thresh=0.40)
    detections = darknet.detect_image(netMain, metaMain, darknet_image,
                                    thresh=0.30)  # Detection occurs at this line and return detections, for customize we can change the threshold.

    return detections, frame_rgb

def  find_target(detections, target, grasp_point):
    nearest_dist = None
    nearest_target = None
    w_,h_ = None, None
    if len(detections) == 0:
        return None, None
    else:
        nearest_dist = np.sqrt(848 ** 2 + 480 ** 2)

        for detection in detections:
            if detection[0] != target:
                continue

            x, y, w, h = detection[2][0], \
                         detection[2][1], \
                         detection[2][2], \
                         detection[2][3]
            dist = np.sqrt((x - grasp_point[0]) ** 2 + (y - grasp_point[1]) ** 2)

            if nearest_dist > dist:
                nearest_dist = dist
                nearest_target = [int(x), int(y)]
                w_, h_ = w, h
        return nearest_target, [w_,h_]
