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

def init_YOLO(config=None,weight=None,meta=None):
    netMain = None
    metaMain = None
    altNames = None

    # global metaMain, netMain, altNames
    configPath = config  # Path to cfg #210715
    weightPath = weight  # Path to weights
    metaPath = meta  # Path to meta data
    # configPath = "./model/yolov4_210722.cfg"  # Path to cfg #210715
    # weightPath = "./model/yolov4_210722.weights"  # Path to weights
    # metaPath = "./model/210722.data"  # Path to meta data

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
    frame_width = int(1280)  # Returns the width and height of capture video
    frame_height = int(720)

    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(frame_width, frame_height,
                                       3)  # Create image according darknet for compatibility of network
    #return network, class_names, darknet_image
    network, class_names, class_colors= darknet.load_network(configPath,metaPath,weightPath,batch_size=1)
    return netMain, class_names, darknet_image
    #return netMain, metaMain, darknet_image

def YOLO(model_yolo, output_img,threshold=0.6):
    netMain, metaMain, darknet_image = model_yolo
    # Load the input frame and write output frame.
    frame_read = np.array(output_img.convert("RGB"))
    frame_rgb = cv2.cvtColor(frame_read,
                             cv2.COLOR_BGR2RGB)  # Convert frame into RGB from BGR and resize accordingly

    darknet.copy_image_from_bytes(darknet_image, frame_rgb.tobytes())  # Copy that frame bytes to darknet_image

    # detections = darknet.detect_image(netMain, class_names, darknet_image,
    #                                   thresh=0.40)
    detections = darknet.detect_image(netMain, metaMain, darknet_image,
                                    thresh=threshold)#0.75  # Detection occurs at this line and return detections, for customize we can change the threshold.

    return detections, frame_rgb

