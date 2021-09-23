from model.function import *

model_picking = init_YOLO("./model/model_picking.cfg","./model/model_picking.weights","./model/model_picking.data")
model_picking2 = init_YOLO("./model/model_picking2.cfg","./model/model_picking2.weights","./model/model_picking2.data")
model_mp = init_YOLO("./model/model_mp.cfg","./model/model_mp.weights","./model/model_mp.data")
model_cwb = init_YOLO("./model/model_cwb.cfg","./model/model_cwb.weights","./model/model_cwb.data")
model_bd = init_YOLO("./model/model_bd.cfg","./model/model_bd.weights","./model/model_bd.data")
model_hp = init_YOLO("./model/model_hp.cfg","./model/model_hp.weights","./model/model_hp.data")