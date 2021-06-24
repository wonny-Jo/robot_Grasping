import glob
import os

file_list = glob.glob("E:/RL_20_obj_dataset/RL_/*.png")

for idx, f in enumerate(file_list):
    os.rename(f, "E:/RL_20_obj_dataset/RL_/{}.png".format(idx))

a= 1


