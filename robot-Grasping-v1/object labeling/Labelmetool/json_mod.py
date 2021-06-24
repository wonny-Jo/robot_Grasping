#!/usr/bin/env python
# -*- coding: utf8 -*-
import json
import os
import sys

import codecs

# TODO : Assignment path : "C:\\Users\\HelloWorld\\PycharmProjects\\json_modify\\"

before_path = "C:/Users/Park/Desktop/mask_rcnn/Labelmetool/json/"  # json Folder
after_path = "C:/Users/Park/Desktop/mask_rcnn/Labelmetool/new_json/"  # new json Folder

# Obj_list 24 Object
obj_list = ['Eraser_big','Usb_Big','Stapler_pink']

# new json folder make
if not os.path.exists(after_path):
    os.mkdir(after_path)

# Make New json files
len_list = len(os.listdir(before_path))

for c, file_name in enumerate(os.listdir(before_path)):
    print("\rCurrent Progress : {},  ({}/{})".format(file_name, c + 1, len_list), end = '', flush=True)

    with open(before_path + file_name, 'rt', encoding='utf-8') as f:
        typo_error = False
        json_data = json.loads(f.read())
        new_json_data = json_data.copy()

        # TODO
        new_json_data['imgWidth'] = 1280
        new_json_data['imgHeight'] = 720

        new_json_data['objects'] = []

        for idx in range(len(json_data['shapes'])):
            try:
                obj_list.index(json_data['shapes'][idx]['label'])

            except ValueError:
                typo_error = True
                typo = input("\nTyping Error : %s -> " %(json_data['shapes'][idx]['label']))
                json_data['shapes'][idx]['label'] = typo
                print("Complete revised\n")

            new_json_data['objects'].append({'id' : obj_list.index(json_data['shapes'][idx]['label']) + 25,
                                             'label':json_data['shapes'][idx]['label'],
                                             'polygon':json_data['shapes'][idx]['points']})

    if typo_error:
        with open(before_path + file_name, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent="  ")

    with open(after_path + file_name, 'w', encoding='utf-8') as a_f:
        json.dump(new_json_data, a_f, ensure_ascii=False, indent="  ")

print("\nSuccessfully completed!!\n")