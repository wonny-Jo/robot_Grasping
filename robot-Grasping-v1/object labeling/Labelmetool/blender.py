import bpy
import numpy as np
import os

import bpy
import numpy as np
import os

rotation = 36 * np.pi / 180
name = '008.wrap_0'
path = 'C:/Users/incorl/Desktop/RL_Project/object_data_gloss'
index = '8'
if not os.path.exists(path + '/' + index + '/' + name):
    os.makedirs(path + '/' + index + '/' + name)

bpy.ops.object.select_by_type(extend=False, type='MESH')
for i in range(10):
    bpy.ops.transform.rotate(value=rotation, orient_axis='X')
    for j in range(10):
        bpy.ops.transform.rotate(value=rotation, orient_axis='Y')
        for k in range(10):
            bpy.ops.transform.rotate(value=rotation, orient_axis='Z')
            bpy.context.scene.render.filepath = path + '/' + index + '/' + name + '/%d_%d_%d.jpg' % (i, j, k)
            bpy.ops.render.render(write_still=True)

bpy.ops.object.camera_add(location=(0,0.7,0.4), rotation=(np.deg2rad(-120),np.deg2rad(180),0))

bpy.ops.object.light_add(type='AREA', radius=1, location=(1,1,1))
bpy.ops.transform.resize(value=(0.001,1,1))
bpy.ops.object.light_add(type='AREA', radius=1, location=(0.9,1,1))
bpy.ops.transform.resize(value=(0.001,1,1))
bpy.ops.object.light_add(type='AREA', radius=1, location=(0,1,1))
bpy.ops.transform.resize(value=(0.001,1,1))
bpy.ops.object.light_add(type='AREA', radius=1, location=(-0.1,1,1))
bpy.ops.transform.resize(value=(0.001,1,1))
bpy.ops.object.light_add(type='AREA', radius=1, location=(-1,1,1))
bpy.ops.transform.resize(value=(0.001,1,1))
bpy.ops.object.light_add(type='AREA', radius=1, location=(-1.1,1,1))
bpy.ops.transform.resize(value=(0.001,1,1))
bpy.ops.object.light_add(type='AREA', radius=1, location=(-2,1,1))
bpy.ops.transform.resize(value=(0.001,1,1))
bpy.ops.object.light_add(type='AREA', radius=1, location=(-2.1,1,1))
bpy.ops.transform.resize(value=(0.001,1,1))
bpy.ops.object.light_add(type='AREA', radius=1, location=(1,-1,1))
bpy.ops.transform.resize(value=(0.001,1,1))
bpy.ops.object.light_add(type='AREA', radius=1, location=(0.9,-1,1))
bpy.ops.transform.resize(value=(0.001,1,1))
bpy.ops.object.light_add(type='AREA', radius=1, location=(0,-1,1))
bpy.ops.transform.resize(value=(0.001,1,1))
bpy.ops.object.light_add(type='AREA', radius=1, location=(-0.1,-1,1))
bpy.ops.transform.resize(value=(0.001,1,1))
bpy.ops.object.light_add(type='AREA', radius=1, location=(-1,-1,1))
bpy.ops.transform.resize(value=(0.001,1,1))
bpy.ops.object.light_add(type='AREA', radius=1, location=(-1.1,-1,1))
bpy.ops.transform.resize(value=(0.001,1,1))
bpy.ops.object.light_add(type='AREA', radius=1, location=(-2,-1,1))
bpy.ops.transform.resize(value=(0.001,1,1))
bpy.ops.object.light_add(type='AREA', radius=1, location=(-2.1,-1,1))
bpy.ops.transform.resize(value=(0.001,1,1))