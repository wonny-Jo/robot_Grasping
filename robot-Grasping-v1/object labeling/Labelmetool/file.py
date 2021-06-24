import os

path = 'C:/Users/incorl/Desktop/RL_Project/objects'

for i in range(8,11):
    for j in range(10):
        if not os.path.exists(path + '/%d/%d' % (i,j)):
            os.makedirs(path + '/%d/%d' % (i,j))