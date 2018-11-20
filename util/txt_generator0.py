'''
txt file generator:
1) 1 ~ 28 types, every batch gets two images from each type
2) then, neighbor batchs get two images from different types
'''
import random
import os
import numpy as np
from numpy.random import choice
from itertools import combinations

# def rename_img(file_path):
#     for filename in os.listdir(file_path):
#         #os.chdir(file_path)
#         os.rename(filename, 'A'+filename[7]+filename[9:])
#
# file_path = "normalImg"
# rename_img(file_path)
class data_txt_writer():
    def __init__(self,image_path):
        np.random.seed(0)
        # random.seed(0)
        self.left = []
        self.right = []
        self.kind = dict()
        self.file = os.listdir(image_path)
        # self.file.remove('.DS_Store')
        # print(len(self.file))
        for ind, filename in enumerate(self.file):
            # print(ind, filename[7], filename[9])
            length = len(filename) # TODO
            if filename[length-5] == '1' or filename[length-5] == '2':
                xtype = filename[5:11] + '12'
                xtype = int(xtype)
                if xtype in self.kind:
                    self.kind[xtype].append(ind)
                else:
                    self.kind[xtype] = [ind]
            else:
                xtype = filename[5:11] + '34'
                xtype = int(xtype)
                if xtype in self.kind:
                    self.kind[xtype].append(ind)
                else:
                    self.kind[xtype] = [ind]

    def writer(self, txt_file):
        with open(txt_file,'w') as target:
            #genuine
            #for round in range(rounds):
            count = 0
            # print(len(self.kind))
            for k, v in self.kind.items():
                if count == 32:
                    #impostor
                    for _ in range(32):
                        i, j = random.sample(list(self.kind.keys()),2)
                        # print(i,j)
                        self.left.append(choice(self.kind[i]))
                        self.right.append(choice(self.kind[j]))
                    count = 0

                
                # for i in range(len(v)):
                #     for _ in range(3-i):
                #         self.right.append(v[i])
                #     self.left.append(v[i])
                # self.left.append(v[1])
                # self.left.append(v[2])
                # self.left.append(v[2])
                for i in range(len(v)):
                    self.right.append(v[i])
                    self.left.append(v[i])
                count += 1
                if len(v) == 2:
                    self.right.append(v[0])
                    self.left.append(v[1])
                    count += 2




            print(len(self.right), len(self.left))
            while len(self.right) != 0 and len(self.left) != 0:
                for _ in range(64):
                    name = self.file[self.left.pop(0)]
                    length = len(name)
                    if name[length-5] == '1' or name[length-5] == '2':
                        index = name[5:11] + '12'
                    else:
                        index = name[5:11] + '34'
                    target.write(name +' '+ index +'\n')
                    if len(self.left) == 0:
                        break
                for _ in range(64):
                    name = self.file[self.right.pop(0)]
                    length = len(name)
                    if name[length-5] == '3' or name[length-5] == '4':
                        index = name[5:11] + '34'
                    else:
                        index = name[5:11] + '12'
                    target.write(name +' '+ index +'\n')
                    if len(self.right) == 0:
                        break




image_path = '/home/nfs/xliu7/CycleGAN-tensorflow/test'
txt_path = 'pits_path_DB.txt'
object = data_txt_writer(image_path)
object.writer(txt_path)
