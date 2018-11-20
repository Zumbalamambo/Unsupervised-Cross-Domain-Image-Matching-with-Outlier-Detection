'''
test txt generator
'''
'''
txt file generator:
1) 1 ~ 28 types, every batch gets two images from each type
2) then, neighbor batchs get two images from different types
'''
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

        self.file = os.listdir(image_path)
        self.file.remove('.DS_Store')
    def writer(self, txt_file):
        with open(txt_file,'w') as target:
            # for _ in range(9):
            # count = 0
            # count = [0]*31
            for item in self.file:
                # if count == 280: break
                # xtype = item[0:3] + item[8:-4]
                if item[17] == '2' and int(item[5:11]) < 401:
                    xtype = item[5:11]+ 'out'
                # if count[int(xtype)] > 9:
                #     continue
                
                # count[int(xtype)] += 1
                # length = len(item)
                # if item[-5] == '1' or item[-5] == '2':
                #     xtype = item[5:11] + 'A'
                # else:
                #     xtype = item[5:11] + 'B'

                # xtype = item[0:6]
                    target.write(item + ' ' + xtype + '\n')
                # count += 1

image_path = '/Users/Lkid/MSc_Thesis/siameseNet/outlierVersion/cycle_GAN_generated/'
txt_path = 'targetPathPitts.txt'
object = data_txt_writer(image_path)
object.writer(txt_path)
