import json
import os

import numpy as np
import torch
from torch import nn


class DataPre(nn.Module):
    def __init__(self,root,split,class_choice=None):
        super().__init__()
        self.root=root
        self.catagory_file= os.path.join(self.root, 'synsetoffset2category.txt')
        self.catagory = {}
        self.meta= {}

        with open(self.catagory_file, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.catagory[ls[0]] = ls[1]
        #del self.catagory['Bag']
        #del self.catagory['Rocket']
        #del self.catagory['Skateboard']
        #del self.catagory['Motorbike']
        #del self.catagory[ 'Mug']
        del self.catagory['Pistol']
        del self.catagory['Laptop']
        del self.catagory['Knife']

        if not class_choice is None:
            self.catagory = {k: v for k, v in self.catagory.items() if k in class_choice}
            print('cat0:',self.catagory) #每个类型数据集对应的文件夹名称
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'),
                  'r') as f:  # 打乱顺序后的训练集文件名列表
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])  # 乱序训练集点云文件名
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'),
                  'r') as f:  # 打乱顺序后的测试集文件名列表
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])

        for item in self.catagory:#对于cat中的每一类点云类型项
            print('category', item)
            self.meta[item] = []#初始化集合meta中的每一个元素为空列表
            dir_point = os.path.join(self.root, self.catagory[item], 'points')#数据集的点云文件路径
            dir_seg = os.path.join(self.root, self.catagory[item], 'points_label')#数据集标签文件路径，标签中数值代表颜色蓝绿红
            #print(dir_point, dir_seg)
            fns = sorted(os.listdir(dir_point)) #os.listdir：列出路径下所有文件名
            #print(len(fns))

            #排除掉乱序文件中不存在的点云文件
            if split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
                #print((fns))
            if split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg'),self.catagory[item], token))
        self.datapath = []
        for item in self.catagory:
            for fn in self.meta[item]:
                self.datapath.append(fn[0])#(item, fn[0], fn[1], fn[2], fn[3])datapath存储cat中每个类型的类型名称，某个点云文件路径，同名同路径的标签文件路径，所在点云类型文件夹名，点云文件名
    def forward(self,):
        return self.datapath


if __name__ == '__main__':
    dset = DataPre(root='data/shapenetcore_partanno_segmentation_benchmark_v0/', split='train',class_choice=None)
    datapath=dset()
    print(datapath)


