# from __future__ import print_function
import csv
import random
from data_utils.curve_mask import unit_sampling_transform, plot_xyz
import torch.utils.data as data
import os
import os.path
import torch
from data_utils.points_sampling import sample_keypoints
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
class Dataset_surface_axis(data.Dataset):
    def __init__(self, trainset_config,dataset,split,transform=True):
        self.pnum = trainset_config['npoints']
        self.pnum_keypoint = trainset_config['num_keypoints']
        self.root = trainset_config['data_dir']
        self.transform = transform
        self.keypoint_cls = trainset_config['keypoint_cls']
        self.datapath_TubePoint = []
        self.datapath_TubePoint_axis = []
        self.datapath_SectionPoint = []
        self.dataset=dataset
        self.split=split
        if self.split == 'train':
            self.dir_TubePoint = os.path.join(self.root, self.dataset, self.dataset + '-data-train')  # 数据集的管段点云文件路径
            self.dir_TubePoint_axis = os.path.join(self.root, self.dataset,
                                                   self.dataset + '-data-axis-train')  # 数据集的管段点云文件路径
            self.fns_TubePoint = sorted(os.listdir(self.dir_TubePoint))  # os.listdir：列出路径下所有文件名\
            self.dir_para = os.path.join(self.root, self.dataset, self.dataset + '-para.csv')
        elif self.split == 'test':
            self.dir_TubePoint = os.path.join(self.root, self.dataset, self.dataset + '-data-test')  # 数据集的管段点云文件路径
            self.dir_TubePoint_axis = os.path.join(self.root, self.dataset,
                                                   self.dataset + '-data-axis-test')  # 数据集的管段点云文件路径
            self.fns_TubePoint = sorted(os.listdir(self.dir_TubePoint))  # os.listdir：列出路径下所有文件名\
            self.dir_para = os.path.join(self.root, self.dataset, self.dataset + '-para.csv')
        for fn_TubePoint in self.fns_TubePoint:
            self.datapath_TubePoint.append(os.path.join(self.dir_TubePoint, fn_TubePoint))  # 所有样本文件的绝对路径

    def __getitem__(self, index):
        fn_TubePoint = self.datapath_TubePoint[index]  #某个管段点云文件的绝对路径
        fn_TubePoint_axis = self.dir_TubePoint_axis + "/data-" + self.dataset + "-axis-" + fn_TubePoint.split('-')[-1]
        fr_tube = open(fn_TubePoint, 'r')  # csv文件名
        reader = csv.reader(fr_tube)
        data_Ori = list(reader)[1:]
        data_Ori= np.array(data_Ori, dtype=np.float32)
        point_Tube_Ori = data_Ori[:,4:7]
        if self.keypoint_cls=='surface':
            point_Tube_keypoint_Ori = point_Tube_Ori
        if self.keypoint_cls == 'axis':
            fr_axis = open(fn_TubePoint_axis, 'r')  # csv文件名
            reader = csv.reader(fr_axis)
            point_Tube_keypoint_Ori = list(reader)[1:]
            point_Tube_keypoint_Ori = np.array(point_Tube_keypoint_Ori, dtype=np.float32)

        if self.transform:
            point_Tube_cat=np.concatenate((point_Tube_Ori,point_Tube_keypoint_Ori),axis=0)
            point_Tube_cat = self.pc_transform(point_Tube_cat,normalization=True)
            point_Tube_Ori = point_Tube_cat[:len(point_Tube_Ori)]
            point_Tube_keypoint_Ori = point_Tube_cat[len(point_Tube_Ori):]
        #normal
        '''
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array(point_Tube_Ori))
        point_cloud.estimate_normals()
        normals = np.asarray(point_cloud.normals,dtype=np.float32)
        '''
        #FPS sampling
        point_Tube, idx_point_Tube = sample_keypoints(point_Tube_Ori, K=self.pnum)
        #normals=masked_gather(torch.from_numpy(normals.reshape(1,normals.shape[0],normals.shape[1])),torch.from_numpy(idx_point_Tube.reshape(1,2048)))[0]
        normals=point_Tube
        point_Tube_keypoint, _ = sample_keypoints(point_Tube_keypoint_Ori, K=self.pnum_keypoint)
        #point_Tube_keypoint=point_Tube_keypoint_Ori[:16.:]
        data = {'points': point_Tube, 'normals': normals, 'label': 1, 'category': '1', 'category_name': 'planar','keypoints':point_Tube_keypoint}
        return data

    def __len__(self):
        return len(self.datapath_TubePoint)

    def pc_transform(self, pc,normalization=True):  # 归一化
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        if normalization:
            m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
            pc = pc / m
        return pc

class Dataset_for_para_optim(data.Dataset):
    def __init__(self, trainset_config,dataset,split,transform=True,para_transform=True):
        self.pnum = trainset_config['npoints']
        self.pnum_keypoint =  trainset_config['num_keypoints']
        self.root = trainset_config['data_dir']
        self.transform = transform
        self.keypoint_cls = trainset_config['keypoint_cls']
        self.datapath_TubePoint = []
        self.datapath_TubePoint_axis = []
        self.datapath_SectionPoint = []
        self.split =split
        self.dataset = dataset
        self.para_transform=para_transform
        self.para_min_max=np.array(trainset_config['para_min_max_'+dataset],dtype=np.float32)
        if self.split == 'train':
            self.dir_TubePoint = os.path.join(self.root, self.dataset,self.dataset+'-data-train')  # 数据集的管段点云文件路径
            self.dir_TubePoint_axis = os.path.join(self.root, self.dataset,self.dataset+'-data-axis-train')  # 数据集的管段点云文件路径
            self.fns_TubePoint = sorted(os.listdir(self.dir_TubePoint))  # os.listdir：列出路径下所有文件名\
            self.dir_para = os.path.join(self.root,self.dataset,self.dataset+'-para.csv')
        elif self.split == 'test':
            self.dir_TubePoint = os.path.join(self.root, self.dataset,self.dataset+'-data-test')  # 数据集的管段点云文件路径
            self.dir_TubePoint_axis = os.path.join(self.root, self.dataset,self.dataset+'-data-axis-test')  # 数据集的管段点云文件路径
            self.fns_TubePoint = sorted(os.listdir(self.dir_TubePoint))  # os.listdir：列出路径下所有文件名\
            self.dir_para = os.path.join(self.root, self.dataset,self.dataset+'-para.csv')
        for fn_TubePoint in self.fns_TubePoint:
            self.datapath_TubePoint.append(os.path.join(self.dir_TubePoint, fn_TubePoint))  # 所有样本文件的绝对路径


    def __getitem__(self, index):
        if self.split == 'train' or 'test':
        #data of measured shape
            fn_TubePoint_m = self.datapath_TubePoint[index]  #某个管段点云文件的绝对路径
            data_num=int(os.path.splitext(fn_TubePoint_m)[0].split('-')[-1])
            fn_TubePoint_axis_m=self.dir_TubePoint_axis + "/data-"+self.dataset+"-axis-" + fn_TubePoint_m.split('-')[-1]
            fr_tube_m = open(fn_TubePoint_m, 'r')  # csv文件名
            reader_m = csv.reader(fr_tube_m)
            data_Ori_m = list(reader_m)[1:]
            data_Ori_m= np.array(data_Ori_m, dtype=np.float32)
            point_Tube_Ori_m = data_Ori_m[:,4:7]
            length= np.max(data_Ori_m[:,3])-np.min(data_Ori_m[:,3])
            fr_para_m = open(self.dir_para, 'r')  # csv文件名
            reader_m = csv.reader(fr_para_m)
            paras_m = list(reader_m)[data_num+1]
            para_m = np.array(paras_m[-7:], dtype=np.float32)
            para_shape_m = np.array(paras_m[:-7], dtype=np.float32)/1000.0
            stress_m = data_Ori_m[:, 7:13]
            stress_m=self.pc_transform(stress_m,normalization=True)
            if self.keypoint_cls=='surface':
                point_Tube_keypoint_Ori_m = point_Tube_Ori_m
            if self.keypoint_cls=='axis':
                fr_axis_m = open(fn_TubePoint_axis_m, 'r')  # csv文件名
                reader_m = csv.reader(fr_axis_m)
                point_Tube_keypoint_Ori_m = list(reader_m)[1:]
                point_Tube_keypoint_Ori_m = np.array(point_Tube_keypoint_Ori_m, dtype=np.float32)

            if self.transform:
                point_Tube_cat_m=np.concatenate((point_Tube_Ori_m,point_Tube_keypoint_Ori_m),axis=0)
                point_Tube_cat_m = self.pc_transform(point_Tube_cat_m,normalization=True)
                point_Tube_Ori_m = point_Tube_cat_m[:len(point_Tube_Ori_m)]
                point_Tube_keypoint_Ori_m = point_Tube_cat_m[len(point_Tube_Ori_m):]
            #normal
            '''
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(np.array(point_Tube_Ori))
            point_cloud.estimate_normals()
            normals = np.asarray(point_cloud.normals,dtype=np.float32)
            '''
            #FPS sampling
            point_Tube_m, idx_point_Tube_m = sample_keypoints(point_Tube_Ori_m, K=self.pnum)
            #normals=masked_gather(torch.from_numpy(normals.reshape(1,normals.shape[0],normals.shape[1])),torch.from_numpy(idx_point_Tube.reshape(1,2048)))[0]
            stress_m=stress_m[idx_point_Tube_m]
            normals_m=point_Tube_m
            point_Tube_keypoint_m, _ = sample_keypoints(point_Tube_keypoint_Ori_m, K=self.pnum_keypoint)
        # data of designed shape (gt)
            if self.split == 'test':
                index =len(self.datapath_TubePoint)-index-1
            else:
                index = np.random.choice(len(self.datapath_TubePoint), replace=False)
            fn_TubePoint_d = self.datapath_TubePoint[index]  # 某个管段点云文件的绝对路径
            data_num = int(os.path.splitext(fn_TubePoint_d)[0].split('-')[-1])
            fn_TubePoint_axis_d = self.dir_TubePoint_axis + "/data-"+self.dataset+"-axis-" + fn_TubePoint_d.split('-')[-1]
            fr_tube_d = open(fn_TubePoint_d, 'r')  # csv文件名
            reader_d = csv.reader(fr_tube_d)
            data_Ori_d = list(reader_d)[1:]
            data_Ori_d = np.array(data_Ori_d, dtype=np.float32)
            point_Tube_Ori_d = data_Ori_d[:, 4:7]
            stress_d = data_Ori_d[:, 7:13]
            stress_d = self.pc_transform(stress_d, normalization=True)
            fr_para_d = open(self.dir_para, 'r')  # csv文件名
            reader_d = csv.reader(fr_para_d)
            paras_d = list(reader_d)[data_num + 1]
            para_d = np.array(paras_d[-7:], dtype=np.float32)
            para_shape_d = np.array(paras_d[:-7], dtype=np.float32) / 1000.0
            if self.keypoint_cls == 'surface':
                point_Tube_keypoint_Ori_d = point_Tube_Ori_d
            if self.keypoint_cls == 'axis':
                fr_axis_d = open(fn_TubePoint_axis_d, 'r')  # csv文件名
                reader_d = csv.reader(fr_axis_d)
                point_Tube_keypoint_Ori_d = list(reader_d)[1:]
                point_Tube_keypoint_Ori_d = np.array(point_Tube_keypoint_Ori_d, dtype=np.float32)

            if self.transform:
                point_Tube_cat_d = np.concatenate((point_Tube_Ori_d, point_Tube_keypoint_Ori_d), axis=0)
                point_Tube_cat_d = self.pc_transform(point_Tube_cat_d, normalization=True)
                point_Tube_Ori_d = point_Tube_cat_d[:len(point_Tube_Ori_d)]
                point_Tube_keypoint_Ori_d = point_Tube_cat_d[len(point_Tube_Ori_d):]
            # normal
            '''
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(np.array(point_Tube_Ori))
            point_cloud.estimate_normals()
            normals = np.asarray(point_cloud.normals,dtype=np.float32)
            '''
            # FPS sampling
            point_Tube_d, idx_point_Tube_d = sample_keypoints(point_Tube_Ori_d, K=self.pnum)
            stress_d = stress_d[idx_point_Tube_d]
            normals_d = point_Tube_d
            point_Tube_keypoint_d, _ = sample_keypoints(point_Tube_keypoint_Ori_d, K=self.pnum_keypoint)
            if  self.para_transform:
                para_d=(para_d-self.para_min_max[:,0])/self.para_min_max[:,1]
                para_m = (para_m-self.para_min_max[:,0] )/ self.para_min_max[:,1]
        data = {'points_m': point_Tube_m,'stress_m': stress_m,'points_d': point_Tube_d,'stress_d': stress_d,'normals_m': normals_m,'normals_d': normals_d,'para_m':para_m,
                'para_d':para_d,'label': 1,'category': '1', 'category_name': 'planar','keypoints_m':point_Tube_keypoint_m,'keypoints_d':point_Tube_keypoint_d,
                'para_shape_m':para_shape_m,'para_shape_d':para_shape_d,'length':length}
        return data

    def __len__(self):
        return len(self.datapath_TubePoint)

    def pc_transform(self, pc,normalization=True):  # 归一化
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        if normalization:
            m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
            pc = pc / m
        return pc
class Dataset_for_para_optim_exp(data.Dataset):
    def __init__(self, trainset_config,dataset,split,transform=True):
        self.pnum = trainset_config['npoints']
        self.pnum_keypoint = trainset_config['num_keypoints']
        self.root =trainset_config['data_dir']
        self.transform = transform
        self.keypoint_cls = trainset_config['keypoint_cls']
        self.datapath_TubePoint = []
        self.datapath_TubePoint_axis = []
        self.datapath_SectionPoint = []
        self.split =split
        self.dataset=dataset
        self.para_min_max=np.array(trainset_config['para_min_max_'+dataset],dtype=np.float32)
        if self.split == 'train':
            self.dir_TubePoint = os.path.join(self.root,dataset,dataset+'-data-train')  # 数据集的管段点云文件路径
            self.dir_TubePoint_axis = os.path.join(self.root, dataset,dataset+'-data-axis-train')  # 数据集的管段点云文件路径
            self.fns_TubePoint = sorted(os.listdir(self.dir_TubePoint))  # os.listdir：列出路径下所有文件名\
            self.dir_para = os.path.join(self.root, dataset,dataset+'-para.csv')
        elif self.split == 'test':
            self.dir_TubePoint = os.path.join(self.root, dataset,dataset+'-data-test')  # 数据集的管段点云文件路径
            self.dir_TubePoint_axis = os.path.join(self.root, dataset,dataset+'-data-axis-test')  # 数据集的管段点云文件路径
            self.fns_TubePoint = sorted(os.listdir(self.dir_TubePoint))  # os.listdir：列出路径下所有文件名\
            self.dir_para = os.path.join(self.root, dataset,dataset+'-para.csv')
        for fn_TubePoint in self.fns_TubePoint:
            self.datapath_TubePoint.append(os.path.join(self.dir_TubePoint, fn_TubePoint))  # 所有样本文件的绝对路径
        if dataset=='exp-6FB-2D':
            self.index_m = 1
            self.fn_TubePoint_m='/home/robot/Wl/DDPM/RO-main/data/dataset-6FB-tube-diffusion-D25-1024/6FB-2D/6FB-2D-data-train/data-6FB-2D-1.csv'
            self.dir_para_m = os.path.join(self.root, '6FB-2D', '6FB-2D-para.csv')
        else:
            self.index_m = 3
            self.fn_TubePoint_m='/home/robot/Wl/DDPM/RO-main/data/dataset-6FB-tube-diffusion-D25-1024/6FB-3D/6FB-3D-data-train/data-6FB-3D-3.csv'
            self.dir_para_m = os.path.join(self.root, '6FB-3D', '6FB-3D-para.csv')
    def __getitem__(self, index):
        if self.split == 'train' or 'test':
        #data of measured shape
            index_m=self.index_m
            fn_TubePoint_m = self.fn_TubePoint_m  #某个管段点云文件的绝对路径
            fr_tube_m = open(fn_TubePoint_m, 'r')  # csv文件名
            reader_m = csv.reader(fr_tube_m)
            data_Ori_m = list(reader_m)[1:]
            data_Ori_m= np.array(data_Ori_m, dtype=np.float32)
            point_Tube_Ori_m = data_Ori_m[:,4:7]
            fr_para_m = open(self.dir_para_m, 'r')  # csv文件名
            reader_m = csv.reader(fr_para_m)
            para_m = list(reader_m)[index_m+1]
            para_m = np.array(para_m[-7:], dtype=np.float32)
            stress_m = data_Ori_m[:, 7:13]
            stress_m=self.pc_transform(stress_m,normalization=True)
            point_Tube_keypoint_Ori_m = point_Tube_Ori_m

            if self.transform:
                point_Tube_cat_m=np.concatenate((point_Tube_Ori_m,point_Tube_keypoint_Ori_m),axis=0)
                point_Tube_cat_m = self.pc_transform(point_Tube_cat_m,normalization=True)
                point_Tube_Ori_m = point_Tube_cat_m[:len(point_Tube_Ori_m)]
                point_Tube_keypoint_Ori_m = point_Tube_cat_m[len(point_Tube_Ori_m):]
            #normal
            '''
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(np.array(point_Tube_Ori))
            point_cloud.estimate_normals()
            normals = np.asarray(point_cloud.normals,dtype=np.float32)
            '''
            #sampling
            point_Tube_m, idx_point_Tube_m = sample_keypoints(point_Tube_Ori_m, K=self.pnum)
            # normals=masked_gather(torch.from_numpy(normals.reshape(1,normals.shape[0],normals.shape[1])),torch.from_numpy(idx_point_Tube.reshape(1,2048)))[0]
            stress_m = stress_m[idx_point_Tube_m]
            normals_m = point_Tube_m
            point_Tube_keypoint_m, _ = sample_keypoints(point_Tube_keypoint_Ori_m, K=self.pnum_keypoint)
        # data of designed shape (gt)
            fn_TubePoint_d = self.datapath_TubePoint[index]  # 某个管段点云文件的绝对路径
            data_Ori_d = np.loadtxt(fn_TubePoint_d)
            data_Ori_d = np.array(data_Ori_d, dtype=np.float32)
            point_Tube_Ori_d = data_Ori_d
            fr_para_d = open(self.dir_para, 'r')  # csv文件名
            reader_d = csv.reader(fr_para_d)
            para_d = list(reader_d)[index + 1]
            para_d = np.array(para_d[-7:], dtype=np.float32)
            point_Tube_keypoint_Ori_d = point_Tube_Ori_d
            if self.transform:
                point_Tube_cat_d = np.concatenate((point_Tube_Ori_d, point_Tube_keypoint_Ori_d), axis=0)
                point_Tube_cat_d = self.pc_transform(point_Tube_cat_d, normalization=True)
                point_Tube_Ori_d = point_Tube_cat_d[:len(point_Tube_Ori_d)]
                point_Tube_keypoint_Ori_d = point_Tube_cat_d[len(point_Tube_Ori_d):]
            # FPS sampling
            point_Tube_d, idx_point_Tube_d = sample_keypoints(point_Tube_Ori_d, K=self.pnum)
            normals_d = point_Tube_d
            point_Tube_keypoint_d, _ = sample_keypoints(point_Tube_keypoint_Ori_d, K=self.pnum_keypoint)
            para_d = (para_d - self.para_min_max[:, 0]) / self.para_min_max[:, 1]
            para_m = (para_m - self.para_min_max[:, 0]) / self.para_min_max[:, 1]
        data = {'points_m': point_Tube_m,'stress_m': stress_m,'points_d': point_Tube_d,'normals_m': normals_m,'normals_d': normals_d,'para_m':para_m,
                'para_d':para_d,'label': 1,'category': '1', 'category_name': 'planar','keypoints_m':point_Tube_keypoint_m,'keypoints_d':point_Tube_keypoint_d}
        return data

    def __len__(self):
        return len(self.datapath_TubePoint)

    def pc_transform(self, pc,normalization=True):  # 归一化
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        if normalization:
            m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
            pc = pc / m
        return pc

class Dataset_for_shape_exp(data.Dataset):
    def __init__(self, trainset_config,dataset,split,transform=True):
        self.pnum = trainset_config['npoints']
        self.pnum_keypoint = trainset_config['num_keypoints']
        self.root =trainset_config['data_dir']
        self.transform = transform
        self.keypoint_cls = trainset_config['keypoint_cls']
        self.datapath_TubePoint = []
        self.datapath_TubePoint_axis = []
        self.datapath_SectionPoint = []
        self.split =split
        self.dataset=dataset
        self.para_min_max=np.array(trainset_config['para_min_max_'+dataset],dtype=np.float32)
        if self.split == 'train':
            self.dir_TubePoint = os.path.join(self.root,dataset,dataset+'-data-train')  # 数据集的管段点云文件路径
            self.dir_TubePoint_axis = os.path.join(self.root, dataset,dataset+'-data-axis-train')  # 数据集的管段点云文件路径
            self.fns_TubePoint = sorted(os.listdir(self.dir_TubePoint))  # os.listdir：列出路径下所有文件名\
            self.dir_para = os.path.join(self.root, dataset,dataset+'-para.csv')
        elif self.split == 'test':
            self.dir_TubePoint = os.path.join(self.root, dataset,dataset+'-data-test')  # 数据集的管段点云文件路径
            self.dir_TubePoint_axis = os.path.join(self.root, dataset,dataset+'-data-axis-test')  # 数据集的管段点云文件路径
            self.fns_TubePoint = sorted(os.listdir(self.dir_TubePoint))  # os.listdir：列出路径下所有文件名\
            self.dir_para = os.path.join(self.root, dataset,dataset+'-para.csv')
        for fn_TubePoint in self.fns_TubePoint:
            self.datapath_TubePoint.append(os.path.join(self.dir_TubePoint, fn_TubePoint))  # 所有样本文件的绝对路径
    def __getitem__(self, index):
        if self.split == 'train' or 'test':
            fn_TubePoint = self.datapath_TubePoint[index]
            data_num = index
            fn_TubePoint_axis = self.dir_TubePoint_axis + "/axis-" + str(index)+ ".txt"
            data_Ori = np.loadtxt(fn_TubePoint)
            data_Ori = np.array(data_Ori, dtype=np.float32)
            point_Tube_Ori = data_Ori
            fr_para = open(self.dir_para, 'r')  # csv文件名
            reader = csv.reader(fr_para)
            para = list(reader)[data_num+1]
            para = np.array(para[-7:], dtype=np.float32)
            point_Tube_keypoint_Ori = np.loadtxt(fn_TubePoint_axis)
            point_Tube_keypoint_Ori = np.array(point_Tube_keypoint_Ori, dtype=np.float32)

            point_Tube_cat=np.concatenate((point_Tube_Ori,point_Tube_keypoint_Ori),axis=0)
            point_Tube_cat = self.pc_transform(point_Tube_cat,normalization=True)
            point_Tube_Ori = point_Tube_cat[:len(point_Tube_Ori)]
            point_Tube_keypoint = point_Tube_cat[len(point_Tube_Ori):]
            #normal
            '''
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(np.array(point_Tube_Ori))
            point_cloud.estimate_normals()
            normals = np.asarray(point_cloud.normals,dtype=np.float32)
            '''
            #sampling
            point_Tube, idx_point_Tube = sample_keypoints(point_Tube_Ori, K=self.pnum)
            # normals=masked_gather(torch.from_numpy(normals.reshape(1,normals.shape[0],normals.shape[1])),torch.from_numpy(idx_point_Tube.reshape(1,2048)))[0]
            normals = point_Tube
            para = (para - self.para_min_max[:, 0]) / self.para_min_max[:, 1]
        data = {'points': point_Tube,'normals': normals,'para':para,'label': 1,'category': '1', 'category_name': 'planar','keypoints':point_Tube_keypoint}

        return data

    def __len__(self):
        return len(self.datapath_TubePoint)

    def pc_transform(self, pc,normalization=True):  # 归一化
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        if normalization:
            m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
            pc = pc / m
        return pc

class Dataset_for_shape_spiral(data.Dataset):
    def __init__(self, trainset_config,dataset,split,transform=True):
        self.pnum = trainset_config['npoints']
        self.pnum_keypoint = trainset_config['num_keypoints']
        self.root = trainset_config['data_dir']
        self.transform = transform
        self.keypoint_cls = trainset_config['keypoint_cls']
        self.datapath_TubePoint = []
        self.datapath_TubePoint_axis = []
        self.datapath_SectionPoint = []
        self.split = split
        self.dataset = dataset
        #self.para_min_max = np.array(trainset_config['para_min_max_' + dataset], dtype=np.float32)
        if split == 'train':
            dir_TubePoint = os.path.join(self.root, 'train_spiral_TubePoint')
            fns_TubePoint = sorted(os.listdir(dir_TubePoint))
            self.dir_TubePoint_axis = os.path.join(self.root, 'train_spiral_skeleton')
        elif split == 'test':
            dir_TubePoint = os.path.join(self.root, 'test_spiral_TubePoint')
            fns_TubePoint = sorted(os.listdir(dir_TubePoint))
            self.dir_TubePoint_axis = os.path.join(self.root, 'test_spiral_skeleton')
        self.dir_para=os.path.join(self.root,dataset+'-para.csv')
        for fn_TubePoint in fns_TubePoint:
            self.datapath_TubePoint.append(os.path.join(dir_TubePoint, fn_TubePoint))

    def __getitem__(self, index):
        fn_TubePoint = self.datapath_TubePoint[index]
        point_Tube_Ori = np.loadtxt(fn_TubePoint).astype(np.float32)
        point_Tube_Ori = np.array(point_Tube_Ori)
        choice = np.random.choice(len(point_Tube_Ori), self.pnum, replace=True)
        point_Tube = point_Tube_Ori[choice]
        keypoints_file=self.dir_TubePoint_axis + "/skeleton_spiral" + os.path.basename(fn_TubePoint).split('_')[-1]
        keypoints = np.loadtxt(keypoints_file)
        keypoints = np.array(keypoints, dtype=np.float32)
        #choice = np.random.choice(len(keypoints), self.pnum_keypoint, replace=True)
        keypoints = keypoints[:16,:]
        if self.transform:
            point_Tube_cat = np.concatenate((point_Tube, keypoints), axis=0)
            point_Tube_cat = self.pc_transform(point_Tube_cat, normalization=True)
            point_Tube = point_Tube_cat[:len(point_Tube)]
            keypoints = point_Tube_cat[len(point_Tube):]
        normals=point_Tube
        data = {'points': point_Tube,'normals':  normals,'label': 3, 'category': '3', 'category_name': 'spiral','keypoints':keypoints}
        return data
    def __len__(self):
        return len(self.datapath_TubePoint)

    def pc_transform(self, pc, normalization=True):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        if normalization:
            m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
            pc = pc / m
        return pc

class Dataset_for_shape_complex(data.Dataset):
    def __init__(self, trainset_config,dataset,split,transform=True):
        self.pnum = trainset_config['npoints']
        self.pnum_keypoint = trainset_config['num_keypoints']
        self.root = trainset_config['data_dir']
        self.transform = transform
        self.keypoint_cls = trainset_config['keypoint_cls']
        self.datapath_TubePoint = []
        self.datapath_TubePoint_axis = []
        self.datapath_SectionPoint = []
        self.split = split
        self.dataset = dataset
        #self.para_min_max = np.array(trainset_config['para_min_max_' + dataset], dtype=np.float32)
        assert split == 'test'
        dir_TubePoint = os.path.join(self.root, 'complex-6FB-3D/data-exp-test')
        fns_TubePoint = sorted(os.listdir(dir_TubePoint))
        self.dir_TubePoint_axis = os.path.join(self.root, 'complex-6FB-3D/data-exp-axis-test')
        for fn_TubePoint in fns_TubePoint:
            self.datapath_TubePoint.append(os.path.join(dir_TubePoint, fn_TubePoint))

    def __getitem__(self, index):
        fn_TubePoint = self.datapath_TubePoint[index]
        point_Tube_Ori = np.loadtxt(fn_TubePoint).astype(np.float32)
        point_Tube_Ori = np.array(point_Tube_Ori)
        choice = np.random.choice(len(point_Tube_Ori), self.pnum, replace=True)
        point_Tube = point_Tube_Ori[choice]
        keypoints_file=self.dir_TubePoint_axis + "/axis_" + os.path.basename(fn_TubePoint).split('_')[-1]
        keypoints = np.loadtxt(keypoints_file)
        keypoints = np.array(keypoints, dtype=np.float32)
        keypoints, _ = sample_keypoints(keypoints, K=512)
        #choice = np.random.choice(len(keypoints), 160, replace=True)
        #keypoints = keypoints[choice]
        '''
        synthetic num +10  NO sampling
        exp sample 144+2,176+5,3-256+5
        
        '''
        keypoints,T=unit_sampling_transform(keypoints,center_npoint=np.ceil(keypoints.shape[0]/self.pnum_keypoint).astype(int)+10,k=self.pnum_keypoint)
        keypoints=np.array(keypoints,dtype=np.float32)
        T = np.array(T, dtype=np.float32)
        if self.transform:
            keypoints,m,centroid = self.pc_transform(keypoints, normalization=True)
            point_Tube = self.pc_transform_tubepoint(point_Tube, normalization=True)
        #plot_xyz(keypoints)
        normals=point_Tube
        data = {'points': point_Tube,'normals':  normals,'label': 3, 'category': '3', 'category_name': 'spiral','keypoints':keypoints,'transform_matrix':T,'scale_m':m,'centroid':centroid}
        return data
    def __len__(self):
        return len(self.datapath_TubePoint)

    def pc_transform(self, pc, normalization=True):

        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=1)
        centroid=np.expand_dims(centroid, axis=1)
        pc = pc - centroid
        if normalization:
            m = np.max(np.sqrt(np.sum(pc ** 2, axis=2)))
            pc = pc / m
        return pc,m,centroid

    def pc_transform_tubepoint(self, pc, normalization=True):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        if normalization:
            m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
            pc = pc / m
        return pc

class Dataset_shapenet(data.Dataset):
    def __init__(self, trainset_config, dataset, split, transform=True,class_name=None):
        self.pnum = trainset_config['npoints']
        self.pnum_keypoint = trainset_config['num_keypoints']
        self.root = trainset_config['data_dir']
        self.transform = transform
        self.keypoint_cls = trainset_config['keypoint_cls']
        self.datapath_TubePoint = []
        self.datapath_TubePoint_axis = []
        self.datapath_SectionPoint = []
        self.split = split
        self.dataset = dataset
        self.DataPre = DataPre(self.root, split, class_choice=class_name)
        self.datapath_TubePoint = self.DataPre()

    def __getitem__(self, index):
        fn_TubePoint = self.datapath_TubePoint[index]  # 某个管段点云文件的绝对路径
        point_Tube_Ori = np.loadtxt(fn_TubePoint).astype(np.float32)  # np.loadtxt:读取txt文件
        point_Tube_Ori = np.array(point_Tube_Ori)
        if self.transform:
            point_Tube_Ori = self.pc_transform(point_Tube_Ori)
        if len(point_Tube_Ori)<1024:
            zeros=np.zeros((1024-point_Tube_Ori.shape[0],3),dtype=point_Tube_Ori.dtype)
            point_Tube_Ori=np.vstack((point_Tube_Ori,zeros))
            #print(fn_TubePoint)
        point_Tube_Ori = torch.from_numpy(point_Tube_Ori)
        #point_Tube_Ori = torch.unsqueeze(point_Tube_Ori, 0)
        #point_Tube_idx = furthest_point_sample(point_Tube_Ori, self.pnum)
        #point_Tube = index_points(point_Tube_Ori, point_Tube_idx)
        #point_Tube = torch.squeeze(point_Tube, 0)
        point_Tube_keypoint_Ori=point_Tube_Ori
        #FPS sampling

        choice = np.random.choice(len(point_Tube_Ori), self.pnum, replace=False)
        point_Tube = point_Tube_Ori[choice]
        #normals=masked_gather(torch.from_numpy(normals.reshape(1,normals.shape[0],normals.shape[1])),torch.from_numpy(idx_point_Tube.reshape(1,2048)))[0]
        normals=point_Tube
        choice = np.random.choice(len(point_Tube_keypoint_Ori), self.pnum_keypoint, replace=False)
        point_Tube_keypoint = point_Tube_keypoint_Ori[choice]
        data = {'points': point_Tube, 'normals': normals, 'label': 1, 'category': '1', 'category_name': 'planar','keypoints':point_Tube_keypoint}
        return data

    def __len__(self):
        return len(self.datapath_TubePoint)

    def pc_transform(self, pc,normalization=True):  # 归一化
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        if normalization:
            m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
            pc = pc / m
        return pc


if __name__ == '__main__':
    dset_train = Dataset_for_para_optim(root=r'data/dataset-6FB-tube-diffusion-D10-2048/6FB-2D', pnum=2048,split='train',pnum_keypoint=16,keypoint_cls='axis',transform=True)
    trainloader = torch.utils.data.DataLoader(dset_train, batch_size=4, shuffle=False, num_workers=0)
    for data in trainloader:
        print(data['points_m'].shape,data['keypoints_m'].shape,data['stress_m'].shape,data['keypoints_m'].shape)
        print(data['points_d'].shape, data['keypoints_d'].shape, data['para_d'].shape)

