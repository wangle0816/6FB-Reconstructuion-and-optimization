import time

import numpy as np
import torch
from thop import profile
from torch import nn
from AFNO_3D.AFNO_3D import AFNO3D
import torch.nn.functional as F

from AFNO_3D.physically_driven_stress_increment_solver_module import stress_increment_solver,elastic_plastic_matrix_compute
from AFNO_3D.learning_based_stress_increment_solver_module import stress_increment_solver as stress_increment_solver_mlp


class Linear_ResBlock_process(nn.Module):
    def __init__(self, input_size=1024, output_size=256):
        super(Linear_ResBlock_process, self).__init__()
        self.conv1 = nn.Linear(input_size, input_size)
        self.conv2 = nn.Linear(input_size, output_size)
        self.conv_res = nn.Linear(input_size, output_size)
        self.af = nn.ReLU(inplace=True)

    def forward(self, feature):
        return self.conv2(self.af(self.conv1(self.af(feature)))) + self.conv_res(feature)

class Linear_ResBlock_point(nn.Module):
    def __init__(self, input_size=1024, output_size=256):
        super(Linear_ResBlock_point,self).__init__()
        self.conv1 = nn.Linear(input_size, input_size)
        self.conv2 = nn.Linear(input_size, output_size)
        self.conv_res = nn.Linear(input_size, output_size)
        self.af = nn.ReLU(inplace=True)
    def forward(self, feature):
        return self.conv2(self.af(self.conv1(self.af(feature)))) + self.conv_res(feature)

class Linear_ResBlock_para(nn.Module):
    def __init__(self, input_size=1024, output_size=256):
        super(Linear_ResBlock_para,self).__init__()
        self.conv1 = nn.Linear(input_size, input_size)
        self.conv2 = nn.Linear(input_size, output_size)
        self.conv_res = nn.Linear(input_size, output_size)
        self.af = nn.ReLU(inplace=True)
    def forward(self, feature):
        return self.conv2(self.af(self.conv1(self.af(feature)))) + self.conv_res(feature)
class para_optim(nn.Module):
    def __init__(self,autoencoder,pnum,para_num,feat_num,dim_feat,elastic_modulus,plastic_modulus,poisson_ratio,train_split):
        super(para_optim,self).__init__()
        self.pnum = pnum
        self.feat_num = feat_num
        self.dim_feat=dim_feat
        self.para_num = para_num
        self.afno=AFNO3D(hidden_size=dim_feat).cuda()
        self.linear1=Linear_ResBlock_point(self.feat_num,self.dim_feat).cuda()
        self.linear2 = Linear_ResBlock_process(self.para_num, self.feat_num).cuda()
        self.linear3 = Linear_ResBlock_process(48, self.para_num).cuda()
        self.linear4 = Linear_ResBlock_para(self.para_num, self.para_num).cuda()
        self.stress_increment_solver_mlp=stress_increment_solver_mlp(pnum).cuda()
        self.stress_increment_solver_physics=stress_increment_solver(pnum).cuda()
        self.elastic_modulus=elastic_modulus
        self.plastic_modulus=plastic_modulus
        self.poisson_ratio=poisson_ratio
        self.autoencoder=autoencoder
        self.bn0=nn.BatchNorm1d(128)
        self.bn1 = nn.BatchNorm1d(48)
        self.bn2= nn.BatchNorm1d(7)
        self.conv0=nn.Conv1d(128, 1, kernel_size=1)
        self.conv1 = nn.Conv1d(16, 1, kernel_size=1)
        self.relu=nn.ReLU(inplace=False)
        self.train_split=train_split
    def forward(self,X_m,X_d,keypoints_m,keypoints_d,stress_m,label,para_m,gt):
        with torch.no_grad():
            feat_m = self.encode(X_m, keypoints_m, label)#4,16,51
            feat_d = self.encode(X_d, keypoints_d, label)#4,16,51
            #feat_m = torch.randn(4,16,48).cuda()
            #feat_d = torch.randn(4,16,48).cuda()
        start = time.time()
        elastic_plastic_matrix = elastic_plastic_matrix_compute(stress_m, self.elastic_modulus,self.poisson_ratio, self.plastic_modulus)
        if self.train_split == 'physics':
            stress_incre,output_stress_incre=self.stress_increment_solver_physics(feat_d, feat_m, stress_m, self.elastic_modulus,self.plastic_modulus,self.poisson_ratio)
            loss_matrix = None
        if self.train_split=='learning_physics':
            stress_incre, elastic_plastic_matrix_mlp = self.stress_increment_solver_mlp(feat_d, feat_m, stress_m,
                                                                                    self.elastic_modulus,
                                                                                    self.plastic_modulus,
                                                                                    self.poisson_ratio)
            #print(elastic_plastic_matrix_mlp[0])
            #print(elastic_plastic_matrix[0])
            loss_matrix=F.mse_loss(elastic_plastic_matrix_mlp,elastic_plastic_matrix, reduction='mean')
        if self.train_split=='learning':
            stress_incre, elastic_plastic_matrix_mlp = self.stress_increment_solver_mlp(feat_d, feat_m, stress_m,
                                                                                    self.elastic_modulus,
                                                                                    self.plastic_modulus,
                                                                                    self.poisson_ratio)
            loss_matrix=None
        B,N,C=stress_incre.shape
        stress_incre=stress_incre.reshape(B,-1,self.dim_feat)
        b_stress_field = self.afno(stress_incre,[16,8])#B,128,48
        b_stress_field=F.relu(self.bn0(b_stress_field))
        b_stress_field= (self.conv0(b_stress_field)).squeeze(1)#B,16
        print('b_stress_field',b_stress_field.shape)
        feat_d= torch.max(feat_d, 2, keepdim=False)[0]#B,16
        print('feat_d',feat_d.shape)
        #feat_d=self.conv1(feat_d).squeeze(1)
        t_latent_feature =F.relu(self.bn1(self.linear1(feat_d)))#B,48
        print('t_latent_feature',t_latent_feature.shape)
        del_para =b_stress_field* t_latent_feature
        del_para = F.relu(self.bn2(self.linear3(del_para)))  # B,N_para
        print('del_para', del_para.shape)
        #output=del_para+self.linear4(para_m)
        output = del_para * self.linear4(para_m)
        end = time.time()
        infer_time= end-start
        print('infer_time:', start - end)
        torch.cuda.synchronize()

        '''
        另一种输出方案
        b_stress_field = self.afno(stress_feature)  # B,128,48
        t_latent_feature = latent_feature.transpose(1,2).contiguous()#B,48,16
        dot_product_0 = torch.multiply(b_stress_field, t_latent_feature)#B,128,16
        t_para_feature = self.linear2(process_para)  # B,16
        dot_product_1 = torch.multiply(dot_product_0, t_para_feature.unsqueeze(2))#B,128,1
        output = self.linear3(dot_product_1.squeeze(2))  # B,N_para
        '''
        loss_mse = F.mse_loss(output,gt, reduction='mean')
        loss_mae = torch.mean(torch.abs(output-gt))
        return loss_mse,loss_mae,loss_matrix,output,output_stress_incre,infer_time
    def encode(self, x, keypoint, label):
        # pdb.set_trace()
        # x is of shape B,N,C
        # keypoint is of shape B,N2,C2
        # label is of shape B
        # feature_at_keypoint is of shape B,N2,C3
        feature_at_keypoint = self.autoencoder.encode(x, keypoint, ts=None, label=label, sample_posterior=True)
        return feature_at_keypoint


if __name__ == '__main__':
    feat_d = torch.randn(3, 16, 48)
    feat_m = torch.randn(3, 16, 48)
    stress_m = torch.randn(3, 1024, 6)
    elastic_modulus = 1e11
    plastic_modulus = 1e13
    poisson_ratio = 0.3
    X_m=torch.randn(1, 1024, 3)
    X_d=torch.randn(1, 1024, 3)
    keypoints_m=torch.randn(1, 16, 3)
    keypoints_d=torch.randn(1, 16, 3)
    label=1
    para_m=torch.randn(1,7)
    gt=torch.randn(1,7)
    model=para_optim()
    macs, params = profile(para_optim, inputs=(X_m, X_d, keypoints_m, keypoints_d, stress_m, label, para_m, gt,))
    print(macs, params)