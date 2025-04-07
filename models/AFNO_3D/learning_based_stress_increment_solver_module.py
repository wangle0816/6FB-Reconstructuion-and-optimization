#**********  Code for computing the elastic-plastic stiffness matrix and increment of strain and stress  **********
# author: Le Wang

import torch
from torch import nn
import torch.nn.functional as F

class Linear_ResBlock_process(nn.Module):
    def __init__(self, input_size=1024*9, output_size=36):
        super(Linear_ResBlock_process, self).__init__()
        self.conv1 = nn.Linear(input_size, input_size)
        self.conv2 = nn.Linear(input_size, output_size)
        self.conv_res = nn.Linear(input_size, output_size)
        self.af = nn.ReLU(inplace=True)

    def forward(self, feature):
        return self.conv2(self.af(self.conv1(self.af(feature)))) + self.conv_res(feature)


def stress_field(elastic_plastic_matrix,strain_increment):
    stress_increment_design=torch.matmul(strain_increment,elastic_plastic_matrix)
    return stress_increment_design

class stress_increment_solver(nn.Module):
    def __init__(self,pnum,dim_feat=48):
        super().__init__()
        self.dim_feat = dim_feat
        self.fc1 = nn.Linear(16*48, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, pnum*6)
        self.conv = nn.Conv1d(6, 6, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.res = Linear_ResBlock_process().cuda()
    def forward(self,feat_d,feat_m,stress_m,elastic_modulus,plastic_modulus,poisson_ratio):
        del_feat=feat_m-feat_d
        B,N,C=del_feat.shape
        del_feat=del_feat.reshape(B,16*48)
        del_feat = F.relu(self.bn1(self.fc1(del_feat)))
        del_feat = F.relu(self.bn2(self.fc2(del_feat)))
        del_feat = F.relu(self.fc3(del_feat))
        del_feat=del_feat.reshape(B,6,-1)
        pred_strain_incre=self.conv(del_feat).permute(0,2,1).contiguous()
        para_material=torch.tensor([elastic_modulus/1e11,poisson_ratio,plastic_modulus/1e11]).squeeze(0)
        para_material=para_material.expand(B,1024,-1).cuda()
        input=torch.cat((stress_m,para_material),dim=-1).reshape(B,-1)
        elastic_plastic_matrix=self.res(input)
        elastic_plastic_matrix=elastic_plastic_matrix.reshape(B,6,6)
        pred_stress_increment = stress_field(elastic_plastic_matrix, pred_strain_incre)
        pred_stress_increment=pred_stress_increment.reshape(B,-1,self.dim_feat)
        return pred_stress_increment,elastic_plastic_matrix




if __name__=='__main__':
    feat_d=torch.randn(3,16,48)
    feat_m=torch.randn(3,16,48)
    stress_m=torch.randn(3,1024,6)
    elastic_modulus=1e11
    plastic_modulus=1e13
    poisson_ratio=0.3
    model=stress_increment_solver(1024)
    pred_stress_increment=model(feat_d,feat_m,stress_m,elastic_modulus,plastic_modulus,poisson_ratio)
    print(pred_stress_increment.shape)