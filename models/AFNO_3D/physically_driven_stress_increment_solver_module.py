#**********  Code for computing the elastic-plastic stiffness matrix and increment of strain and stress  **********
# author: Le Wang

import torch
from torch import nn
import torch.nn.functional as F

def elastic_plastic_matrix_compute(stress_direct,elastic_modulus,poisson_ratio,plastic_modulus):
    '''

    :param stress_direct:[B,N,6]
    :param elastic_modulus: constant
    :param poisson_ratio: constant
    :return:elastic_plastic_matrix: [B,6,6]
    '''


    B,N,_=stress_direct.shape
    lamda = elastic_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    shear_modulus = elastic_modulus / (2 * (1 + poisson_ratio))
    elastic_matrix = torch.tensor([[lamda + 2 * shear_modulus, 0, 0, 0, 0, 0],
                                   [lamda, lamda + 2 * shear_modulus, 0, 0, 0, 0],
                                   [lamda, lamda, lamda + 2 * shear_modulus, 0, 0, 0],
                                   [0, 0, 0, shear_modulus, 0, 0],
                                   [0, 0, 0, 0, shear_modulus, 0],
                                   [0, 0, 0, 0, 0, shear_modulus]]).cuda()
    equivalent_stress = torch.sqrt(
        0.5 * ((stress_direct[:,:, 0] - stress_direct[:,:, 1]) ** 2 + (stress_direct[:,:, 1] - stress_direct[:,:, 2]) ** 2 + \
               (stress_direct[:,:, 2] - stress_direct[:,:, 0]) ** 2 + 6 * (
                       stress_direct[:,:, 3] ** 2 + stress_direct[:,:, 4] ** 2 + stress_direct[:,:, 5] ** 2)))

    average_stress =( 1 / 3 * (stress_direct[:,:, 0] + stress_direct[:,:, 1] + stress_direct[:,:, 2]))
    bias_stress = (torch.stack(
        [stress_direct[:,:, 0] - average_stress, stress_direct[:,:, 1] - average_stress, stress_direct[:,:, 2] - average_stress, \
         stress_direct[:,:, 3], stress_direct[:,:, 4], stress_direct[:,:, 5]], dim=2)).transpose(1,2).contiguous()
    H_d = plastic_modulus / (1 - plastic_modulus / elastic_modulus)
    plastic_matrix = (9 * shear_modulus ** 2) / ((H_d + 3 * shear_modulus) * torch.matmul(
        equivalent_stress.unsqueeze(1), equivalent_stress.unsqueeze(-1)).expand(B, 6, 6)) * torch.matmul(bias_stress, bias_stress.transpose(1,2).contiguous())
    elastic_plastic_matrix = elastic_matrix.unsqueeze(0).expand(B, 6, 6) - plastic_matrix
    return elastic_plastic_matrix*1e-11


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
    def forward(self,feat_d,feat_m,stress_m,elastic_modulus,plastic_modulus,poisson_ratio):
        del_feat=feat_m-feat_d
        B,N,C=del_feat.shape
        del_feat=del_feat.reshape(B,16*48)
        del_feat = F.relu(self.bn1(self.fc1(del_feat)))
        del_feat = F.relu(self.bn2(self.fc2(del_feat)))
        del_feat = F.relu(self.fc3(del_feat))
        del_feat=del_feat.reshape(B,6,-1)

        pred_strain_incre=self.conv(del_feat).permute(0,2,1).contiguous()

        elastic_plastic_matrix=elastic_plastic_matrix_compute(stress_m,elastic_modulus,poisson_ratio,plastic_modulus)
        #print('elastic_plastic_matrix',elastic_plastic_matrix)
        pred_stress_increment = stress_field(elastic_plastic_matrix, pred_strain_incre)
        output_stress_incre=pred_stress_increment
        pred_stress_increment=pred_stress_increment.reshape(B,-1,self.dim_feat)
        return pred_stress_increment,output_stress_incre




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