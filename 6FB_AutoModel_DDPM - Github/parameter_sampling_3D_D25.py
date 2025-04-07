# -*- coding: mbcs -*-
#  Author : Le Wang (zju.edu.cn)
# The code is for sampling processing parameters by Latin Hypercube.

import csv
import numpy as np
from pyDOE import lhs

D=[25,25]
R_D=[4.6,8.2]
P_D=[6,8]
compensation_coeff_A=[1.05,1.35]
compensation_coeff_R=[0.95,1.10]
v_pd=[10,50]
A_D=[2,2.5]
gap_bd=[0,0.3]
gap_gd=[0,0.3]
f_bd=[0,0.20]
num_samples=100
bounds = np.array([D, R_D,P_D,compensation_coeff_A,compensation_coeff_R,v_pd,A_D,gap_bd,gap_gd,f_bd])
samples = lhs(10, samples=num_samples, criterion='maximin')
scaled_samples = bounds[:, 0] + samples * (bounds[:, 1] - bounds[:, 0])
D = scaled_samples[:, 0]
R= scaled_samples[:, 1]*D
P = scaled_samples[:, 2]*D
compensation_coeff_A = scaled_samples[:, 3]
compensation_coeff_R = scaled_samples[:, 4]
v_pd = scaled_samples[:, 5]
A_bg = scaled_samples[:, 6]*D
gap_bd=scaled_samples[:, 7]
gap_gd=scaled_samples[:, 8]
f_bd=scaled_samples[:, 9]

data_dict = {
    'D': D,
    'R': R,
    'P': P,
    'compensation_coeff_A': compensation_coeff_A,
    'compensation_coeff_R': compensation_coeff_R,
    'v_pd': v_pd,
    'A_bg': A_bg,
    'gap_bd': gap_bd,
    'gap_gd': gap_gd,
    'f_bd': f_bd
}

data_array = np.array(list(data_dict.values())).T

# CSV name
csv_file_name = 'parameter_sampling_6FB_3D_D25.csv'


with open(csv_file_name, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(data_dict.keys())
    csv_writer.writerows(data_array)

