# -*- coding: mbcs -*-
#  Author : Le Wang (zju.edu.cn)
# The code is for sampling processing parameters by Latin Hypercube.

import csv
import numpy as np
from pyDOE import lhs

D=[25,25]
R_D=[4.6,8.2]
compensation_coeff_A=[1.05,1.40]
compensation_coeff_B=[0.95,1.05]
v_pd=[10,50]
A_D=[1.25,1.75]
gap_bd=[0,0.3]
gap_gd=[0,0.3]
f_bd=[0,0.20]
num_samples=100
bounds = np.array([D, R_D,compensation_coeff_A,compensation_coeff_B,v_pd,A_D,gap_bd,gap_gd,f_bd])
samples = lhs(9, samples=num_samples, criterion='maximin')
print(samples)
scaled_samples = bounds[:, 0] + samples * (bounds[:, 1] - bounds[:, 0])
D = scaled_samples[:, 0]
R= scaled_samples[:, 1]*D
compensation_coeff_A = scaled_samples[:, 2]
compensation_coeff_B = scaled_samples[:, 3]
v_pd = scaled_samples[:, 4]
A_bg = scaled_samples[:, 5]*D
gap_bd = scaled_samples[:, 6]
gap_gd=scaled_samples[:, 7]
f_bd=scaled_samples[:, 8]


data_dict = {
    'D': D,
    'R': R,
    'compensation_coeff_A': compensation_coeff_A,
    'compensation_coeff_B': compensation_coeff_B,
    'v_pd': v_pd,
    'A_bg': A_bg,
    'gap_bd': gap_bd,
    'gap_gd': gap_gd,
    'f_bd': f_bd
}

data_array = np.array(list(data_dict.values())).T

# CSV name
csv_file_name = 'parameter_sampling_6FB_2D_D25.csv'


with open(csv_file_name, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(data_dict.keys())
    csv_writer.writerows(data_array)

