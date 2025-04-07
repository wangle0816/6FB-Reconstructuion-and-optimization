import  numpy as np
D=25.0
n=24
R=8*D
P=8*D
size=np.pi*D/n
A_bg=35.0
r_out = D / 2.0
thickness_tube = 2.0
thickness_bd_half = 5.0
Dist = A_bg  # Actual 11+11
alpha0 = np.arcsin(Dist * 1 / R)
L = np.sqrt((2 * np.pi * R) ** 2 + P ** 2)
l_length = 0.4 * L + 20 + A_bg  + alpha0 * R
num_p=l_length/size*n

eta_k=np.linspace(0,6,num=7)
print(eta_k)