
#  Author : Le Wang
# Abaqus/CAE Release 2016 replay file
# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...

#1.对于csv点云文件，本代码用于轴线坐标
import csv
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
file_para= r"D:\WL\AbaqusWork\6FB_AutoModel_DDPM\dataset-6FB-D25\dataset-6FB-tube-diffusion-D25-2048\6FB-3D\6FB-3D-para.csv"
fr = open(file_para,'r')#csv文件名
reader = csv.reader(fr)
paralist=list(reader)
for jobindex in range(0,80):
    D=float(paralist[jobindex+1][0])
    R=float(paralist[jobindex+1][1])
    file_data = r"D:\WL\AbaqusWork\6FB_AutoModel_DDPM\dataset-6FB-D25\dataset-6FB-tube-diffusion-D25-2048\6FB-3D\6FB-3D-data-axis-train\data-6FB-3D-axis-"+str(jobindex)+".csv"
    data_file = r"D:\WL\AbaqusWork\6FB_AutoModel_DDPM\dataset-6FB-D25\dataset-6FB-tube-diffusion-D25-2048\6FB-3D\6FB-3D-data-train\data-6FB-3D-"+str(jobindex)+".csv"
    fr = open(data_file, 'r')  # csv文件名
    data_reader = csv.reader(fr)
    data_paralist = list(data_reader)
    data_sele=data_paralist[1:]
    data_sele=np.array(data_sele).astype(float)
    data_sele=data_sele[:,:7]

    #1. 从原始坐标中找出不重复的z坐标
    #2. 按照z坐标值找出原始坐标对应的行号组，每组为一圈
    #3. 按行号组对应的变形后的坐标计算中心点
    #4. 中心点的数组为轴线

    z_unique=np.unique(data_sele[:,3])
    circle_node=[]
    axis_node=[]
    for i,z_unique_i in enumerate(z_unique):
        indices = np.where(data_sele[:,3] == z_unique_i)[0]
        circle_node.append(data_sele[indices,4:7])
        axis_node.append(np.mean(data_sele[indices,4:7],axis=0))
    axis_node=np.array(axis_node)

    keys_name=['x_axis','y_axis','z_axis']
    # 创建一个新的 3D 图形
    fig = plt.figure()

    # 添加 3D 子图
    ax = fig.add_subplot(111, projection='3d')

    # 绘制第一组点云，使用红色表示
    ax.scatter(axis_node[:, 0], axis_node[:, 1], axis_node[:, 2], c='r', marker='o', label='Group 1')

    # 绘制第二组点云，使用蓝色表示
    ax.scatter(data_sele[:, 4], data_sele[:, 5], data_sele[:, 6], c='b', marker='^', label='Group 2')
    ax.set_box_aspect([np.ptp(arr) for arr in [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])

    # 添加标签和图例
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # 显示图形
    #plt.show()
    with open(file_data, 'w', newline='', encoding='utf-8') as csvfile:
        # 创建CSV写入器对象
        csv_writer = csv.writer(csvfile)
        # 写入列名
        csv_writer.writerow(keys_name)
        # 写入数据
        csv_writer.writerows(axis_node)