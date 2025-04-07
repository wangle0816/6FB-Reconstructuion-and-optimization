# -*- coding: mbcs -*-
#  Author : Le Wang
# Abaqus/CAE Release 2016 replay file
# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...

#1.本代码用于提取odb结果文件中的三维坐标及应力
import csv
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
import numpy as np
executeOnCaeStartup()
file_para= r"D:\WL\AbaqusWork\6FB_AutoModel_DDPM\parameter_sampling_6FB_3D_D25.csv"
fr = open(file_para,'r')#csv文件名
reader = csv.reader(fr)
paralist=list(reader)
for jobindex in range(0,100):
    D=float(paralist[jobindex+1][0])
    R=float(paralist[jobindex+1][1])
    file_data = r"D:\WL\AbaqusWork\6FB_AutoModel_DDPM\dataset-6FB-D25\dataset-6FB-tube-diffusion-D25-2048\6FB-3D\data-6FB-3D-"+str(jobindex)+".csv"
    odb_file = r"C:\SIMULIA\tempforrpy\Job-6FB-3D-"+str(jobindex)+".odb"
    #stress
    odb = session.openOdb(name=odb_file, readOnly=True)
    session.viewports['Viewport: 1'].setValues(displayedObject=odb)
    odbName=session.viewports[session.currentViewportName].odbDisplay.name
    session.odbData[odbName].setValues(activeFrames=(('Step-7', (-1, )), ))
    #Generation of stress data in post-processing
    excel_1=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('S', INTEGRATION_POINT, ((COMPONENT, 'S11'),\
            (COMPONENT, 'S22'), (COMPONENT,'S33'), (COMPONENT, 'S12'), (COMPONENT, 'S13'), (COMPONENT, 'S23'), )), ),nodeSets=('TUBE-1.SET-TUBE-NODE', ))

    stress_list=[]
    for tmp in excel_1:
        data=tmp.data
        stress_list.append(data)
    # free up space
    keys_list = list(session.xyDataObjects.keys())
    for key in keys_list:
        del session.xyDataObjects[key]

    # corr_orig
    odbName=session.viewports[session.currentViewportName].odbDisplay.name
    session.odbData[odbName].setValues(activeFrames=(('Step-1', (0, )), ))
    odb = session.odbs[odb_file]
    excel_2=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('COORD',
        NODAL, ((COMPONENT, 'COOR1'), (COMPONENT, 'COOR2'), (COMPONENT, 'COOR3'),
        )), ), nodeSets=('TUBE-1.SET-TUBE-NODE', ))

    coor_orig_list=[]
    for tmp in excel_2:
        data=tmp.data
        coor_orig_list.append(data)

    keys_list = list(session.xyDataObjects.keys())
    for key in keys_list:
        del session.xyDataObjects[key]

    # corr_def
    odbName=session.viewports[session.currentViewportName].odbDisplay.name
    session.odbData[odbName].setValues(activeFrames=(('Step-7', (-1, )), ))
    odb = session.odbs[odb_file]
    excel_3=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('COORD',
        NODAL, ((COMPONENT, 'COOR1'), (COMPONENT, 'COOR2'), (COMPONENT, 'COOR3'),
        )), ), nodeSets=('TUBE-1.SET-TUBE-NODE', ))

    coor_def_list=[]
    for tmp in excel_3:
        data=tmp.data
        coor_def_list.append(data)

    keys_list = list(session.xyDataObjects.keys())
    for key in keys_list:
        del session.xyDataObjects[key]

    #create array
    num_nodes=len(stress_list)/6
    s=np.zeros((num_nodes,13))
    for i in range(num_nodes):
        s[i, 0] = i
        s[i, 1] = coor_orig_list[num_nodes * 0 + i][0][1]
        s[i, 2] = coor_orig_list[num_nodes * 1 + i][0][1]
        s[i, 3] = coor_orig_list[num_nodes * 2 + i][0][1]
        s[i, 4] = coor_def_list[num_nodes * 0 + i][0][1]
        s[i, 5] = coor_def_list[num_nodes * 1 + i][0][1]
        s[i, 6] = coor_def_list[num_nodes * 2 + i][0][1]
        s[i, 7] = stress_list[num_nodes * 0 + i][0][1]
        s[i, 8] = stress_list[num_nodes * 1 + i][0][1]
        s[i, 9] = stress_list[num_nodes * 2 + i][0][1]
        s[i, 10] = stress_list[num_nodes * 3 + i][0][1]
        s[i, 11] = stress_list[num_nodes * 4 + i][0][1]
        s[i, 12] = stress_list[num_nodes * 5 + i][0][1]
    z_orig_max=max(s[:,3])
    #外壁点数据&目标弯曲段
    data_sele=s[(s[:,3]>z_orig_max-D-0.5*np.pi*R) & (np.sqrt(s[:,1]**2+s[:,2]**2)>D/2-0.5)]
    data_sele[:, 0] = np.arange(data_sele.shape[0])
    keys_name=['Index_node','x_orig','y_orig','z_orig','x_def','y_def','z_def','s11','s12','s13','s22','s23','s33']
    with open(file_data, 'wb') as csvfile:#'wb'防止CSV空白行的出现
        # 创建CSV写入器对象
        csv_writer = csv.writer(csvfile)
        # 写入列名
        csv_writer.writerow(keys_name)
        # 写入数据
        csv_writer.writerows(data_sele)
    print  "job "+str(jobindex)+" complete"





