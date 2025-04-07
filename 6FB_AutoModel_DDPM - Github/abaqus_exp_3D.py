# -*- coding: mbcs -*-
#  Author : Le Wang (zju.edu.cn)
#  The code is a scripts for generate cae files according to the sampled processing parameters in ABAQUS2016.

# Abaqus/CAE Release 2016 replay file
# Internal Version: 2015_09_25-04.31.09 126547
# Run by Wangle on Fri Oct 29 15:05:06 2021

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
import csv
import numpy as np
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
import os
filePath ="F:/1_abaquswork/FE_FreeBending/6FB_AutoModel_DDPM/"#csv文件路径
fr = open(filePath+"parameter_sampling_6FB_3D_D25.csv",'r')#csv文件名
reader = csv.reader(fr)
paralist=list(reader)
PARALISTindex=range(len(paralist)-1)
#mdb.models.changeKey(fromName='Model-0', toName='Model-1')
for i in range(0,1):
    D =float(paralist[i+1][0])
    R = float(paralist[i+1][1])
    P = float(paralist[i+1][2])
    compensation_coeff_A = float(paralist[i+1][3])
    compensation_coeff_R = float(paralist[i+1][4])
    v_pd = float(paralist[i+1][5])
    A_bg = float(paralist[i+1][6])
    gap_bd = float(paralist[i+1][7])
    gap_gd = float(paralist[i+1][8])
    f_bd = float(paralist[i+1][9])

    r_out=D/2.0
    thickness_tube=2
    thickness_bd_half=5
    Dist=A_bg # Actual 11+11
    alpha0=np.arcsin(Dist*compensation_coeff_A/R)
    alpha_0=alpha0*180/np.pi

    num_node_section=24 #0.5*np.pi*R/(np.pi*D/n)*n=2048
    l_mesh=2*np.pi*r_out/num_node_section
    L = np.sqrt((2 * np.pi * R) ** 2 + P ** 2)
    l_length=0.4*L+20+A_bg+v_pd#+alpha0*R
    l_guide=l_length-20-A_bg+D
    num_node_section_bd=36
    delta=R*(1-np.cos(alpha0))
    alpha=alpha0*compensation_coeff_R
    eta = 0.4*(2 * np.pi * P) / (L)
    k_rotation=6
    eta_k=np.linspace(0,eta,num=k_rotation+1)
    disp_X=delta*(1-np.cos(eta_k))
    diff_disp_X=-np.diff(disp_X)
    disp_Y = delta *  np.sin(eta_k)
    diff_disp_Y = np.diff(disp_Y)
    angle_z=eta/k_rotation
    #part-bend
    s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__',
        sheetSize=200.0)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=STANDALONE)
    s1.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, 100.0))
    s1.FixedConstraint(entity=g[2])
    s1.ArcByCenterEnds(center=(r_out+5+gap_bd, 0.0), point1=(r_out+5+gap_bd, 5.0), point2=(r_out+5+gap_bd, -5.0),
        direction=COUNTERCLOCKWISE)
    p = mdb.models['Model-1'].Part(name='bend', dimensionality=THREE_D,
        type=DISCRETE_RIGID_SURFACE)
    p = mdb.models['Model-1'].parts['bend']
    p.BaseShellRevolve(sketch=s1, angle=360.0, flipRevolveDirection=OFF)
    s1.unsetPrimaryObject()
    p = mdb.models['Model-1'].parts['bend']
    del mdb.models['Model-1'].sketches['__profile__']
    ##reference point
    p = mdb.models['Model-1'].parts['bend']
    p.ReferencePoint(point=(0.0, 0.0, 0.0))
    mdb.models['Model-1'].parts['bend'].features.changeKey(fromName='RP',
        toName='RP-bd')
    ##set
    p = mdb.models['Model-1'].parts['bend']
    r = p.referencePoints
    refPoints=(r[2], )
    p.Set(referencePoints=refPoints, name='Set-bd')
    ##surface
    p = mdb.models['Model-1'].parts['bend']
    s = p.faces
    side1Faces = s.getSequenceFromMask(mask=('[#1 ]', ), )
    p.Surface(side1Faces=side1Faces, name='Surf-bd')

    #part-tube
    s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__',
        sheetSize=200.0)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=STANDALONE)
    s1.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(r_out, 0.0))
    s1.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(r_out-thickness_tube, 0.0))
    p = mdb.models['Model-1'].Part(name='tube', dimensionality=THREE_D,
        type=DEFORMABLE_BODY)
    p = mdb.models['Model-1'].parts['tube']
    p.BaseSolidExtrude(sketch=s1, depth=l_length)
    s1.unsetPrimaryObject()
    p = mdb.models['Model-1'].parts['tube']
    del mdb.models['Model-1'].sketches['__profile__']
    p.ReferencePoint(point=(0.0, 0.0, 0.0))
    mdb.models['Model-1'].parts['tube'].features.changeKey(fromName='RP',
        toName='RP-tube')
    p = mdb.models['Model-1'].parts['tube']
    r = p.referencePoints
    refPoints=(r[2], )
    p.Set(referencePoints=refPoints, name='Set-tube')
    p = mdb.models['Model-1'].parts['tube']
    s = p.faces
    side1Faces = s.getSequenceFromMask(mask=('[#1 ]', ), )
    p.Surface(side1Faces=side1Faces, name='Surf-tube')
    p = mdb.models['Model-1'].parts['tube']
    s = p.faces
    side1Faces = s.getSequenceFromMask(mask=('[#4 ]', ), )
    p.Surface(side1Faces=side1Faces, name='Surf-tube-section')


    #part-guide
    s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__',
        sheetSize=200.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(r_out+gap_gd, 0.0))
    p = mdb.models['Model-1'].Part(name='guide', dimensionality=THREE_D,
        type=DISCRETE_RIGID_SURFACE)
    p = mdb.models['Model-1'].parts['guide']
    p.BaseShellExtrude(sketch=s, depth=l_guide)
    s.unsetPrimaryObject()
    p = mdb.models['Model-1'].parts['guide']
    del mdb.models['Model-1'].sketches['__profile__']
    p = mdb.models['Model-1'].parts['guide']
    p.ReferencePoint(point=(0.0, 0.0, 0.0))
    mdb.models['Model-1'].parts['guide'].features.changeKey(fromName='RP',
        toName='RP-gd')
    p = mdb.models['Model-1'].parts['guide']
    r = p.referencePoints
    refPoints=(r[2], )
    p.Set(referencePoints=refPoints, name='Set-gd')
    p = mdb.models['Model-1'].parts['guide']
    s = p.faces
    side2Faces = s.getSequenceFromMask(mask=('[#1 ]', ), )
    p.Surface(side2Faces=side2Faces, name='Surf-gd')

    #part-push
    s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__',
        sheetSize=200.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, 100.0))
    s.FixedConstraint(entity=g[2])
    s.Line(point1=(0.0, 7.0), point2=(2.0, 7.0))
    s.HorizontalConstraint(entity=g[3], addUndoState=False)
    s.Line(point1=(2.0, 7.0), point2=(2.0, 5.0))
    s.VerticalConstraint(entity=g[4], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[3], entity2=g[4], addUndoState=False)
    s.Line(point1=(2.0, 5.0), point2=(r_out, 5.0))
    s.HorizontalConstraint(entity=g[5], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[4], entity2=g[5], addUndoState=False)
    s.Line(point1=(r_out, 5.0), point2=(r_out, -5.0))
    s.VerticalConstraint(entity=g[6], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[5], entity2=g[6], addUndoState=False)
    s.Line(point1=(r_out, -5.0), point2=(0.0, -5.0))
    s.HorizontalConstraint(entity=g[7], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[6], entity2=g[7], addUndoState=False)
    p = mdb.models['Model-1'].Part(name='push', dimensionality=THREE_D,
        type=DISCRETE_RIGID_SURFACE)
    p = mdb.models['Model-1'].parts['push']
    p.BaseShellRevolve(sketch=s, angle=360.0, flipRevolveDirection=OFF)
    s.unsetPrimaryObject()
    p = mdb.models['Model-1'].parts['push']
    del mdb.models['Model-1'].sketches['__profile__']
    p = mdb.models['Model-1'].parts['push']
    p.ReferencePoint(point=(0.0, -5.0, 0.0))
    p = mdb.models['Model-1'].parts['push']
    r = p.referencePoints
    refPoints=(r[2], )
    p.Set(referencePoints=refPoints, name='Set-pd')
    p = mdb.models['Model-1'].parts['push']
    s = p.faces
    side1Faces = s.getSequenceFromMask(mask=('[#10 ]', ), )
    p.Surface(side1Faces=side1Faces, name='Surf-pd')

    #property
    from material import createMaterialFromDataString
    createMaterialFromDataString('Model-1', 'SI_mm113111_6061-T6(GB)', '6-10',
        """{'specificHeat': {'temperatureDependency': OFF, 'table': ((895999989.0,),), 'dependencies': 0, 'law': CONSTANTVOLUME}, 'materialIdentifier': '', 'description': '\xb2\xc4\xc1\xcf\xbf\xe2\xc0\xb4\xd4\xb4: https://xcbjx.taobao.com/\n\xbb\xb6\xd3\xad\xbc\xd3\xc8\xebqq\xc8\xba : Abaqus \xbb\xb6\xc0\xd6\xb9\xb2\xbd\xf8\xc6\xbd\xcc\xa8 431603427\n6061-T6 (GB)[N_mm_T]\n\xb2\xe2\xca\xd4\xc0\xe0\xd0\xcd\xa3\xba\n\xcb\xb5\xc3\xf7\xa3\xba\n\xd0\xc5\xcf\xa2\xa3\xba\n\xce\xaa\xc2\xfa\xd7\xe3\xc6\xf3\xd2\xb5\xbb\xf2\xb8\xf6\xc8\xcb\xb6\xa8\xd6\xc6\xbb\xaf\xd0\xe8\xc7\xf3\xa3\xac\xce\xd2\xc3\xc7\xcc\xe1\xb9\xa9\xb2\xc4\xc1\xcf\xbf\xe2\xb6\xa8\xd6\xc6\xb7\xfe\xce\xf1\xa3\xac\xd2\xb2\xbd\xab\xcd\xc6\xb3\xf6\xb8\xfc\xb6\xe0\xc0\xa9\xb3\xe4\xa1\xa2\xd7\xa8\xd2\xb5\xb5\xc4\xb2\xc4\xc1\xcf\xbf\xe2\xa3\xac\xce\xaaABAQUS\xb7\xc2\xd5\xe6\xd0\xa7\xc2\xca\xcc\xe1\xb8\xdf\xb6\xf8\xc5\xac\xc1\xa6\xa1\xa3', 'elastic': {'temperatureDependency': OFF, 'moduli': LONG_TERM, 'noCompression': OFF, 'noTension': OFF, 'dependencies': 0, 'table': ((69000.0006661372, 0.33),), 'type': ISOTROPIC}, 'density': {'temperatureDependency': OFF, 'table': ((2.7e-09,),), 'dependencies': 0}, 'name': 'SI_mm113111_6061-T6(GB)', 'plastic': {'temperatureDependency': OFF, 'strainRangeDependency': OFF, 'rate': OFF, 'dependencies': 0, 'hardening': ISOTROPIC, 'dataType': HALF_CYCLE, 'table': ((275.0, 0.0), (275.0, 0.004), (275.79029, 0.01), (277.16924, 0.015), (282.68505, 0.02), (296.47456, 0.03), (310.26408, 0.04), (324.05359, 0.06), (344.73786, 0.08), (355.76948, 0.09)), 'numBackstresses': 1}, 'expansion': {'temperatureDependency': OFF, 'userSubroutine': OFF, 'zero': 0.0, 'dependencies': 0, 'table': ((2.4e-05,),), 'type': ISOTROPIC}, 'conductivity': {'temperatureDependency': OFF, 'table': ((166.9,),), 'dependencies': 0, 'type': ISOTROPIC}}""")
    mdb.models['Model-1'].HomogeneousSolidSection(name='Section-tube',
        material='SI_mm113111_6061-T6(GB)', thickness=None)
    p = mdb.models['Model-1'].parts['tube']
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
    region = p.Set(cells=cells, name='Set-section-tube')
    p = mdb.models['Model-1'].parts['tube']
    p.SectionAssignment(region=region, sectionName='Section-tube', offset=0.0,
        offsetType=MIDDLE_SURFACE, offsetField='',
        thicknessAssignment=FROM_SECTION)

    p = mdb.models['Model-1'].parts['bend']
    region=p.sets['Set-bd']
    mdb.models['Model-1'].parts['bend'].engineeringFeatures.PointMassInertia(
        name='Inertia-bd', region=region, mass=1.0, i11=1.0, i22=1.0, i33=1.0,
        alpha=0.0, composite=0.0)

    #step
    a = mdb.models['Model-1'].rootAssembly
    mdb.models['Model-1'].ExplicitDynamicsStep(name='Step-1', previous='Initial',
        massScaling=((SEMI_AUTOMATIC, MODEL, AT_BEGINNING, 10000.0, 0.0, None, 0, 0,
        0.0, 0.0, 0, None), ))
    mdb.models['Model-1'].ExplicitDynamicsStep(name='Step-2', previous='Step-1',
        timePeriod=(0.4*L)/v_pd/k_rotation, massScaling=((SEMI_AUTOMATIC, MODEL, AT_BEGINNING, 10000.0,
        0.0, None, 0, 0, 0.0, 0.0, 0, None), ))
    mdb.models['Model-1'].ExplicitDynamicsStep(name='Step-3', previous='Step-2',
                                               timePeriod=(0.4*L)/v_pd/k_rotation,
                                               massScaling=((SEMI_AUTOMATIC, MODEL, AT_BEGINNING, 10000.0,
                                                             0.0, None, 0, 0, 0.0, 0.0, 0, None),))
    mdb.models['Model-1'].ExplicitDynamicsStep(name='Step-4', previous='Step-3',
                                               timePeriod=(0.4*L)/v_pd/k_rotation,
                                               massScaling=((SEMI_AUTOMATIC, MODEL, AT_BEGINNING, 10000.0,
                                                             0.0, None, 0, 0, 0.0, 0.0, 0, None),))
    mdb.models['Model-1'].ExplicitDynamicsStep(name='Step-5', previous='Step-4',
                                               timePeriod=(0.4*L)/v_pd/k_rotation,
                                               massScaling=((SEMI_AUTOMATIC, MODEL, AT_BEGINNING, 10000.0,
                                                             0.0, None, 0, 0, 0.0, 0.0, 0, None),))
    mdb.models['Model-1'].ExplicitDynamicsStep(name='Step-6', previous='Step-5',
                                               timePeriod=(0.4*L)/v_pd/k_rotation,
                                               massScaling=((SEMI_AUTOMATIC, MODEL, AT_BEGINNING, 10000.0,
                                                             0.0, None, 0, 0, 0.0, 0.0, 0, None),))
    mdb.models['Model-1'].ExplicitDynamicsStep(name='Step-7', previous='Step-6',
                                               timePeriod=(0.4*L)/v_pd/k_rotation,
                                               massScaling=((SEMI_AUTOMATIC, MODEL, AT_BEGINNING, 10000.0,
                                                             0.0, None, 0, 0, 0.0, 0.0, 0, None),))
    mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
        'S', 'SVAVG', 'PE', 'PEVAVG', 'PEEQ', 'PEEQVAVG', 'LE', 'U', 'V', 'A',
        'RF', 'CSTRESS', 'EVF', 'COORD'))

    #assembly
    a = mdb.models['Model-1'].rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    p = mdb.models['Model-1'].parts['tube']
    a.Instance(name='tube-1', part=p, dependent=ON)
    p = mdb.models['Model-1'].parts['bend']
    a.Instance(name='bend-1', part=p, dependent=ON)
    p = mdb.models['Model-1'].parts['push']
    a.Instance(name='push-1', part=p, dependent=ON)
    p = mdb.models['Model-1'].parts['guide']
    a.Instance(name='guide-1', part=p, dependent=ON)
    a.rotate(instanceList=('bend-1', ), axisPoint=(0.0, 0.0, 0.0), axisDirection=(
        1.0, 0.0, 0.0), angle=90.0)
    a.translate(instanceList=('tube-1', ), vector=(0.0, 0.0, -Dist-20))
    a.translate(instanceList=('bend-1', ), vector=(0.0, 0.0, -Dist))
    a.rotate(instanceList=('push-1', ), axisPoint=(0.0, 0.0, 0.0), axisDirection=(
        1.0, 0.0, 0.0), angle=90.0)
    a.translate(instanceList=('push-1', ), vector=(0.0, 0.0,l_guide+5))


    #interaction
    mdb.models['Model-1'].ContactProperty('IntProp-frictionless')
    mdb.models['Model-1'].interactionProperties['IntProp-frictionless'].TangentialBehavior(
        formulation=FRICTIONLESS)
    mdb.models['Model-1'].ContactProperty('IntProp-rough')
    mdb.models['Model-1'].interactionProperties['IntProp-rough'].TangentialBehavior(
        formulation=ROUGH)
    mdb.models['Model-1'].ContactProperty('IntProp-penalty')
    mdb.models['Model-1'].interactionProperties['IntProp-penalty'].TangentialBehavior(
        formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF,
        pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=((
        f_bd, ), ), shearStressLimit=None, maximumElasticSlip=FRACTION,
        fraction=0.005, elasticSlipStiffness=None)
    mdb.models['Model-1'].interactionProperties['IntProp-penalty'].NormalBehavior(
        pressureOverclosure=HARD, allowSeparation=ON,
        constraintEnforcementMethod=DEFAULT)
    a = mdb.models['Model-1'].rootAssembly
    region1=a.instances['bend-1'].surfaces['Surf-bd']
    a = mdb.models['Model-1'].rootAssembly
    region2=a.instances['tube-1'].surfaces['Surf-tube']
    mdb.models['Model-1'].SurfaceToSurfaceContactExp(name ='Int-bd-tube',
        createStepName='Step-1', master = region1, slave = region2,
        mechanicalConstraint=PENALTY, sliding=FINITE,
        interactionProperty='IntProp-penalty', initialClearance=OMIT,
        datumAxis=None, clearanceRegion=None)
    a = mdb.models['Model-1'].rootAssembly
    region1=a.instances['guide-1'].surfaces['Surf-gd']
    a = mdb.models['Model-1'].rootAssembly
    region2=a.instances['tube-1'].surfaces['Surf-tube']
    mdb.models['Model-1'].SurfaceToSurfaceContactExp(name ='Int-gd-tube',
        createStepName='Step-1', master = region1, slave = region2,
        mechanicalConstraint=PENALTY, sliding=FINITE,
        interactionProperty='IntProp-frictionless', initialClearance=OMIT,
        datumAxis=None, clearanceRegion=None)
    a = mdb.models['Model-1'].rootAssembly
    region1=a.instances['push-1'].surfaces['Surf-pd']
    a = mdb.models['Model-1'].rootAssembly
    region2=a.instances['tube-1'].surfaces['Surf-tube-section']
    mdb.models['Model-1'].SurfaceToSurfaceContactExp(name ='Int-pb-tube',
        createStepName='Step-1', master = region1, slave = region2,
        mechanicalConstraint=PENALTY, sliding=FINITE,
        interactionProperty='IntProp-rough', initialClearance=OMIT, datumAxis=None,
        clearanceRegion=None)

    #load
    a = mdb.models['Model-1'].rootAssembly
    mdb.models['Model-1'].TabularAmplitude(name='Amp-bd', timeSpan=STEP,
        smooth=SOLVER_DEFAULT, data=((0.0, 0.0), (1.0, 1.0)))
    mdb.models['Model-1'].TabularAmplitude(name='Amp-rotation', timeSpan=STEP,
                                           smooth=SOLVER_DEFAULT, data=((0.0, 0.0), ((0.4*L)/v_pd/k_rotation, 1.0)))
    region = a.instances['bend-1'].sets['Set-bd']
    mdb.models['Model-1'].DisplacementBC(name='BC-bd', createStepName='Step-1',
        region=region, u1=delta, u2=0.0, u3=0.0, ur1=0.0, ur2=-alpha, ur3=0.0,
        amplitude='Amp-bd', fixed=OFF, distributionType=UNIFORM, fieldName='',
        localCsys=None)
    mdb.models['Model-1'].boundaryConditions['BC-bd'].setValuesInStep(
        stepName='Step-2', u1=diff_disp_X[0], u2=diff_disp_Y[0],ur2=0,ur3=angle_z, amplitude='Amp-rotation')
    mdb.models['Model-1'].boundaryConditions['BC-bd'].setValuesInStep(
        stepName='Step-3', u1=diff_disp_X[1], u2=diff_disp_Y[1],ur3=angle_z, amplitude='Amp-rotation')
    mdb.models['Model-1'].boundaryConditions['BC-bd'].setValuesInStep(
        stepName='Step-4', u1=diff_disp_X[2], u2=diff_disp_Y[2], ur3=angle_z, amplitude='Amp-rotation')
    mdb.models['Model-1'].boundaryConditions['BC-bd'].setValuesInStep(
        stepName='Step-5', u1=diff_disp_X[3], u2=diff_disp_Y[3], ur3=angle_z, amplitude='Amp-rotation')
    mdb.models['Model-1'].boundaryConditions['BC-bd'].setValuesInStep(
        stepName='Step-6', u1=diff_disp_X[4], u2=diff_disp_Y[4], ur3=angle_z, amplitude='Amp-rotation')
    mdb.models['Model-1'].boundaryConditions['BC-bd'].setValuesInStep(
        stepName='Step-7', u1=diff_disp_X[5], u2=diff_disp_Y[5], ur3=angle_z, amplitude='Amp-rotation')
    region = a.instances['push-1'].sets['Set-pd']
    mdb.models['Model-1'].VelocityBC(name='BC-pd', createStepName='Step-1',
        region=region, v1=0.0, v2=0.0, v3=-v_pd, vr1=0.0, vr2=0.0, vr3=0.0,
        amplitude=UNSET, localCsys=None, distributionType=UNIFORM, fieldName='')
    region = a.instances['guide-1'].sets['Set-gd']
    mdb.models['Model-1'].EncastreBC(name='BC-gd', createStepName='Step-1',
        region=region, localCsys=None)


    #mesh
    p = mdb.models['Model-1'].parts['tube']
    e = p.edges
    pickedEdges = e.getSequenceFromMask(mask=('[#1 ]', ), )
    p.seedEdgeByNumber(edges=pickedEdges, number=num_node_section, constraint=FINER)
    p = mdb.models['Model-1'].parts['tube']
    e = p.edges
    pickedEdges = e.getSequenceFromMask(mask=('[#4 ]', ), )
    p.seedEdgeByNumber(edges=pickedEdges, number=num_node_section, constraint=FINER)
    p = mdb.models['Model-1'].parts['tube']
    p.seedPart(size=l_mesh, deviationFactor=0.1, minSizeFactor=0.1)
    p = mdb.models['Model-1'].parts['tube']
    pickedRegions = c.getSequenceFromMask(mask=('[#1 ]',), )
    p.setMeshControls(regions=pickedRegions, algorithm=MEDIAL_AXIS)
    p.generateMesh()

    p = mdb.models['Model-1'].parts['tube']
    n = p.nodes
    p.Set(nodes=n, name='Set-tube-node')

    p = mdb.models['Model-1'].parts['guide']
    e = p.edges
    pickedEdges = e.getSequenceFromMask(mask=('[#1 ]', ), )
    p.seedEdgeByNumber(edges=pickedEdges, number=36, constraint=FINER)
    p.seedPart(size=2*np.pi*r_out/num_node_section, deviationFactor=0.1, minSizeFactor=0.1)
    p.generateMesh()

    p = mdb.models['Model-1'].parts['push']
    e = p.edges
    pickedEdges = e.getSequenceFromMask(mask=('[#80 ]', ), )
    p.seedEdgeByNumber(edges=pickedEdges, number=36, constraint=FINER)
    p = mdb.models['Model-1'].parts['push']
    p.seedPart(size=2*np.pi*r_out/num_node_section, deviationFactor=0.1, minSizeFactor=0.1)
    p.generateMesh()

    p = mdb.models['Model-1'].parts['bend']
    e = p.edges
    pickedEdges = e.getSequenceFromMask(mask=('[#2 ]', ), )
    p.seedEdgeByNumber(edges=pickedEdges, number=48, constraint=FINER)
    p = mdb.models['Model-1'].parts['bend']
    e = p.edges
    pickedEdges = e.getSequenceFromMask(mask=('[#4 ]', ), )
    p.seedEdgeByNumber(edges=pickedEdges, number=48, constraint=FINER)
    p.seedPart(size=2*np.pi*(r_out+5)/num_node_section_bd, deviationFactor=0.1, minSizeFactor=0.1)
    p = mdb.models['Model-1'].parts['bend']
    p.generateMesh()

    #job
    mdb.models.changeKey(fromName='Model-1', toName='Model-1-' + str(i))  # 修改模型树中模型名
    mdb.Model(name='Model-1', modelType=STANDARD_EXPLICIT)  # 建立新模型
    mdb.Job(name='Job-6FB-3D-'+ str(i), model='Model-1-' + str(i), description='', type=ANALYSIS,
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90,
        memoryUnits=PERCENTAGE, explicitPrecision=DOUBLE,
        nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF,
        contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='',
        resultsFormat=ODB, parallelizationMethodExplicit=DOMAIN, numDomains=1,
        activateLoadBalancing=False, multiprocessingMode=DEFAULT, numCpus=1)
del mdb.models['Model-1']
#mdb.saveAs(pathName='F:/1_abaquswork/irregular_spiral/odb'+paralist[i+1][0])#mdb文件保存路径及文件名
print 'End of programm'



