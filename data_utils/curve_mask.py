import numpy as np
import torch
from scipy.spatial import distance_matrix
from scipy.spatial.transform import Rotation as R
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation, ball_query, three_nn, three_interpolate, grouping_operation
import matplotlib.pyplot as plt
import numpy.random
from mpl_toolkits.mplot3d import Axes3D # 空间三维画图
def plot_xyz(A):
    #设置x、y、z轴
    colors=plt.cm.jet(np.linspace(0,1,A.shape[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', )
    #ax.set_xlim(-1, 1)
    #ax.set_ylim(-1, 1)
    #ax.set_zlim(-1, 1)
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')
    for i in range(A.shape[0]):
        x=A[i,:,0]
        y=A[i,:,1]
        z=A[i,:,2]
        ax.scatter(x, y, z,c=colors[i])
    ax.view_init(elev=90, azim=0)
    ax.set_axis_off()
    plt.show()
def calculate_tangent_vector(points):
    dist_matrix = distance_matrix(points, points)

    # 计算每个点到其他所有点的平均距离
    avg_distances = np.mean(dist_matrix, axis=1)

    # 找到平均距离最大的两个点的索引
    endpoint_indices = np.argsort(avg_distances)[-2:]

    # 获取端点
    endpoints = points[endpoint_indices]
    # 选择一个端点，找到最近的点
    selected_endpoint_index = endpoint_indices[0]
    selected_endpoint = points[selected_endpoint_index]

    # 从距离矩阵中获取该端点到其他点的距离
    distances_from_selected_endpoint = dist_matrix[selected_endpoint_index]

    # 排除自己，找到最近的点索引
    closest_point_index = np.argsort(distances_from_selected_endpoint)[1]
    # 获取最近的点
    closest_point = points[closest_point_index]
    tangent_vector = closest_point - selected_endpoint
    if np.linalg.norm(tangent_vector)==0.0:
        closest_point_index = np.argsort(distances_from_selected_endpoint)[2]
        # 获取最近的点
        closest_point = points[closest_point_index]
        tangent_vector = closest_point - selected_endpoint

    return selected_endpoint,tangent_vector


def translate_to_origin_matrix(A):
    T = np.eye(4)
    T[:3, 3] = -A
    return T


def rotate_to_align_with_negative_z_matrix(tangent_vector):
    tangent_unit_vector = tangent_vector / np.linalg.norm(tangent_vector)
    z_negative = np.array([0, 0, -1])
    cross_product = np.cross(tangent_unit_vector, z_negative)
    dot_product = np.dot(tangent_unit_vector, z_negative)

    k = cross_product
    sin_theta = np.linalg.norm(k)
    cos_theta = dot_product

    if sin_theta == 0:
        return np.eye(4)

    kx, ky, kz = k / sin_theta
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])

    R_3x3 = np.eye(3) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
    R_4x4 = np.eye(4)
    R_4x4[:3, :3] = R_3x3
    return R_4x4


def optimize_rotation_angle(points):
    max_x = 0.0
    best_angle = 0
    best_points = None

    for angle in np.linspace(0, 2 * np.pi, 360):
        rotation_matrix = R.from_euler('z', angle).as_matrix()
        rotated_points = np.dot(points, rotation_matrix.T)

        #if np.all(rotated_points[:, 0] > 0) and np.all(rotated_points[:, 1] > 0):
        x = np.max(rotated_points[:, 0])
        if x > max_x:
            max_x = x
            best_angle = angle
            best_points = rotated_points

    best_rotation_matrix = R.from_euler('z', best_angle).as_matrix()
    R2 = np.eye(4)
    R2[:3, :3] = best_rotation_matrix
    return R2


def transform_curve_and_get_matrix(points):
    selected_endpoint,tangent_vector = calculate_tangent_vector(points)
    A = np.array(selected_endpoint)
    T = translate_to_origin_matrix(A)
    R1 = rotate_to_align_with_negative_z_matrix(tangent_vector)
    translated_points = np.dot(T, np.hstack((points, np.ones((len(points), 1)))).T).T[:, :3]
    rotated_points = np.dot(R1, np.hstack((translated_points, np.ones((len(translated_points), 1)))).T).T[:, :3]
    R2 = optimize_rotation_angle(rotated_points)
    M = R2 @ R1 @ T
    return M

def transformation(points,transformation_matrix,inverter=False):
    if inverter==False:
        homogeneous_points=np.hstack((points,np.ones((len(points),1))))
        transformed_points=np.dot(transformation_matrix,homogeneous_points.T).T[:,:3]
    else:
        transformation_matrix=np.linalg.inv(transformation_matrix)
        homogeneous_points = np.hstack((points, np.ones((len(points), 1))))
        transformed_points = np.dot(transformation_matrix, homogeneous_points.T).T[:, :3]
    return transformed_points

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def query_knn(nsample, xyz, new_xyz, include_self=True):
    """Find k-NN of new_xyz in xyz"""
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)  # B, S, N
    idx = torch.argsort(sqrdists, dim=-1, descending=False)[:, :, pad: nsample+pad]
    return idx.int()


def sample_and_group_knn(xyz_axis,center_npoint, k):
    """
    Args:
        xyz: Tensor, (B, 3, N)
        points: Tensor, (B, f, N)
        npoint: int
        nsample: int
        radius: float
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, npoint)
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample)
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)

    """
    xyz_flipped = xyz_axis.permute(0, 2, 1).contiguous()  # (B, N, 3)
    xyz_axis_knn_center = gather_operation(xyz_axis, furthest_point_sample(xyz_flipped, center_npoint))  # (B, 3, center_npoint)
    idx = query_knn(k, xyz_flipped, xyz_axis_knn_center.permute(0, 2, 1).contiguous())
    grouped_xyz = grouping_operation(xyz_axis, idx)  # (B, 3, center_npoint, nsample)
    return xyz_axis_knn_center, idx, grouped_xyz

def unit_sampling_transform(xyz_axis,center_npoint,k):
    xyz_axis = torch.tensor(xyz_axis).unsqueeze(0).cuda()
    xyz_axis = xyz_axis.transpose(1, 2).contiguous()
    _, _, grouped_xyz = sample_and_group_knn(xyz_axis, center_npoint, k)
    grouped_xyz = grouped_xyz.squeeze(0).permute(1, 2, 0).contiguous()
    grouped_xyz = np.array(grouped_xyz.cpu())
    grouped_unit = []
    T = []
    for unit_index in range(grouped_xyz.shape[0]):
        unit_point = grouped_xyz[unit_index]
        M = transform_curve_and_get_matrix(grouped_xyz[unit_index])
        transformed_points = transformation(unit_point, M)
        grouped_unit.append(transformed_points)
        T.append(M)
    grouped_unit = np.array(grouped_unit)
    T = np.array(T)

    return grouped_unit,T

if __name__=='__main__':
    # 示例散点
    xyz_axis =np.loadtxt('../data/dataset-6FB-tube-complex/complex-6FB-3D/data-axis-test/axis_FB3-tube.pts',dtype=np.float32)
    xyz_axis=torch.tensor(xyz_axis).unsqueeze(0).cuda()
    xyz_axis=xyz_axis.transpose(1,2).contiguous()
    center_npoint=20
    k=16
    _,_,grouped_xyz=sample_and_group_knn(xyz_axis, center_npoint, k)
    grouped_xyz=grouped_xyz.squeeze(0).permute(1,2,0).contiguous()
    grouped_xyz=np.array(grouped_xyz.cpu())
    grouped_unit =[]
    T=[]
    for unit_index in range(grouped_xyz.shape[0]):
        unit_point=grouped_xyz[unit_index]
        M = transform_curve_and_get_matrix(grouped_xyz[unit_index])
        transformed_points=transformation(unit_point,M)
        grouped_unit.append(transformed_points)
        T.append(M)
    grouped_unit=np.array(grouped_unit)
    T=np.array(T)
    print(grouped_unit.shape)
