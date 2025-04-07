import argparse
import MyDataLoader
from DDPM_keypoint.autoencoder import AutoEncoder
from data_utils.curve_mask import plot_xyz, transformation
from data_utils.json_reader import read_json_file
from metrics_point_cloud.chamfer_and_f1 import calc_cd
import numpy as np
import os
import torch
from diffusion_utils.diffusion import LatentDiffusion
from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from models.autoencoder import PointAutoencoder
from data_utils.points_sampling import sample_keypoints
from data_utils.json_reader import replace_list_with_string_in_a_dict, restore_string_to_list_in_a_dict, read_json_file, autoencoder_read_config
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def evaluation(args,dataset,net):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    batch_size =1
    point_feature_dim=pointnet_config['in_fea_dim']
    dset_test = MyDataLoader.Dataset_for_shape_complex(trainset_config, dataset, split='test', transform=True)
    testloader = torch.utils.data.DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=0)
    results_hd_points = 0
    results_cd_points = 0
    results_f1_points=0
    save_dir='generated_point_cloud_and_mesh/'+args.keypoints_type+'/'+str(trainset_config['num_keypoints'])+'_keypoints'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for iter,data in enumerate(testloader):
            torch.cuda.synchronize()
            # load data
            X = data['points'].cuda()  # of shape (npoints, 3), roughly in the range of -scale to scale
            gt_keypoints = data['keypoints'].cuda()
            gt_keypoints=gt_keypoints.squeeze(0)
            label = data['label'].cuda()
            keypoints = gt_keypoints
            T= data['transform_matrix'].squeeze(0).cuda()
            scale_m=data['scale_m'].cuda()
            centroid = data['centroid'].cuda()
            data=torch.zeros(keypoints.shape[0],1024,6).cuda()
            print('unit_num',keypoints.shape[0])
            for i in range(keypoints.shape[0]):
                print('unit_ind', i)
                keypoints_unit=keypoints[i].unsqueeze(0)
                generated_data, _, _ = diffusion_model.denoise_and_reconstruct(
                    n=keypoints_unit.shape[0], model=net, keypoint_dim=3,shape=(trainset_config['num_keypoints'], 3 + point_feature_dim), label=label,
                    keypoint=keypoints_unit,
                    return_keypoint_feature=True,
                    local_resampling=False, complete_x0=None, keypoint_mask=None)
                data[i]=generated_data
            #plot_xyz(np.array(generated_data[:,:,:3].cpu()))

            generated_data=data[:,:,:3]*scale_m+centroid.squeeze(0)
            generated_data=np.array(generated_data[:, :, :3].cpu())
            T=np.array(T.cpu())
            grouped_unit=[]
            print('generated_data', generated_data.shape)
            print('T', T.shape)
            for unit_index in range(generated_data.shape[0]):
                transformed_points=transformation(generated_data[unit_index],T[unit_index],inverter=True)
                grouped_unit.append(transformed_points)
            grouped_unit=np.array(grouped_unit)
            print('grouped_unit',grouped_unit.shape)
            plot_xyz(grouped_unit)
            prediction_complete=grouped_unit.reshape(1,-1,3)
            prediction_complete, _ = sample_keypoints(prediction_complete, K=1024)
            prediction_complete=torch.tensor(prediction_complete).cuda()
            #loss_dict_keypoints = calc_cd(keypoints, gt_keypoints, calc_f1=False, f1_threshold=0.0001, normal_loss_type='mse')
            #loss_cd_keypoints = loss_dict_keypoints['cd_p'].mean()
            #print('loss_cd_keypoints:', loss_cd_keypoints)
            #results_cd_keypoints += loss_cd_keypoints / len(testloader)
            loss_dict_points = calc_cd(prediction_complete[:,:,:3], X, calc_f1=True, f1_threshold=0.0001,  normal_loss_type='mse')
            #print(loss_dict_points['cd_p'])
            loss_cd_points = loss_dict_points['cd_p'].mean()
            loss_f1_points = loss_dict_points['f1'].mean()
            loss_hd = loss_dict_points['hd'].mean().detach()
            print('loss_cd:', loss_cd_points, "loss_f1:", loss_f1_points, "loss_hd:", loss_hd)
            results_hd_points += loss_hd / len(testloader)
            results_cd_points += loss_cd_points / len(testloader)
            results_f1_points+= loss_f1_points / len(testloader)
            '''
            for j in range(keypoints.shape[0]):
                keypointsi=np.array(keypoints[j].cpu())
                np.savetxt(save_dir+'/'+str(iter*batch_size+j)+'_'+str(keypointsi.shape[0])+'_keypoints.pts',keypointsi,'%f %f %f')
                generated_datai = np.array(generated_data[:,:,:3][j])
                np.savetxt(save_dir + '/' + str(iter*batch_size+j) + '_' + str(generated_datai.shape[0]) + '_points.pts', generated_datai, '%f %f %f')
            '''
    #print('results_cd_keypoints:', results_cd_keypoints)
    print('results_cd_points:', results_cd_points)
    print('results_f1_points:', results_f1_points)
    print('results_f1_points:', results_hd_points)

if __name__ == "__main__":
    m_seed = 4
    torch.manual_seed(m_seed)
    torch.cuda.manual_seed(m_seed)
    # import pdb
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--keypoint_config', type=str, default='../configs/config_DDPM_keypoint.json', help='JSON file for configuration')
    parser.add_argument('--config', type=str, default='../configs/config_DDPM_latent.json', help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0, help='rank of process for distributed')
    parser.add_argument('-g', '--model_name', type=str, default='dpm', help='')
    parser.add_argument('-d', '--CUDA_VISIBLE_DEVICES', type=str, default="0",help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--split', type=str, default="test", help='')
    parser.add_argument('--keypoints_type', type=str, default='axis_complex', help='')
    parser.add_argument('--ckpt', type=str, default='../exps/exp_6FB_generation_1024/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_latent_feature_generation_6FB-3D/checkpoint/pointnet_ckpt_20399.pkl', help='the checkpoint to use')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    latent_config = read_json_file(args.config)
    config_file_path = os.path.split(args.config)[0]
    global dist_config
    dist_config = latent_config["dist_config"]         # to initialize distributed training
    global trainset_config
    trainset_config = latent_config['dataset_config']
    os.environ['CUDA_VISIBLE_DEVICES'] = latent_config["dist_config"]["CUDA_VISIBLE_DEVICES"]
    print('Visible GPUs are', os.environ['CUDA_VISIBLE_DEVICES'], flush=True)
    num_gpus = torch.cuda.device_count()
    print('%d GPUs are available' % num_gpus, flush=True)
    if num_gpus > 1:
        assert args.group_name != ''
    else:
        assert args.rank == 0
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    global config
    config = read_json_file(args.config)
    global pointnet_config
    pointnet_config = config["pointnet_config"]  # to define pointnet
    net = PointNet2CloudCondition(pointnet_config)
    state = torch.load(args.ckpt, map_location='cpu')
    net.load_state_dict(state['model_state_dict'])
    net.cuda()
    net.eval()
    global standard_diffusion_config
    standard_diffusion_config = config['standard_diffusion_config']
    # read autoencoder configs
    autoencoder_config_file = '../' + config['autoencoder_config']['config_file']
    global autoencoder_config
    autoencoder_config = read_json_file(autoencoder_config_file)
    autoencoder_config_file_path = os.path.split(autoencoder_config_file)[0]
    global encoder_config
    global decoder_config_list
    encoder_config, decoder_config_list = autoencoder_read_config(autoencoder_config_file_path, autoencoder_config)
    autoencoder = PointAutoencoder(encoder_config, decoder_config_list,
                                   apply_kl_regularization=autoencoder_config['pointnet_config'].get(
                                       'apply_kl_regularization', False),
                                   kl_weight=autoencoder_config['pointnet_config'].get('kl_weight', 0))
    # check if the path of ckpt is absolute path
    if not os.path.isabs(config['autoencoder_config']['ckpt']):
        config['autoencoder_config']['ckpt'] = os.path.join('../', config['autoencoder_config']['ckpt'])
    else:
        config['autoencoder_config']['ckpt'] = config['autoencoder_config']['ckpt']
    print(config['autoencoder_config']['ckpt'])
    autoencoder.load_state_dict(
        torch.load(config['autoencoder_config']['ckpt'], map_location='cpu')['model_state_dict'])
    autoencoder.cuda()
    autoencoder.eval()
    diffusion_model = LatentDiffusion(standard_diffusion_config, autoencoder=autoencoder)
    evaluation(args,trainset_config['dataset'],net)
    torch.cuda.synchronize()


