import sys
sys.path.append('../')

import pdb
import os
import json
import copy
import argparse
import numpy as np
import torch
from util import print_size
from diffusion_utils.diffusion import LatentDiffusion
from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from models.autoencoder import PointAutoencoder
from point_cloud_evaluation import evaluate_per_rank
from data_utils.json_reader import replace_list_with_string_in_a_dict, restore_string_to_list_in_a_dict, read_json_file, autoencoder_read_config


if __name__ == '__main__':
    m_seed=1   #spiral (2) #exp2D(1) #exp3D(3)
    torch.manual_seed(m_seed)
    torch.cuda.manual_seed(m_seed)
    # this file generate features on given keypoints and then reconstruct the dense point clouds
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/config_DDPM_latent.json', help='JSON file for configuration')
    parser.add_argument('--ckpt', type=str, default='../exps/exp_6FB_generation_1024/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_latent_feature_generation_6FB-2D/checkpoint/pointnet_ckpt_23999.pkl', help='the checkpoint to use')
    parser.add_argument('--ema_idx', type=int, default=0, help='the idx of the ema state to use')
    parser.add_argument('--keypoint_file', type=str, default='exps/ddpm_generated_point_clouds/shapenet_psr_generated_data_16_pts.npz', help='the npz file that stores the keypoints, it should contain keys: points(keypoints: shape (B,N,3)), label (B), category (B), category_name (B)')
    parser.add_argument('--save_dir', type=str, default='exps/generated_point_cloud_and_mesh/16_keypoints/', help='the directory to save point clouds')
    parser.add_argument('--batch_size', type=int, default=4, help='the batchsize to use')
    parser.add_argument('--local_resampling', action='store_true', help='if false, we sample features for all points in keypoint_file; if true, we resample features only for a portion of points in keypoint_file, while fix features for other points. In this case, keypoint_file should also contain keys keypoint_feature (B,N,F), and keypoint_mask (B,N) contain 0 and 1. 1 indicates points we want to resample features for.')
    parser.add_argument('--not_include_idx_to_save_name', action='store_true', help='whether to not include idx to the save name of generated point clouds. This is used only when each point cloud has a unique category_name')
    parser.add_argument('--save_keypoint_feature', action='store_true', help='whether to save the generated features at every keypoint')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    global config
    config = read_json_file(args.config)
    print('The configuration is:')
    print(json.dumps(replace_list_with_string_in_a_dict(copy.deepcopy(config)), indent=4))
    
    global train_config
    train_config = config["train_config"]        # training parameters

    global pointnet_config
    pointnet_config = config["pointnet_config"]     # to define pointnet

    
    global trainset_config
    trainset_config = config['dataset_config']


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
    print('The autoencoder configuration is:')
    print(json.dumps(replace_list_with_string_in_a_dict(copy.deepcopy(autoencoder_config)), indent=4))
       
    os.environ['CUDA_VISIBLE_DEVICES'] = config["dist_config"]["CUDA_VISIBLE_DEVICES"]

    try:
        print('Using cuda device', os.environ["CUDA_VISIBLE_DEVICES"], flush=True)
    except:
        print('CUDA_VISIBLE_DEVICES has bot been set', flush=True)
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    net = PointNet2CloudCondition(pointnet_config)
    state = torch.load(args.ckpt, map_location='cpu')
    
    net.load_state_dict(state['model_state_dict'])
    if args.ema_idx >= 0:
        net.load_state_dict(state['ema_state_list'][args.ema_idx])
    net.cuda()
    net.eval()
    print('latent ddpm size:')
    print_size(net)

    autoencoder = PointAutoencoder(encoder_config, decoder_config_list, 
                apply_kl_regularization=autoencoder_config['pointnet_config'].get('apply_kl_regularization', False),
                kl_weight=autoencoder_config['pointnet_config'].get('kl_weight', 0))
    # check if the path of ckpt is absolute path
    if not os.path.isabs(config['autoencoder_config']['ckpt']):
        config['autoencoder_config']['ckpt'] = os.path.join('../', config['autoencoder_config']['ckpt'])
    else:
        config['autoencoder_config']['ckpt'] = config['autoencoder_config']['ckpt']
    autoencoder.load_state_dict( torch.load(config['autoencoder_config']['ckpt'], map_location='cpu')['model_state_dict'] )
    autoencoder.cuda()
    autoencoder.eval()
    print('autoencoder size:')
    print_size(autoencoder)
    diffusion_model = LatentDiffusion(standard_diffusion_config, autoencoder=autoencoder)

    task = config['train_config']['task']
    trainset_config['eval_batch_size'] = args.batch_size
    save_dir = args.save_dir+trainset_config['dataset']

    if args.local_resampling:
        data = np.load(args.keypoint_file)
        keypoint_feature = torch.from_numpy(data['keypoint_feature']).float() # (B,N,F)
        keypoint_mask = torch.from_numpy(data['keypoint_mask']).float() # (B,N)
        keypoint = torch.from_numpy(data['points']).float() # (B,N,3)
        complete_x0 = torch.cat([keypoint, keypoint_feature], dim=2) # (B,N,3+F)
    else:
        complete_x0 = None
        keypoint_mask = None

    evaluate_per_rank(net, trainset_config, None, save_dir, task,
                    point_feature_dim=pointnet_config['in_fea_dim'], 
                    rank=0, world_size=1, ckpt_info='',
                    diffusion_model=diffusion_model, keypoint_dim=3,
                    test_external_keypoint=False, external_keypoint_file=args.keypoint_file,
                    split_points_and_normals=True, include_idx_to_save_name=not args.not_include_idx_to_save_name,
                    save_keypoint_feature=args.save_keypoint_feature,
                    local_resampling=args.local_resampling, complete_x0=complete_x0, keypoint_mask=keypoint_mask)

    