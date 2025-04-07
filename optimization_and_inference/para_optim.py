import os
import argparse
import json
import time

import numpy as np
import torch.nn.functional as F
import torch
from thop import profile
from torch.utils.tensorboard import SummaryWriter
import MyDataLoader
from AFNO_3D.process_optimizer import para_optim
from models.autoencoder import PointAutoencoder
import copy
from data_utils.json_reader import replace_list_with_string_in_a_dict, restore_string_to_list_in_a_dict, read_json_file, \
    autoencoder_read_config
if __name__ == "__main__":
    m_seed=1
    torch.manual_seed(m_seed)
    torch.cuda.manual_seed(m_seed)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/config_para_optim.json',
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0, help='rank of process for distributed')
    parser.add_argument('-d', '--dataset', type=str, default='6FB-3D', help='')
    parser.add_argument('-g', '--group_name', type=str, default='', help='name of group for distributed')
    parser.add_argument('--dist_url', type=str, default='', help='distributed training url')
    args = parser.parse_args()
    dataset=args.dataset
    global config
    config = read_json_file(args.config)
    print('The configuration is:')
    print(json.dumps(replace_list_with_string_in_a_dict(copy.deepcopy(config)), indent=4))

    global train_config
    train_config = config["train_config"]  # training parameters
    global dist_config
    dist_config = config["dist_config"]  # to initialize distributed training
    if len(args.dist_url) > 0:
        dist_config['dist_url'] = args.dist_url
    global pointnet_config
    pointnet_config = config["pointnet_config"]  # to define pointnet
    # global diffusion_config
    # diffusion_config = config["diffusion_config"]    # basic hyperparameters

    global trainset_config
    trainset_config = config['dataset_config']

    # global diffusion_hyperparams
    # diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters

    global para_optim_config
    para_optim_config = config['para_optim_config']

    # read autoencoder configs
    autoencoder_config_file ='../' + config['autoencoder_config']['config_file']
    global autoencoder_config
    autoencoder_config = read_json_file(autoencoder_config_file)
    autoencoder_config_file_path = os.path.split(autoencoder_config_file)[0]
    global encoder_config
    global decoder_config_list
    encoder_config, decoder_config_list = autoencoder_read_config(autoencoder_config_file_path, autoencoder_config)
    global material_config
    material_config=config['material_property_config']

    os.environ['CUDA_VISIBLE_DEVICES'] = config["dist_config"]["CUDA_VISIBLE_DEVICES"]
    print('Visible GPUs are', os.environ['CUDA_VISIBLE_DEVICES'], flush=True)
    num_gpus = torch.cuda.device_count()
    print('%d GPUs are available' % num_gpus, flush=True)
    if num_gpus > 1:
        assert args.group_name != ''
    else:
        assert args.rank == 0
    torch.cuda.synchronize()

    autoencoder = PointAutoencoder(encoder_config, decoder_config_list,
                                   apply_kl_regularization=autoencoder_config['pointnet_config'].get(
                                       'apply_kl_regularization', False),
                                   kl_weight=autoencoder_config['pointnet_config'].get('kl_weight', 0))
    autoencoder.load_state_dict(
        torch.load('../' +config['autoencoder_config']['ckpt'], map_location='cpu')['model_state_dict'])
    autoencoder.cuda()
    autoencoder.eval()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    output_directory = os.path.join(train_config['root_directory'], dataset, train_config["output_directory"])
    model_path = os.path.join(output_directory, 'pointnet_ckpt_0.pkl')
    print('model_path',model_path)
    checkpoint = torch.load('../' +model_path, map_location='cpu')
    train_split = 'physics'
    model_para_optim = para_optim(autoencoder=autoencoder, pnum=trainset_config['npoints'], **para_optim_config,
                                  **material_config,train_split = 'physics').cuda()
    model_para_optim.load_state_dict(checkpoint['model_state_dict'])
    model_para_optim.eval()
    batch_size = 4
    if (dataset).split('-')[0]=='exp':
        dset_test = MyDataLoader.Dataset_for_para_optim_exp(trainset_config, dataset, split='test', transform=True)
        testloader = torch.utils.data.DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        dset_test = MyDataLoader.Dataset_for_para_optim(trainset_config, dataset, split='test', transform=True)
        testloader = torch.utils.data.DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=0)
    results_mse=0
    results_mae=0
    infer_time=0
    results_mse_i=torch.zeros(7)
    results_mae_i = torch.zeros( 7)
    loss_mse_i = torch.zeros(7)
    loss_mae_i = torch.zeros(7)
    output_stress_incre=[]
    with torch.no_grad():
        for iter,data in enumerate(testloader):
            # load data
            X_m = data['points_m'].cuda()  # of shape (npoints, 3), roughly in the range of -scale to scale
            X_d = data['points_d'].cuda()  # of shape (npoints, 3), roughly in the range of -scale to scale
            normals_m = data['normals_m'].cuda()  # of shape (npoints, 3), the normals are normalized
            normals_d = data['normals_d'].cuda()  # of shape (npoints, 3), the normals are normalized
            label = data['label'].cuda()
            keypoints_m = data['keypoints_m'].cuda()
            keypoints_d = data['keypoints_d'].cuda()
            stress_m = data['stress_m'].cuda()
            para_m = data['para_m'].cuda()
            gt = data['para_d'].cuda()

            keypoint_noise_magnitude = trainset_config.get('keypoint_noise_magnitude', 0)
            if keypoint_noise_magnitude > 0:
                keypoints_m = keypoints_m + keypoint_noise_magnitude * torch.randn_like(keypoints_m)
                keypoints_d = keypoints_d + keypoint_noise_magnitude * torch.randn_like(keypoints_d)
            else:
                keypoints_m = keypoints_m
                keypoints_d = keypoints_d
            if trainset_config.get('include_normals', True):
                X_m = torch.cat([X_m, normals_m], dim=2)
                X_d = torch.cat([X_d, normals_d], dim=2)
            # back-propagation
            loss_mse,loss_mae,_,output,output_stress_incre_b,infer_time_i = model_para_optim(X_m, X_d, keypoints_m, keypoints_d, stress_m, label, para_m, gt)
            macs, params = profile(model_para_optim,
                                   inputs=(X_m, X_d, keypoints_m, keypoints_d, stress_m, label, para_m, gt,))
            print(macs, params)
            para_min_max = np.array(trainset_config['para_min_max_' + dataset], dtype=np.float32)
            #print(para_min_max[:, 1]*np.array(output[0].cpu())+para_min_max[:, 0])
            for i in range(gt.shape[-1]):
                loss_mse_i[i] = F.mse_loss(output[:,i], gt[:,i], reduction='mean')
                loss_mae_i[i] = torch.mean(torch.abs(output[:,i] - gt[:,i]))
            print('loss_mse:',loss_mse,'lose_mae:',loss_mae)
            infer_time+=infer_time_i/len(testloader)
            results_mse_i+=loss_mse_i/len(testloader)
            results_mae_i += loss_mae_i / len(testloader)
            results_mse+=loss_mse/len(testloader)
            results_mae += loss_mae / len(testloader)
            output_stress_incre.append(np.array((torch.cat((X_d[:,:,:3],output_stress_incre_b),dim=2)).cpu()))
    output_stress_incre=np.concatenate(output_stress_incre, axis=0)
    print(output_stress_incre.shape)
    for i in range(output_stress_incre.shape[0]):
        np.savetxt('../sampling_and_inference/visual_incre_stress/'+str(i)+'.txt',output_stress_incre[i],"%f %f %f %f %f %f %f %f %f")
    print('infer_time:', infer_time)
    print('results_mse:',results_mse)
    print("results_mae:", results_mae)
    print('results_mse_i:', results_mse_i)
    print("results_mae_i:", results_mae_i)
