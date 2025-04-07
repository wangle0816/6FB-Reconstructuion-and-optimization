import os
import time
import argparse
import json
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import MyDataLoader
from util import find_max_epoch, print_size

from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor, broadcast_params

from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from models.pointwise_net import get_pointwise_net
from AFNO_3D.process_optimizer import para_optim
from models.autoencoder import PointAutoencoder

from shutil import copyfile
import copy

# from mesh_evaluation import evaluate_per_rank, gather_generated_results
from data_utils.json_reader import replace_list_with_string_in_a_dict, restore_string_to_list_in_a_dict, read_json_file, \
    autoencoder_read_config
from data_utils.ema import EMAHelper
import pickle
import pdb


def train(num_gpus, rank, dataset,config_file,  root_directory, output_directory,
          tensorboard_directory, ckpt_iter, n_epochs, epochs_per_ckpt, iters_per_logging,
          learning_rate, loss_type, conditioned_on_cloud,eval_per_ckpt,task,split_dataset_to_multi_gpus,
          eval_start_epoch=0,ema_rate=None):

    local_path =dataset

    # Create tensorboard logger.
    if rank == 0:
        tb = SummaryWriter(os.path.join(root_directory, local_path, tensorboard_directory))

    # Get shared output_directory ready
    output_directory = os.path.join(root_directory, local_path, output_directory)
    if rank == 0:
        os.makedirs(output_directory, exist_ok=True)
        print("output directory is", output_directory, flush=True)

        config_file_copy_path = os.path.join(root_directory, local_path, os.path.split(config_file)[1])
        try:
            copyfile(config_file, config_file_copy_path)
        except:
            print('The two files are the same, no need to copy')
        print("Config file has been copied from %s to %s" % (config_file, config_file_copy_path), flush=True)

    print('Data loaded')



    # build autoencoder and load the checkpoint
    autoencoder = PointAutoencoder(encoder_config, decoder_config_list,
                                   apply_kl_regularization=autoencoder_config['pointnet_config'].get(
                                       'apply_kl_regularization', False),
                                   kl_weight=autoencoder_config['pointnet_config'].get('kl_weight', 0))
    autoencoder.load_state_dict(
        torch.load(config['autoencoder_config']['ckpt'], map_location='cpu')['model_state_dict'])
    autoencoder.cuda()
    autoencoder.eval()
    print('autoencoder size:')
    print_size(autoencoder)
    print(autoencoder)
    # build  network
    train_split='physics'
    model_para_optim=para_optim(autoencoder=autoencoder,pnum=trainset_config['npoints'],**para_optim_config,**material_config,train_split=train_split).cuda()
    model_para_optim.train()
    print('model_para_optim size:')
    print_size(model_para_optim)

    # set ema
    if ema_rate is not None and rank == 0:
        assert isinstance(ema_rate, list)
        ema_helper_list = [EMAHelper(mu=rate) for rate in ema_rate]
        for ema_helper in ema_helper_list:
            ema_helper.register(model_para_optim)

    # optimizer
    optimizer = torch.optim.Adam(model_para_optim.parameters(), lr=learning_rate)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1000,gamma=0.8)
    # load checkpoint model
    time0 = time.time()
    _, num_ckpts = find_max_epoch(output_directory, 'pointnet_ckpt', return_num_ckpts=True)
    # num_ckpts is number of ckpts found in the output_directory
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory, 'pointnet_ckpt')
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, 'pointnet_ckpt_{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            model_para_optim.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if ema_rate is not None and rank == 0:
                for i in range(len(ema_helper_list)):
                    ema_helper_list[i].load_state_dict(checkpoint['ema_state_list'][i])
                    ema_helper_list[i].to(torch.device('cuda'))
                print('Ema helper has been loaded', flush=True)

            # record training time based on elapsed time
            time0 -= checkpoint['training_time_seconds']
            print('Model at iteration %s has been trained for %s seconds' % (
            ckpt_iter, checkpoint['training_time_seconds']))
            print('checkpoint model loaded successfully', flush=True)
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization.', flush=True)
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.', flush=True)

    # load training data
    batch_size = trainset_config['batch_size']
    dset_train = MyDataLoader.Dataset_for_para_optim(trainset_config,dataset,split='train',transform=True)
    trainloader = torch.utils.data.DataLoader(dset_train, batch_size=batch_size, shuffle=False, num_workers=8)
    dset_test = MyDataLoader.Dataset_for_para_optim(trainset_config, dataset,split='test',transform=True)
    testloader = torch.utils.data.DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=8)
    print('Data loaded')
    # print(net)
    # training
    loader_len = len(trainloader)
    n_iters = int(loader_len * n_epochs)  # number of total training steps
    iters_per_ckpt = int(loader_len * epochs_per_ckpt)  # save a ckpt every iters_per_ckpt steps
    n_iter = ckpt_iter + 1  # starting iter number
    eval_start_iter = eval_start_epoch * loader_len - 1
    min_result_mse=1.0

    # n_iter from 0 to n_iters if we train the model from sratch
    while n_iter < n_iters + 1:
        for data in trainloader:
            epoch_number = int((n_iter + 1) / loader_len)
            # load data
            X_m = data['points_m'].cuda()  # of shape (npoints, 3), roughly in the range of -scale to scale
            X_d = data['points_d'].cuda()  # of shape (npoints, 3), roughly in the range of -scale to scale
            normals_m  = data['normals_m'].cuda()  # of shape (npoints, 3), the normals are normalized
            normals_d = data['normals_d'].cuda()  # of shape (npoints, 3), the normals are normalized
            label = data['label'].cuda()
            keypoints_m  = data['keypoints_m'].cuda()
            keypoints_d = data['keypoints_d'].cuda()
            stress_m=(data['stress_m'].cuda())
            stress_d = (data['stress_d'].cuda())/1e3
            para_m=data['para_m'].cuda()
            gt=data['para_d'].cuda()

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
            optimizer.zero_grad()
            loss_mse,loss_mae,loss_matrix,_,output_stress_incre_b=model_para_optim(X_m,X_d,keypoints_m,keypoints_d,stress_m,label,para_m,gt)
            if train_split=='learning_physics':
                loss=loss_mse+loss_matrix
            else:
                loss_stress=F.mse_loss(output_stress_incre_b, stress_d/1e3, reduction='mean')
                loss = loss_mse+loss_stress

            reduced_loss = loss.item()
            loss.backward()
            optimizer.step()

            if ema_rate is not None and rank == 0:
                for ema_helper in ema_helper_list:
                    ema_helper.update(model_para_optim)

            # output to log
            if n_iter % len(trainloader) == 0:
                log_start_time = time.time()
                print("iteration: {} \treduced loss: {:.6f} \tloss: {:.6f} \ttime: {:.2f}s".format(n_iter, reduced_loss, loss.item(), time.time() - log_start_time), flush=True)
                tb.add_scalar("Train-Loss", (loss_mse), n_iter)
            if n_iter % len(trainloader) == 0:
            # save checkpoint
            #if n_iter > 0 and (n_iter + 1) % iters_per_ckpt == 0:
                num_ckpts = num_ckpts + 1
                results_mse = 0
                results_mae = 0
                results_mse_i = torch.zeros(7)
                results_mae_i = torch.zeros(7)
                loss_mse_i = torch.zeros(7)
                loss_mae_i = torch.zeros(7)
                with torch.no_grad():
                    for data in testloader:
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
                        loss_mse_test, loss_mae,_, output,_ = model_para_optim(X_m, X_d, keypoints_m, keypoints_d, stress_m,
                                                                      label, para_m, gt)
                        for i in range(gt.shape[-1]):
                            loss_mse_i[i] = F.mse_loss(output[:, i], gt[:, i], reduction='mean')
                            loss_mae_i[i] = torch.mean(torch.abs(output[:, i] - gt[:, i]))
                        #print('loss_mse:', loss_mse, 'lose_mae:', loss_mae)
                        results_mse_i += loss_mse_i / len(testloader)
                        results_mae_i += loss_mae_i / len(testloader)
                        results_mse += loss_mse_test / len(testloader)
                        results_mae += loss_mae / len(testloader)
                #print('results_mse:', results_mse, "results_mae:", results_mae)
                #print('results_mse_i:', results_mse_i, "results_mae_i:", results_mae_i)
                tb.add_scalar("Test-Loss", (loss_mse_test), n_iter)
                # save checkpoint

                if results_mse<min_result_mse and n_iter>2000:
                    min_result_mse=results_mse
                    checkpoint_name = 'pointnet_ckpt_0.pkl'
                    checkpoint_states = {'iter': n_iter,
                                         'model_state_dict': model_para_optim.state_dict(),
                                         'optimizer_state_dict': optimizer.state_dict(),
                                         'training_time_seconds': int(time.time() - time0)}
                    if not ema_rate is None:
                        checkpoint_states['ema_state_list'] = [ema_helper.state_dict() for ema_helper in
                                                               ema_helper_list]
                    torch.save(checkpoint_states, os.path.join(output_directory, checkpoint_name))
                    print('model at iteration %s at epoch %d is saved' % (n_iter, epoch_number), flush=True)

            n_iter += 1
            scheduler.step()

if __name__ == "__main__":
    m_seed = 1
    torch.manual_seed(m_seed)
    torch.cuda.manual_seed(m_seed)
    # import pdb
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/config_para_optim.json',
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0, help='rank of process for distributed')
    parser.add_argument('-d', '--dataset', type=str, default='6FB-3D', help='')
    parser.add_argument('-g', '--group_name', type=str, default='', help='name of group for distributed')
    parser.add_argument('--dist_url', type=str, default='', help='distributed training url')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
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
    autoencoder_config_file = config['autoencoder_config']['config_file']
    global autoencoder_config
    autoencoder_config = read_json_file(autoencoder_config_file)
    autoencoder_config_file_path = os.path.split(autoencoder_config_file)[0]
    global encoder_config
    global decoder_config_list
    encoder_config, decoder_config_list = autoencoder_read_config(autoencoder_config_file_path, autoencoder_config)
    global material_config
    material_config=config['material_property_config']

    print('The autoencoder configuration is:')
    print(json.dumps(replace_list_with_string_in_a_dict(copy.deepcopy(autoencoder_config)), indent=4))
    os.environ['CUDA_VISIBLE_DEVICES'] = config["dist_config"]["CUDA_VISIBLE_DEVICES"]
    print('Visible GPUs are', os.environ['CUDA_VISIBLE_DEVICES'], flush=True)
    num_gpus = torch.cuda.device_count()
    print('%d GPUs are available' % num_gpus, flush=True)
    if num_gpus > 1:
        assert args.group_name != ''
    else:
        assert args.rank == 0

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train(num_gpus, args.rank,args.dataset,args.config, **train_config)
