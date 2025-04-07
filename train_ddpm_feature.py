import os
import time
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from thop import profile
from torch.utils.tensorboard import SummaryWriter

import MyDataLoader
from dataset import get_dataloader
from util import find_max_epoch, print_size
from util import training_loss, calc_diffusion_hyperparams

from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor, broadcast_params
from diffusion_utils.diffusion import LatentDiffusion

from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from models.point_upsample_module import point_upsample
from models.pointwise_net import get_pointwise_net

from models.autoencoder import PointAutoencoder
from data_utils.points_sampling import sample_keypoints


from shutil import copyfile
import copy

#from mesh_evaluation import evaluate_per_rank, gather_generated_results
from data_utils.json_reader import replace_list_with_string_in_a_dict, restore_string_to_list_in_a_dict, read_json_file, autoencoder_read_config
from data_utils.ema import EMAHelper
import pickle
import pdb

def train(num_gpus, config_file, rank, group_name, dataset, root_directory, output_directory, 
          tensorboard_directory, ckpt_iter, n_epochs, epochs_per_ckpt, iters_per_logging,
          learning_rate, loss_type, conditioned_on_cloud,
          eval_start_epoch = 0, eval_per_ckpt = 1, task='latent_generation', split_dataset_to_multi_gpus=False, ema_rate=None):
    """
    Train the PointNet2SemSegSSG model on the 3D dataset

    Parameters:
    num_gpus, rank, group_name:     parameters for distributed training
    config_file:                    path to the config file
    output_directory (str):         save model checkpoints to this path
    tensorboard_directory (str):    save tensorboard events to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automitically selects the maximum iteration if 'max' is selected
    n_epochs (int):                 number of epochs to train
    epochs_per_ckpt (int):          number of epochs to save checkpoint
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate
    """
    assert task in ['latent_generation', 'latent_keypoint_conditional_generation']
    # generate experiment (local) path
    local_path = "T{}_betaT{}".format(standard_diffusion_config['num_diffusion_timesteps'], standard_diffusion_config['beta_end'])
    local_path = local_path + '_' + pointnet_config['model_name']+ '_' +dataset
        
    # Create tensorboard logger.
    if rank == 0:
        tb = SummaryWriter(os.path.join(root_directory, local_path, tensorboard_directory))

    # distributed running initialization
    if num_gpus > 1:
        dist_config.pop('CUDA_VISIBLE_DEVICES', None)
        init_distributed(rank, num_gpus, group_name, **dist_config)

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

    net = PointNet2CloudCondition(pointnet_config).cuda()
    net.train()
    print('latent ddpm size:')
    print_size(net)

    # build autoencoder and load the checkpoint
    autoencoder = PointAutoencoder(encoder_config, decoder_config_list, 
                apply_kl_regularization=autoencoder_config['pointnet_config'].get('apply_kl_regularization', False),
                kl_weight=autoencoder_config['pointnet_config'].get('kl_weight', 0))
    autoencoder.load_state_dict( torch.load(config['autoencoder_config']['ckpt'], map_location='cpu')['model_state_dict'] )
    autoencoder.cuda()
    autoencoder.eval()
    print('autoencoder size:')
    print_size(autoencoder)
    print(autoencoder)
    diffusion_model = LatentDiffusion(standard_diffusion_config, autoencoder=autoencoder)

    # apply gradient all reduce
    if num_gpus > 1:
        net = apply_gradient_allreduce(net)

    # set ema
    if ema_rate is not None and rank == 0:
        assert isinstance(ema_rate, list)
        ema_helper_list = [EMAHelper(mu=rate) for rate in ema_rate]
        for ema_helper in ema_helper_list:
            ema_helper.register(net)


    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint model
    time0 = time.time()
    _, num_ckpts = find_max_epoch(output_directory, 'pointnet_ckpt', return_num_ckpts=True)
    # num_ckpts is number of ckpts found in the output_directory
    print('ckpt_iter',ckpt_iter)
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory, 'pointnet_ckpt')
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, 'pointnet_ckpt_{}.pkl'.format(ckpt_iter))
            print('model_path',model_path)
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if ema_rate is not None and rank==0:
                for i in range(len(ema_helper_list)):
                    ema_helper_list[i].load_state_dict(checkpoint['ema_state_list'][i])
                    ema_helper_list[i].to(torch.device('cuda'))
                print('Ema helper has been loaded', flush=True)

            # record training time based on elapsed time
            time0 -= checkpoint['training_time_seconds']
            print('Model at iteration %s has been trained for %s seconds' % (ckpt_iter, checkpoint['training_time_seconds']))
            print('checkpoint model loaded successfully', flush=True)
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization.', flush=True)
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.', flush=True)

    # load training data
    batch_size = trainset_config['batch_size']
    if (dataset).split('-')[0] == 'spiral':
        dset_train = MyDataLoader.Dataset_for_shape_spiral(trainset_config, dataset, split='train', transform=True)
        trainloader = torch.utils.data.DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    else:
        dset_train =MyDataLoader.Dataset_surface_axis(trainset_config,dataset,split='train',transform=True)
        trainloader = torch.utils.data.DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    print('Data loaded')
    # print(net)
    # training
    loader_len = len(trainloader)
    n_iters = int(loader_len * n_epochs) # number of total training steps 
    iters_per_ckpt = int(loader_len * epochs_per_ckpt) # save a ckpt every iters_per_ckpt steps
    n_iter = ckpt_iter + 1 # starting iter number
    eval_start_iter = eval_start_epoch *  loader_len - 1 
    # we start evaluating the trained model at least after eval_start_epoch steps

    log_start_time = time.time() # used to compute how much time is consumed between 2 printing log

    # n_iter from 0 to n_iters if we train the model from sratch
    while n_iter < n_iters + 1:
        for data in trainloader: 
            epoch_number = int((n_iter+1)/loader_len)
            # load data
            X = data['points'].cuda() # of shape (npoints, 3), roughly in the range of -scale to scale
            normals = data['normals'].cuda() # of shape (npoints, 3), the normals are normalized
            label = data['label'].cuda()
            keypoints = data['keypoints'].cuda()
            keypoint_noise_magnitude = trainset_config.get('keypoint_noise_magnitude', 0)
            if keypoint_noise_magnitude > 0:
                keypoints = keypoints + keypoint_noise_magnitude * torch.randn_like(keypoints)
            else:
                keypoints = keypoints
            if trainset_config.get('include_normals', True):
                X = torch.cat([X, normals], dim=2)
            condition = None
            
            # back-propagation
            optimizer.zero_grad()
            
            loss_batch = diffusion_model.train_loss(net, X, keypoints, label)

            loss = loss_batch.mean()
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()
            loss.backward()
            optimizer.step()

            if ema_rate is not None and rank==0:
                for ema_helper in ema_helper_list:
                    ema_helper.update(net)

            # output to log
            if n_iter % iters_per_logging == 0:
                print("iteration: {} \treduced loss: {:.6f} \tloss: {:.6f} \ttime: {:.2f}s".format(
                    n_iter, reduced_loss, loss.item(), time.time()-log_start_time), flush=True)
                log_start_time = time.time()
                if rank == 0:
                    tb.add_scalar("Log-Train-Loss", torch.log(loss).item(), n_iter)
                    tb.add_scalar("Log-Train-Reduced-Loss", np.log(reduced_loss), n_iter)
            
            # save checkpoint
            if n_iter > 0 and (n_iter+1) % iters_per_ckpt == 0:
                num_ckpts = num_ckpts + 1
                # save checkpoint
                if rank == 0:
                    checkpoint_name = 'pointnet_ckpt_{}.pkl'.format(n_iter)
                    checkpoint_states = {'iter': n_iter,
                                'model_state_dict': net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'training_time_seconds': int(time.time()-time0)}
                    if not ema_rate is None:
                        checkpoint_states['ema_state_list'] = [ema_helper.state_dict() for ema_helper in ema_helper_list]
                    torch.save(checkpoint_states, os.path.join(output_directory, checkpoint_name))
                    print('model at iteration %s at epoch %d is saved' % (n_iter, epoch_number), flush=True)
            n_iter += 1


if __name__ == "__main__":
    # import pdb
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/config_DDPM_latent.json', help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0, help='rank of process for distributed')
    # parser.add_argument('-d', '--device', type=int, default=0, help='cuda gpu index for training')
    parser.add_argument('-g', '--group_name', type=str, default='', help='name of group for distributed')
    parser.add_argument('--dist_url', type=str, default='', help='distributed training url')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    global config
    config = read_json_file(args.config)
    print('The configuration is:')
    print(json.dumps(replace_list_with_string_in_a_dict(copy.deepcopy(config)), indent=4))
    
    global train_config
    train_config = config["train_config"]        # training parameters
    global dist_config
    dist_config = config["dist_config"]         # to initialize distributed training
    if len(args.dist_url) > 0:
        dist_config['dist_url'] = args.dist_url
    global pointnet_config
    pointnet_config = config["pointnet_config"]     # to define pointnet
    # global diffusion_config
    # diffusion_config = config["diffusion_config"]    # basic hyperparameters
    
    global trainset_config
    trainset_config = config['dataset_config']

    # global diffusion_hyperparams 
    # diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters

    global standard_diffusion_config
    standard_diffusion_config = config['standard_diffusion_config']

    # read autoencoder configs
    autoencoder_config_file = config['autoencoder_config']['config_file']
    global autoencoder_config
    autoencoder_config = read_json_file(autoencoder_config_file)
    autoencoder_config_file_path = os.path.split(autoencoder_config_file)[0]
    global encoder_config
    global decoder_config_list
    encoder_config, decoder_config_list = autoencoder_read_config(autoencoder_config_file_path, autoencoder_config)
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
    train(num_gpus, args.config, args.rank, args.group_name,trainset_config['dataset'], **train_config)
