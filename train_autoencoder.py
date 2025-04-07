import os
import time
import argparse
import numpy as np
import torch
from thop import profile
from torch.utils.tensorboard import SummaryWriter
import MyDataLoader
from util import find_max_epoch, print_size
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from models.autoencoder import PointAutoencoder
from shutil import copyfile
from data_utils.json_reader import read_json_file, autoencoder_read_config

def train(num_gpus, config_file, rank, group_name, dataset, root_directory, output_directory, 
          tensorboard_directory, ckpt_iter, n_epochs, epochs_per_ckpt, iters_per_logging,
          learning_rate, loss_type, conditioned_on_cloud,eval_epoch, task='generation', split_dataset_to_multi_gpus=False):
    assert task in ['autoencode']
    # generate experiment (local) path
    local_path =pointnet_config['model_name']+dataset
        
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
        #print("output directory is", output_directory, flush=True)

        config_file_copy_path = os.path.join(root_directory, local_path, os.path.split(config_file)[1])
        try:
            copyfile(config_file, config_file_copy_path)
        except:
            print('The two files are the same, no need to copy')
        print("Config file has been copied from %s to %s" % (config_file, config_file_copy_path), flush=True)

    # load network
    net = PointAutoencoder(encoder_config, decoder_config_list, 
                apply_kl_regularization=pointnet_config.get('apply_kl_regularization',False),
                kl_weight=pointnet_config.get('kl_weight', 0),
                feature_weight=pointnet_config.get('feature_weight', None)).cuda()

    net.train()
    print_size(net)

    # apply gradient all reduce
    if num_gpus > 1:
        net = apply_gradient_allreduce(net)

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

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
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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

    #print(net)

    # load training data
    batch_size=trainset_config['batch_size']
    if (dataset).split('-')[0] == 'spiral':
        dset_train = MyDataLoader.Dataset_for_shape_spiral(trainset_config, dataset, split='train', transform=True)
        trainloader = torch.utils.data.DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=8)
        dset_test = MyDataLoader.Dataset_for_shape_spiral(trainset_config, dataset, split='test', transform=True)
        testloader = torch.utils.data.DataLoader(dset_test, batch_size=1, shuffle=False, num_workers=0)
    else:
        dset_train = MyDataLoader.Dataset_surface_axis(trainset_config,dataset,split='train',transform=True)
        trainloader = torch.utils.data.DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=8)
        dset_test = MyDataLoader.Dataset_surface_axis(trainset_config,dataset,split='test', transform=True)
        testloader = torch.utils.data.DataLoader(dset_test, batch_size=1, shuffle=False, num_workers=0)
    print('Data loaded')

    # training
    loader_len = len(trainloader)
    n_iters = int(loader_len * n_epochs)
    iters_per_ckpt = int(loader_len * epochs_per_ckpt)
    n_iter = ckpt_iter + 1
    iters_eval_epoch=int(loader_len * eval_epoch)
    log_start_time = time.time()

    # n_iter from 0 to n_iters if we train the model from sratch
    while n_iter < n_iters + 1:
        for data in trainloader: 
            epoch_number = int((n_iter+1)/loader_len)
            # load data
            X = data['points'].cuda()  # of shape (npoints, 3), roughly in the range of -scale to scale
            normals = data['normals'].cuda()  # of shape (npoints, 3), the normals are normalized
            label = data['label'].cuda()
            keypoints=data['keypoints'].cuda()
            normals = normals / torch.norm(normals, p=2, dim=2, keepdim=True)
            keypoint_noise_magnitude = trainset_config.get('keypoint_noise_magnitude', 0)#噪声规模
            if keypoint_noise_magnitude > 0:
                keypoints = keypoints + keypoint_noise_magnitude * torch.randn_like(keypoints)#加噪，数据增强
            X = torch.cat([X, normals], dim=2) #B,N,6
            print('X', X.shape)
            macs, params = profile(net, inputs=(X, keypoints, None, label, 'cd_p',))
            print('macs',macs, params)
            l_xyz, loss_list = net(X, keypoints, ts=None, label=label, loss_type='cd_p')
            loss = 0
            for loss_dict in loss_list:
                loss = loss + loss_dict['training_loss'].mean()
            # back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # pdb.set_trace()

            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()

            # output to log
            if n_iter % iters_per_logging == 0:
                print("iteration: {} \treduced loss: {:.6f} \tloss: {:.6f} \ttime: {:.2f}s".format(
                    n_iter, reduced_loss, loss.item(), time.time()-log_start_time), flush=True)
                for key in loss_list[0].keys():
                    values = [loss_dict[key].mean().item() for loss_dict in loss_list]
                    print(key, values)
                log_start_time = time.time()
                if rank == 0:
                    tb.add_scalar("Log-Train-Loss", torch.log(loss).item(), n_iter)
                    tb.add_scalar("Log-Train-Reduced-Loss", np.log(reduced_loss), n_iter)
                    #tensorboard 2.15.1 at http://localhost:8008/
            # save checkpoint
            if n_iter > 0 and (n_iter+1) % iters_per_ckpt == 0:
                num_ckpts = num_ckpts + 1
                # save checkpoint
                if rank == 0:
                    checkpoint_name = 'pointnet_ckpt_{}.pkl'.format(n_iter)
                    torch.save({'iter': n_iter,
                                'model_state_dict': net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'training_time_seconds': int(time.time()-time0)}, 
                                os.path.join(output_directory, checkpoint_name))
                    print('model at iteration %s at epoch %d is saved' % (n_iter, epoch_number), flush=True)

            #evaluation
            if n_iter<0 and (n_iter) % iters_eval_epoch==0:
                net.eval()
                eval_results = 0
                with torch.no_grad():
                    for data in testloader:
                        X = data['points'].cuda()  # of shape (npoints, 3), roughly in the range of -scale to scale
                        normals = data['normals'].cuda()  # of shape (npoints, 3), the normals are normalized
                        label = data['label'].cuda()
                        keypoints = data['keypoints'].cuda()
                        normals = normals / torch.norm(normals, p=2, dim=2, keepdim=True)
                        keypoint_noise_magnitude = trainset_config.get('keypoint_noise_magnitude', 0)  # 噪声规模
                        if keypoint_noise_magnitude > 0:
                            keypoints = keypoints + keypoint_noise_magnitude * torch.randn_like(keypoints)  # 加噪，数据增强
                        X = torch.cat([X, normals], dim=2)  # B,N,6

                        l_xyz, loss_list = net(X, keypoints, ts=None, label=label, loss_type='cd_p')
                        # for i in range(len(l_xyz)): print('l_xyz',l_xyz[i].shape)
                        #np.savetxt('testpoint',np.array(l_xyz[-2][0][:,:3].cpu().detach()),'%f %f %f')
                        loss = 0
                        for loss_dict in loss_list:
                            loss = loss + loss_dict['training_loss'].mean()
                        print("loss: {:.6f}".format(loss.item()))
                        eval_results+=loss/(len(testloader))
                print("results: {:.6f}".format(eval_results.item()))
            n_iter += 1




if __name__ == "__main__":
    # import pdb
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/config_autoencoder.json', help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0, help='rank of process for distributed')
    # parser.add_argument('-d', '--device', type=int, default=0, help='cuda gpu index for training')
    parser.add_argument('-g', '--group_name', type=str, default='', help='name of group for distributed')
    parser.add_argument('--dist_url', type=str, default='', help='distributed training url')
    parser.add_argument('-d', '--CUDA_VISIBLE_DEVICES', type=str, default="0",help='CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    config = read_json_file(args.config)
    config_file_path = os.path.split(args.config)[0]
    global encoder_config
    global decoder_config_list
    encoder_config, decoder_config_list = autoencoder_read_config(config_file_path, config)
    #print(decoder_config_list)
    #print('The configuration is:')
    #print(json.dumps(replace_list_with_string_in_a_dict(copy.deepcopy(config)), indent=4))
    # global gen_config
    # gen_config = config["gen_config"]
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
    net = PointAutoencoder(encoder_config, decoder_config_list,
                           apply_kl_regularization=pointnet_config.get('apply_kl_regularization', False),
                           kl_weight=pointnet_config.get('kl_weight', 0),
                           feature_weight=pointnet_config.get('feature_weight', None)).cuda()

    train(num_gpus, args.config, args.rank, args.group_name,trainset_config['dataset'], **train_config)




