import os
import time
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import MyDataLoader
from data_utils.ema import EMAHelper
from shutil import copyfile
from data_utils.json_reader import read_json_file
from metrics_point_cloud.chamfer_and_f1 import calc_cd
from DDPM_keypoint.autoencoder import AutoEncoder


def train( args, dataset, root_directory, output_directory,
          tensorboard_directory, ckpt_iter, n_epochs, epochs_per_ckpt, iters_per_logging,
          learning_rate,eval_epoch):
    # generate experiment (local) path
    local_path = args.model_name
    config_file=args.config
    # Get shared output_directory ready
    output_directory = os.path.join(root_directory,dataset, local_path, output_directory)
    if args.rank == 0:
        os.makedirs(output_directory, exist_ok=True)
        #print("output directory is", output_directory, flush=True)

        config_file_copy_path = os.path.join(root_directory, local_path, os.path.split(config_file)[1])
        try:
            copyfile(config_file, config_file_copy_path)
        except:
            print('The two files are the same, no need to copy')
        print("Config file has been copied from %s to %s" % (config_file, config_file_copy_path), flush=True)

    net =AutoEncoder().cuda()

    net.train()
    ema_rate = [0.999, 0.9999]
    ema_helper_list = [EMAHelper(mu=rate) for rate in ema_rate]
    for ema_helper in ema_helper_list:
        ema_helper.register(net)
    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint model
    time0 = time.time()
    ckpt_iter = -1
    print('No valid checkpoint model found, start training from initialization.', flush=True)

    # load training data
    batch_size=trainset_config['batch_size']
    if (dataset).split('-')[0]=='spiral':
        dset_train = MyDataLoader.Dataset_for_shape_spiral(trainset_config, dataset, split='train', transform=True)
        trainloader = torch.utils.data.DataLoader(dset_train, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        dset_train = MyDataLoader.Dataset_for_para_optim(trainset_config,dataset,split='train',transform=True)
        trainloader = torch.utils.data.DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    print('Data loaded')

    # training
    loader_len = len(trainloader)
    n_iters = int(loader_len * n_epochs) # number of total training steps 
    iters_per_ckpt = int(loader_len * epochs_per_ckpt) # save a ckpt every iters_per_ckpt steps
    n_iter = ckpt_iter + 1 # starting iter number
    iters_eval_epoch=int(loader_len * eval_epoch)
    # we start evaluating the trained model at least after eval_start_epoch steps
    log_start_time = time.time() # used to compute how much time is consumed between 2 printing log

    # n_iter from 0 to n_iters if we train the model from sratch
    while n_iter < n_iters + 1:
        for data in trainloader:
            epoch_number = int((n_iter+1)/loader_len)
            # load data
            X = data['points_m'].cuda()  # of shape (npoints, 3), roughly in the range of -scale to scale
            keypoints = data['keypoints_m'].cuda()
            para = data['para_m'].cuda()
            para_shape = data['para_shape_m'].cuda()
            condition=torch.cat((para,para_shape),dim=1)
            condition=condition.unsqueeze(2)
            optimizer.zero_grad()
            loss = net.get_loss(keypoints,condition)
            loss.backward()
            optimizer.step()
            if ema_rate is not None :
                for ema_helper in ema_helper_list:
                    ema_helper.update(net)

            if n_iter % len(trainloader) == 0:
                print("iteration: {}  \tloss: {:.6f} \ttime: {:.2f}s".format(
                    n_iter, loss.item(), time.time() - log_start_time), flush=True)

            # save checkpoint
            if n_iter > 0 and (n_iter + 1) % iters_per_ckpt == 0:
                # save checkpoint
                checkpoint_name = args.model_name+'_pointnet_ckpt_{}.pkl'.format(n_iter)
                checkpoint_states = {'iter': n_iter,
                                     'model_state_dict': net.state_dict(),
                                     'optimizer_state_dict': optimizer.state_dict(),
                                     'training_time_seconds': int(time.time() - time0)}
                checkpoint_states['ema_state_list'] = [ema_helper.state_dict() for ema_helper in
                                                       ema_helper_list]
                torch.save(checkpoint_states, os.path.join(output_directory, checkpoint_name))
                print('model at iteration %s at epoch %d is saved' % (n_iter, epoch_number), flush=True)
            n_iter += 1



if __name__ == "__main__":
    # import pdb
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/config_DDPM_keypoint.json', help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0, help='rank of process for distributed')
    parser.add_argument('-g', '--model_name', type=str, default='dpm', help='name of group for distributed')
    parser.add_argument('-d', '--CUDA_VISIBLE_DEVICES', type=str, default="0",help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--split', type=str, default="train", help='')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    config = read_json_file(args.config)
    config_file_path = os.path.split(args.config)[0]
    global train_config
    train_config = config["train_"+args.model_name+"_config"]        # training parameters
    global dist_config
    dist_config = config["dist_config"]         # to initialize distributed training
    global pointnet_config
    global trainset_config
    trainset_config = config['dataset_config']
    global material_config
    material_config = config['material_property_config']

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
    train(args,**train_config)
