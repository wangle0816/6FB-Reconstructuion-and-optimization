import argparse
import MyDataLoader
from DDPM_keypoint.autoencoder import AutoEncoder
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

def evaluation(args,train_config,dataset,net):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    local_path =args.model_name
    output_directory = os.path.join("../"+train_config['root_directory'],dataset,  local_path, train_config["output_directory"])
    model_path = os.path.join(output_directory, args.model_name+'_pointnet_ckpt_{}.pkl'.format(train_config['ckpt_iter']))
    print('model_path', model_path)
    net_dpm =AutoEncoder().cuda()
    checkpoint = torch.load( model_path, map_location='cpu')
    net_dpm.load_state_dict(checkpoint['model_state_dict'])
    net_dpm.eval()
    batch_size =4
    point_feature_dim=pointnet_config['in_fea_dim']
    if dataset.split('-')[0]=='exp':
        dset_test = MyDataLoader.Dataset_for_para_optim_exp(trainset_config,dataset,split='test',transform=True)
        testloader = torch.utils.data.DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=0)
    elif (dataset).split('-')[0]=='spiral':
        dset_test = MyDataLoader.Dataset_for_shape_spiral(trainset_config, dataset, split='test', transform=True)
        testloader = torch.utils.data.DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        if args.keypoints_type=='axis_physics':
            para_transform=False
        else:
            para_transform = True
        dset_test = MyDataLoader.Dataset_for_para_optim(trainset_config, dataset, split='test', transform=True,para_transform=para_transform)
        testloader = torch.utils.data.DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=0)
    results_cd_keypoints = 0
    results_cd_points = 0
    save_dir='generated_point_cloud_and_mesh/'+args.keypoints_type+'/'+str(trainset_config['num_keypoints'])+'_keypoints'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for iter,data in enumerate(testloader):
            torch.cuda.synchronize()
            # load data
            X = data['points_m'].cuda()  # of shape (npoints, 3), roughly in the range of -scale to scale
            gt_keypoints = data['keypoints_m'].cuda()
            para = data['para_m'].cuda()
            para_shape = data['para_shape_m'].cuda()
            label = data['label'].cuda()
            length=data['length'].cuda()/1000.0
            print('args.keypoints_type',args.keypoints_type)
            if args.keypoints_type in ['surface_ddpm' , 'axis_ddpm']:
                condition = torch.cat((para, para_shape), dim=1)
                condition = condition.unsqueeze(2)
                code = net_dpm.encode(condition)
                keypoints = net_dpm.decode(code, gt_keypoints.size(1), flexibility=0.0)
                print('*******************')
            if args.keypoints_type  in [ 'surface_gt' , 'axis_gt']:
                keypoints=gt_keypoints
                print('###################')
            if args.keypoints_type == 'axis_physics':#Coordinate is the same as in the simulaion
                D = para_shape[:, 0]
                R = para_shape[:,1]
                P = para_shape[:,2]
                A = para[:,3]/1000.0
                kA = para[:,0]
                L=torch.sqrt((2*torch.pi*R)**2+P**2)
                alpha=length/R
                source_keypoint_num=64
                alpha_data=torch.zeros(gt_keypoints.shape[0],source_keypoint_num-1).cuda()
                for i,alphai in enumerate(alpha):
                    alpha_data[i]=torch.linspace(-alphai, 0, steps=source_keypoint_num - 1)
                R=R.unsqueeze(-1)
                axis_points_x = -R*(torch.cos(alpha_data)-1)
                print('axis_points_x',axis_points_x)
                axis_points_y=torch.zeros(gt_keypoints.shape[0],source_keypoint_num-1).cuda()
                for i,yi in enumerate(P*length/L):
                    axis_points_y[i] = torch.linspace(yi,0,source_keypoint_num - 1)
                dist = (A*(kA-1)).unsqueeze(-1)
                axis_points_z = R * (torch.sin(alpha_data))+dist
                keypoints=torch.cat((axis_points_x.unsqueeze(2),axis_points_y.unsqueeze(2),axis_points_z.unsqueeze(2)),dim=2)
                strat_point=torch.zeros(gt_keypoints.shape[0],1,3).cuda()
                strat_point[:,0,-1]=D
                keypoints=torch.cat((keypoints,strat_point),dim=1)
                keypoint_normalized=torch.zeros_like(keypoints)
                for i in range(keypoints.shape[0]):
                    pc=np.array(keypoints[i].cpu())
                    centroid = np.mean(pc, axis=0)
                    pc = pc - centroid
                    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
                    pc = pc / m
                    keypoint_normalized[i]=torch.tensor(pc).cuda()
                #keypoints=keypoint_normalized
                keypoints, _ = sample_keypoints(keypoint_normalized, K=gt_keypoints.shape[1])
                '''
                for i in range(keypoints.shape[0]):
                    fig=plt.figure()
                    ax=fig.add_subplot(111,projection='3d')
                    ax.scatter(np.array(keypoints[i][:,0].cpu()),np.array(keypoints[i][:,1].cpu()),np.array(keypoints[i][:,2].cpu()),c='b')
                    ax.scatter(np.array(gt_keypoints[i][:,0].cpu()), np.array(gt_keypoints[i][:,1].cpu()), np.array(gt_keypoints[i][:,2].cpu()), c='r')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    plt.show()
                '''
            generated_data, _, _ = diffusion_model.denoise_and_reconstruct(
                n=keypoints.shape[0], model=net, keypoint_dim=3,shape=(trainset_config['num_keypoints'], 3 + point_feature_dim), label=label,
                keypoint=keypoints,
                return_keypoint_feature=True,
                local_resampling=False, complete_x0=None, keypoint_mask=None)

            loss_dict_keypoints = calc_cd(keypoints, gt_keypoints, calc_f1=False, f1_threshold=0.0001, normal_loss_type='mse')
            loss_cd_keypoints = loss_dict_keypoints['cd_p'].mean()
            print('loss_cd_keypoints:', loss_cd_keypoints)
            results_cd_keypoints += loss_cd_keypoints / len(testloader)
            loss_dict_points = calc_cd(generated_data[:,:,:3], X, calc_f1=False, f1_threshold=0.0001,  normal_loss_type='mse')
            print(loss_dict_points['cd_p'])
            loss_cd_points = loss_dict_points['cd_p'].mean()
            print('loss_cd_points:', loss_cd_points)
            results_cd_points += loss_cd_points / len(testloader)
            for j in range(keypoints.shape[0]):
                keypointsi=np.array(keypoints[j].cpu())
                np.savetxt(save_dir+'/'+str(iter*batch_size+j)+'_'+str(keypointsi.shape[0])+'_keypoints.pts',keypointsi,'%f %f %f')
                generated_datai = np.array(generated_data[:,:,:3][j].cpu())
                np.savetxt(save_dir + '/' + str(iter*batch_size+j) + '_' + str(generated_datai.shape[0]) + '_points.pts', generated_datai, '%f %f %f')
    print('results_cd_keypoints:', results_cd_keypoints)
    print('results_cd_points:', results_cd_points)

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
    parser.add_argument('--keypoints_type', type=str, default='axis_physics', help='')
    parser.add_argument('--ckpt', type=str, default='../exps/exp_6FB_generation_1024/latent_ddpm_exps/32_keypoints/T1000_betaT0.02_latent_feature_generation_6FB-3D/checkpoint/pointnet_ckpt_20399.pkl', help='the checkpoint to use')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    keypoint_config = read_json_file(args.keypoint_config)
    config_file_path = os.path.split(args.config)[0]
    global train_keypoint_config
    train_config = keypoint_config["train_"+args.model_name+"_config"]        # training parameters
    global dist_config
    dist_config = keypoint_config["dist_config"]         # to initialize distributed training
    global trainset_config
    trainset_config = keypoint_config['dataset_config']
    global material_config
    material_config = keypoint_config['material_property_config']

    os.environ['CUDA_VISIBLE_DEVICES'] = keypoint_config["dist_config"]["CUDA_VISIBLE_DEVICES"]
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
    autoencoder.load_state_dict(
        torch.load(config['autoencoder_config']['ckpt'], map_location='cpu')['model_state_dict'])
    autoencoder.cuda()
    autoencoder.eval()
    diffusion_model = LatentDiffusion(standard_diffusion_config, autoencoder=autoencoder)
    evaluation(args,train_config,train_config['dataset'],net)
    torch.cuda.synchronize()


