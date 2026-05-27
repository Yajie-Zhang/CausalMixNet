import argparse
import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models as torchvision_models
import numpy as np
from scipy.linalg import hadamard
from utils.logger import *

from utils.validate_ex import algorithm_validate,algorithm_validate_he,mean_average_precision

from models import resnet_ex_sp
from utils.dataloader import CUB_DATASET
from utils.fix_seeds import fix_random_seeds

import warnings

warnings.filterwarnings('ignore')

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))

model_names = ['main_vit_tiny', 'main_vit_base'] + torchvision_archs


def get_args_parser():
    parser = argparse.ArgumentParser('CausalMixNet', add_help=False)

    # Model params

    parser.add_argument('--img_size', default=256)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1993, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dataset', default='xray', type=str)  # COVID ot BreakHis
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--job_id', default=0)
    parser.add_argument('--model', default='resnet_attribute', type=str)
    parser.add_argument('--cv', default=5, type=int)
    parser.add_argument('--ce', default=True, type=bool)
    parser.add_argument('--aa', default=True, type=bool)
    parser.add_argument('--ra', default=True, type=bool)
    parser.add_argument('--mode', default='multi_cls', type=str)
    parser.add_argument('--num_att', default=6, type=int)
    parser.add_argument('--kl_weight', default=1.0, type=float)
    parser.add_argument('--ce2_weight', default=0.5, type=float)
    parser.add_argument('--data',default='covid',type=str)
    parser.add_argument('--name',type=str)
    parser.add_argument('--algorithm',default='resnet18',type=str)
    parser.add_argument('--source_domains',default='APT',type=str)
    parser.add_argument('--is_HE',default=None,type=bool)
    parser.add_argument('--N_Times',default=100,type=int)
    parser.add_argument('--test_iter',type=int)
    parser.add_argument('--K',default=3,type=int)
    parser.add_argument('--ratio',default=0.8,type=float)
    parser.add_argument('--alpha',default=5.0,type=float)
    parser.add_argument('--beta',default=5.0,type=float)
    return parser

def read_txt(List):
    all = []
    for line in open(List, encoding='utf-8'):
        # line.replace('\n','.jpg\n')
        all.append(line)
    return all


def train(train_loader,val_loader, args,percent,writer):
    device = torch.device(args.device)
    model = resnet_ex_sp.Model(args.num_classes, mode=args.mode,K=args.K)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    iter_num = 0
    best_val_auc = 0
    for epoch in range(args.epoch):
        loss = 0.0
        for i, (index, img, label) in enumerate(train_loader):
            model.train()
            img = img.to(device)
            if label.shape[0] < 2:
                pass
            else:
                B = img.shape[0]
                label = label.to(device)
                label_oh = torch.tensor(np.eye(args.num_classes, dtype=np.uint8)[label.cpu().numpy()]).float().to(
                    label.device)

                optimizer.zero_grad()
                x_sim, x, y, maskX, maskY, exX, exY = model(img, None, label_oh)

                loss = criterion(y, label) + args.beta * criterion(exY, label) + args.alpha * criterion(maskY, label)
                raw_loss = loss
                total_loss = raw_loss
                total_loss.backward()
                optimizer.step()

                if iter_num % args.test_iter == 0:
                    val_acc, val_f1, val_auc = algorithm_validate(model, val_loader, epoch, 'val', device,writer=writer)
                    # print('epoch:',epoch,'result:',result)
                    if (val_acc+val_auc+val_f1)>best_val_auc:
                        best_val_auc=(val_acc+val_auc+val_f1)

                iter_num = iter_num + 1


def main(args):
    fix_random_seeds(args.seed)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    # print("{}".format(args).replace(', ', ',\n'))

    cudnn.benchmark = True

    args.num_classes = 100
    for i in [0,10,20,30,40,50,60]:
        args.root='...'

        train_dir='./data/CUB/noise_train_'+str(i)+'.txt'
        val_dir='./data/CUB/noise_val_'+str(i)+'.txt'
        args.num_classes = 100
        args.save_path='./result/'+args.algorithm+'_cub_'+str(i)
        args.save_path=args.save_path+'_best_model.pth'
        args.iter_num=100
        args.test_iter=50
        train_=read_txt(train_dir)
        val_=read_txt(val_dir)
        dataset_train = CUB_DATASET(args.root, train_, args.img_size, is_train=True, is_HE=True)
        print(f"Train data loaded: there are {len(dataset_train)} images.")

        dataset_val = CUB_DATASET(args.root, val_, args.img_size, is_train=False, is_HE=True)
        print(f"Val data loaded: there are {len(dataset_val)} images.")

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers, drop_last=False)
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers, drop_last=False)
        dataset_size = [len(dataset_train), len(dataset_val)]

        print(args)
        log_path = './logger'
        dataset_size = [len(train_), len(val_)]
        writer = init_log(args, log_path, len(train_loader), dataset_size)
        print('======the noise percent is ', str(i), '=======')

        train(train_loader, val_loader, args,str(i), writer)
        writer.close()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # print(args)
    start_train = time.time()
    main(args)
    end_train = time.time()
    print('Training time in: %s' % ((end_train - start_train) / 3600))

    # python main_cub.py --data cub --source_domains APT --device cuda:6 --algorithm resnet18-MIX-SP --K 3 --ratio 0.8 --alpha 5.0 --beta 5.0 --N_Times 0
