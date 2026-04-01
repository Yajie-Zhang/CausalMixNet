# The baseline implementation of resnet, alexnet, vit, and resnet_attention

import argparse
import os
import sys
import datetime
import time
import math
from pathlib import Path
from functools import partial
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models
from torchvision import models as torchvision_models
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from scipy.linalg import hadamard
import cv2

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import torch.nn.functional as F
from utils.validate_ex import algorithm_validate,algorithm_validate_he

from models import resnet_model_intervention,resnet,resnet_ex_sp
from utils.dataloader import HE_DATASET, Xray_DATASET, MIMIC_DATASET, ISIC_DATASET
from utils.fix_seeds import fix_random_seeds
from sklearn.metrics._ranking import roc_auc_score
from torch.optim import lr_scheduler
from loss import FocalLoss
from utils.logger import *

import warnings

warnings.filterwarnings('ignore')

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))

model_names = ['main_vit_tiny', 'main_vit_base'] + torchvision_archs


def get_args_parser():
    parser = argparse.ArgumentParser('DAMA', add_help=False)

    # Model params

    parser.add_argument('--img_size', default=256)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1000, type=int)
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
    parser.add_argument('--data',default='he',type=str)
    parser.add_argument('--name',type=str)
    parser.add_argument('--algorithm',default='resnet18',type=str)
    parser.add_argument('--source_domains',default='APT',type=str)
    parser.add_argument('--is_HE',default=None,type=bool)
    parser.add_argument('--N_Times',default=20,type=int)
    parser.add_argument('--test_iter',type=int)
    parser.add_argument('--K',default=5,type=int)
    parser.add_argument('--ratio',default=0.8,type=float)
    parser.add_argument('--alpha',default=0.2,type=float)
    parser.add_argument('--beta',default=0.0,type=float)
    return parser


def read_txt(List):
    all = []
    for line in open(List, encoding='utf-8'):
        # line.replace('\n','.jpg\n')
        all.append(line)
    return all

def train(train_loader,val_loader,domain1_loader,args,writer):
    device = torch.device(args.device)
    model = resnet_ex_sp.Model(args.num_classes, mode=args.mode,K=args.K)
    # path = torch.load('./result/resnet18_he_APT_best_model.pth', map_location=device)
    # model.load_state_dict(path, strict=False)
    model.to(device)
    if args.data == 'he':
        val_multi_acc, val_multi_f1, val_by_acc, val_by_pr, val_by_rc, val_by_f1 = algorithm_validate_he(
            model, val_loader, 0 ,'val', device, writer=writer)
        val_multi_acc, _, _, _, _, _ = algorithm_validate_he(model, domain1_loader, 0, 'test',
                                                             device, writer=writer)

    # state_dict = model.state_dict()
    # model = resnet_ex_sp.Model(args.num_classes, mode=args.mode, class_att=class_att_var)
    # model.load_state_dict(state_dict, strict=False)
    # model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    SpCriterion = torch.nn.CrossEntropyLoss(reduction='none')
    parameters = model.parameters()
    # optimizer=torch.optim.AdamW(parameters, lr=1e-4,weight_decay=1e-2)
    # optimizer = torch.optim.Adam(parameters, lr=args.lr)
    # optimizer=torch.optim.SGD(parameters,lr=0.0001)
    # lr_scheduler_global = lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)
    optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)

    iter_num = 0
    best_val_auc=0
    for epoch in range(args.epoch):
        loss = 0.0
        prec = 0.0
        # print(epoch)
        for i, (index, img, label) in enumerate(train_loader):
            model.train()
            img = img.to(device)
            if label.shape[0]<2:
                pass
            else:
                B=img.shape[0]
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
                    if args.data == 'he':
                        val_multi_acc, val_multi_f1, val_by_acc, val_by_pr, val_by_rc, val_by_f1 = algorithm_validate_he(
                            model, val_loader, epoch, 'val', device,writer=writer)
                        _, _, _, _, _, _ = algorithm_validate_he(model, domain1_loader, epoch, 'test',
                                                                             device,writer=writer)
                        if (val_multi_acc+val_multi_f1+val_by_acc+val_by_pr+val_by_rc+val_by_f1) > best_val_auc:
                            if epoch>10:
                                best_val_auc = val_multi_acc + val_multi_f1 + val_by_acc + val_by_pr + val_by_rc + val_by_f1
                                torch.save(model.state_dict(), args.save_path)

                iter_num = iter_num + 1
    if args.data == 'he':
        val_multi_acc, val_multi_f1, val_by_acc, val_by_pr, val_by_rc, val_by_f1 = algorithm_validate_he(model,
                                                                                                         val_loader,
                                                                                                         epoch, 'val',
                                                                                                         device,args.save_path,writer=writer)
        val_multi_acc, _, _, _, _, _ = algorithm_validate_he(model, domain1_loader, epoch, 'test', device,args.save_path,writer=writer )

def main(args):
    fix_random_seeds(args.seed)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    # print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    cudnn.benchmark = True

    if args.data == 'he':
        train_dir = './data/HE_breast/BRACS/BRACS_TRAIN.txt'
        val_dir = './data/HE_breast/BRACS/BRACS_VAL.txt'
        test_dir = './data/HE_breast/BRACS/BRACS_TEST.txt'
        args.root = '/datasets_hdd2/yjzhang/data/BRACS_breast/BRACS_RoI/latest_version'
        train_=read_txt(train_dir)
        val_=read_txt(val_dir)
        test_=read_txt(test_dir)
        args.test_iter=500
        args.is_HE = True
        args.num_classes = 5

        dataset_train = HE_DATASET(args.root, train_, args.img_size, is_train=True, is_HE=args.is_HE)
        print(f"Train data loaded: there are {len(dataset_train)} images.")

        dataset_val = HE_DATASET(args.root, val_, args.img_size, is_train=False, is_HE=args.is_HE)
        print(f"Val data loaded: there are {len(dataset_val)} images.")

        dataset_test = HE_DATASET(args.root, test_, args.img_size, is_train=False, is_HE=args.is_HE)
        print(f"Domain1 data loaded: there are {len(dataset_test)} images.")

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers, drop_last=False)
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers, drop_last=False)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, drop_last=False)


    if args.data == 'covid':
        args.is_HE = None
        if args.source_domains == 'APT':
            train_dir = './data/OCT/apt_train.txt'
            val_dir = './data/OCT/apt_test.txt'
            args.root = '/datasets_hdd2/yjzhang/data/yanke'
            domain1_train_dir = './data/OCT/ddr_train.txt'
            domain1_test_dir = './data/OCT/ddr_test.txt'
            domain2_train_dir = './data/OCT/deepdr_train.txt'
            domain2_test_dir = './data/OCT/deepdr_test.txt'
            args.domain_root = '/datasets_hdd2/yjzhang/data/yanke'
            args.test_iter=30
            train_=read_txt(train_dir)
            val_=read_txt(val_dir)
            test_=read_txt(domain1_train_dir) + read_txt(domain1_test_dir)+read_txt(domain2_train_dir) + read_txt(domain2_test_dir)



        elif args.source_domains == 'DDR':
            train_dir = './data/OCT/ddr_train.txt'
            test_dir = './data/OCT/ddr_test.txt'
            args.root = '/datasets_hdd2/yjzhang/data/yanke'
            domain1_train_dir = './data/OCT/apt_train.txt'
            domain1_test_dir = './data/OCT/apt_test.txt'
            domain2_train_dir = './data/OCT/deepdr_train.txt'
            domain2_test_dir = './data/OCT/deepdr_test.txt'
            args.domain_root = '/datasets_hdd2/yjzhang/data/yanke'
            args.test_iter=100
            train_=read_txt(train_dir)
            val_=read_txt(test_dir)
            test_ = read_txt(domain1_train_dir) + read_txt(domain1_test_dir)+read_txt(domain2_train_dir) + read_txt(domain2_test_dir)

        elif args.source_domains == 'DEEPDR':
            train_dir = './data/OCT/deepdr_train.txt'
            test_dir = './data/OCT/deepdr_test.txt'
            args.root = '/datasets_hdd2/yjzhang/data/yanke'
            domain1_train_dir = './data/OCT/apt_train.txt'
            domain1_test_dir = './data/OCT/apt_test.txt'
            domain2_train_dir = './data/OCT/ddr_train.txt'
            domain2_test_dir = './data/OCT/ddr_test.txt'
            args.domain_root = '/datasets_hdd2/yjzhang/data/yanke'
            args.test_iter=10
            train_=read_txt(train_dir)
            val_=read_txt(test_dir)
            test_= read_txt(domain1_train_dir) + read_txt(domain1_test_dir)+read_txt(domain2_train_dir) + read_txt(domain2_test_dir)
        args.num_classes=5
        dataset_train = HE_DATASET(args.root, train_, args.img_size, is_train=True, is_HE=args.is_HE)
        print(f"Train data loaded: there are {len(dataset_train)} images.")

        dataset_val = HE_DATASET(args.root, val_, args.img_size, is_train=False, is_HE=args.is_HE)
        print(f"Val data loaded: there are {len(dataset_val)} images.")

        dataset_test = HE_DATASET(args.domain_root, test_, args.img_size, is_train=False, is_HE=args.is_HE)
        print(f"Domain1 data loaded: there are {len(dataset_test)} images.")

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers, drop_last=False)
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers, drop_last=False)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, drop_last=False)

    if args.data=='covid':
        args.save_path='./result/'+args.algorithm+'_oct_'+args.source_domains
    else:
        args.save_path = './result/' + args.algorithm + '_'+args.data+'_' + args.source_domains+'_'+str(args.alpha)+str(args.beta)
    # log_path = os.path.join(args.save_path)
    args.save_path=args.save_path+'_best_model.pth'

    print(args)

    log_path = './logger'
    dataset_size = [len(train_), len(val_), len(test_)]
    writer = init_log(args, log_path, len(train_loader), dataset_size)

    train(train_loader, val_loader, test_loader, args, writer)
    os.mknod(os.path.join(log_path, 'done'))
    writer.close()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # print(args)
    start_train = time.time()
    main(args)
    end_train = time.time()
    print('Training time in: %s' % ((end_train - start_train) / 3600))

    # python main_bracs.py --data he --source_domains APT --device cuda:9 --algorithm resnet18-MIX-SP --K 5 --ratio 0.8 --alpha 0.2 --N_Times 20
