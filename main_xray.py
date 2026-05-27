import argparse
import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models as torchvision_models
import numpy as np
from scipy.linalg import hadamard

from models import resnet_ex_sp_50
from utils.dataloader import Xray_DATASET
from utils.fix_seeds import fix_random_seeds
# from sklearn.metrics._ranking import roc_auc_score
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import copy

import warnings
from utils.logger import *

warnings.filterwarnings('ignore')

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))

model_names = ['main_vit_tiny', 'main_vit_base'] + torchvision_archs


def get_args_parser():
    parser = argparse.ArgumentParser('CausalMixNet', add_help=False)

    # Model params

    parser.add_argument('--img_size', default=256)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)  # [0, 1993]
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dataset', default='xray', type=str) 
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
    parser.add_argument('--algorithm',default='resnet18-MIX-SP',type=str)
    parser.add_argument('--source_domains',default='APT',type=str)
    parser.add_argument('--is_HE',default=None,type=bool)
    parser.add_argument('--N_Times',default=20,type=int)
    parser.add_argument('--test_iter',type=int)
    parser.add_argument('--K',default=5,type=int)
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

def eval(model, testloader,device,num_class=3):
    model.eval()
    # print('Evaluating')

    num_data=len(testloader.dataset)
    pred_global = torch.zeros(num_data, num_class).to(device)
    tru_label = torch.zeros(num_data).to(device)

    with (torch.no_grad()):
        for i, data in enumerate(testloader):
            index,images, labels = data
            images = images.to(device)
            labels = labels.to(device).float()

            _,_,raw_logits = model(images)
            pred_global[index]=raw_logits
            tru_label[index]=labels

    pred_softmax=torch.softmax(pred_global, dim=1)
    pred_true=torch.cat((pred_softmax,tru_label.view(-1,1)),dim=1).cpu().numpy()

    label = tru_label.cpu().numpy()
    pred_softmax_global = torch.softmax(pred_global, dim=1).cpu().numpy()
    pred_global = torch.max(pred_global, dim=1)[1]
    pred_global = pred_global.cpu().numpy()



    acc_global = accuracy_score(label, pred_global)
    f1_global = f1_score(label, pred_global, average='macro')
    # print()

    auc_ovo_global = roc_auc_score(label, pred_softmax_global[:,1])
    print(' acc_global: ', acc_global, ' F1_global: ',f1_global, ' auc_global: ',auc_ovo_global)
    return auc_ovo_global


def train(train_loader,val_loader,test_loader,args,writer):
    device = torch.device(args.device)
    model = resnet_ex_sp_50.Model(args.num_classes, mode=args.mode, K=args.K)
    model.to(device)
    best_acc = 0.0

    criterion = torch.nn.CrossEntropyLoss()
    parameters = model.parameters()
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-5,betas=(0.9, 0.999))

    for epoch in range(args.epoch):
        loss = 0.0
        # print(epoch)
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

        if epoch % 1 == 0:
            print('-------------------epoch: ', epoch, '----------------------')
            val_auc=eval(model,val_loader,device,args.num_classes)
            if val_auc>best_acc:
                best_acc=val_auc
                test_auc = eval(model, test_loader, device, args.num_classes)
                best_test_auc=test_auc
            print('best_test_auc: ',best_test_auc)


def main(args):
    fix_random_seeds(args.seed)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    # print("{}".format(args).replace(', ', ',\n'))

    cudnn.benchmark = True

    train_dir = 'data/NIH_Xray/bias_setting/GbPB.txt'
    val_dir = 'data/NIH_Xray/bias_setting/NIHval.txt'
    test_dir = 'data/NIH_Xray/bias_setting/NIHtest.txt'

    # train_dir = 'data/NIH_Xray/bias_setting/GbPTr1.txt'
    # val_dir = 'data/NIH_Xray/bias_setting/NIHval.txt'
    # test_dir = 'data/NIH_Xray/bias_setting/NIHtest.txt'

    # train_dir = 'data/NIH_Xray/bias_setting/GbPTr2.txt'
    # val_dir = 'data/NIH_Xray/bias_setting/NIHval.txt'
    # test_dir = 'data/NIH_Xray/bias_setting/NIHtest.txt'


    args.root = '...'
    train_ = read_txt(train_dir)
    val_ = read_txt(val_dir)
    test_ = read_txt(test_dir)
    args.test_iter = 10
    args.is_HE = True
    args.num_classes = 2

    dataset_train = Xray_DATASET(args.root, train_, args.img_size, is_train=True, is_HE=args.is_HE)
    print(f"Train data loaded: there are {len(dataset_train)} images.")

    dataset_val = Xray_DATASET(args.root, val_, args.img_size, is_train=False, is_HE=args.is_HE)
    print(f"Val data loaded: there are {len(dataset_val)} images.")

    dataset_test = Xray_DATASET(args.root, test_, args.img_size, is_train=False, is_HE=args.is_HE)
    print(f"Domain1 data loaded: there are {len(dataset_test)} images.")

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, drop_last=False)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, drop_last=False)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, drop_last=False)



    args.save_path = './result/' + args.algorithm + '_'+'Xray'+'_' + args.source_domains
    args.save_path=args.save_path+'_best_model.pth'

    print(args)

    log_path = './logger'
    dataset_size = [len(train_), len(val_), len(test_)]
    writer = init_log(args, log_path, len(train_loader), dataset_size)

    train(train_loader, val_loader, test_loader, args, writer)
    writer.close()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # print(args)
    start_train = time.time()
    main(args)
    end_train = time.time()
    print('Training time in: %s' % ((end_train - start_train) / 3600))

    # python main_xray.py --data covid --source_domains APT --algorithm resnet18-MIX-SP --K 5 --ratio 0.8 --alpha 5.0 --beta 5.0 --N_Times 20 --device cuda:1 --batch_size 32
