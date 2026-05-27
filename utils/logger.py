import sys, os, logging, shutil
from torch.utils.tensorboard import SummaryWriter
import torch, random
import numpy as np
from collections import Counter

def init_output_foler(log_path):
    if os.path.isdir(log_path):
        pass
    else:
        os.makedirs(log_path)

def init_log(args, log_path, train_loader_length, dataset_size):

    init_output_foler(log_path)
    writer = SummaryWriter(os.path.join(log_path, 'tensorboard'))
    writer.add_text('config', str(args))
    logging.basicConfig(filename=log_path + args.save_path.replace('./result','')+'_'+str(args.K)+'_'+str(args.alpha)+'_'+str(args.beta)+'_log.txt',level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("{} iterations per epoch".format(train_loader_length))
    logging.info(
        "We have {} images in train set, {} images in val set, and {} images in test set.".format(dataset_size[0],
                                                                                                  dataset_size[1],
                                                                                                  dataset_size[2]))
    logging.info(str(args))
    # logging.info(str(cfg))
    return writer