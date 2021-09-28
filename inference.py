"""
ADC inference based on NeuroSim
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.nn import modules
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import models
import logging
from torchsummary import summary

from utils import *
from collections import OrderedDict
from models.quant import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/ImageNet Training')
parser.add_argument('--model', type=str, help='model type')
parser.add_argument('--multiplier', type=float, default=1.0, help='Scale of the mobilenet')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 64)')

parser.add_argument('--schedule', type=int, nargs='+', default=[60, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--lr_decay', type=str, default='step', help='mode for learning rate decay')
parser.add_argument('--print_freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 200)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--log_file', type=str, default=None,
                    help='path to log file')

# dataset
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset: CIFAR10 / ImageNet_1k')
parser.add_argument('--data_path', type=str, default='./data/', help='data directory')

# model saving
parser.add_argument('--save_path', type=str, default='./save/', help='Folder to save checkpoints and log.')
parser.add_argument('--evaluate', action='store_true', help='evaluate the model')

# Acceleration
parser.add_argument('--ngpu', type=int, default=4, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=4,help='number of data loading workers (default: 2)')

# Fine-tuning
parser.add_argument('--fine_tune', dest='fine_tune', action='store_true',
                    help='fine tuning from the pre-trained model, force the start epoch be zero')
parser.add_argument('--resume', default='', type=str, help='path of the pretrained model')
parser.add_argument('--ref', default='', type=str, help='path of the reference model')

# quantization
parser.add_argument('--wbit', type=int, default=4, help='weight precision')
parser.add_argument('--abit', type=int, default=4, help='activation precision')
parser.add_argument('--alpha_init', type=int, default=10., help='initial activation clipping')
parser.add_argument('--q_mode', type=str, default="mean", help='weight quantization mode')
parser.add_argument('--k', type=int, default=2, help='coefficient of quantization boundary')

# activation clipping(PACT)
parser.add_argument('--clp', dest='clp', action='store_true', help='using clipped relu in each stage')
parser.add_argument('--a_lambda', type=float, default=0.01, help='The parameter of alpha L2 regularization')

# RRAM inference
parser.add_argument('--col_size', type=int, default=16, help='Column size of the RRAM')
parser.add_argument('--cellBit', type=int, default=2, help='precision of the rram cell')
parser.add_argument('--adc_prec', type=int, default=5, help='adc precision')
parser.add_argument('--subArray', type=int, default=128, help='subarray size')
parser.add_argument('--swipe_ll', type=int, default=0, help='SWIPE level')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()  # check GPU

def main():
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    logger = logging.getLogger('training')
    if args.log_file is not None:
        fileHandler = logging.FileHandler(args.save_path+args.log_file)
        fileHandler.setLevel(0)
        logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(0)
    logger.addHandler(streamHandler)
    logger.root.setLevel(0)

    logger.info(args)

    # log = open(os.path.join(args.save_path, 'log.txt'), 'w')
    
    # Preparing data
    if args.dataset == 'cifar10':
        data_path = args.data_path + 'cifar'

        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)

        testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=16)
        num_classes = 10
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])  # here is actually the validation dataset

        train_dir = os.path.join(args.data_path, 'train')
        test_dir = os.path.join(args.data_path, 'val')

        train_data = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
        test_data = torchvision.datasets.ImageFolder(test_dir, transform=test_transform)

        trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        num_classes = 1000
    else:
        raise ValueError("Dataset must be either cifar10 or imagenet")  

    # Prepare the model
    logger.info('==> Building model..\n')
    model_cfg = getattr(models, args.model)
    model_cfg.kwargs.update({"num_classes": num_classes, "wbit": args.wbit, "abit":args.abit, 
            "alpha_init": args.alpha_init, "ADCprecision":args.adc_prec, "cellBit":args.cellBit, "swipe_ll":args.swipe_ll})
    net = model_cfg.base(*model_cfg.args, **model_cfg.kwargs) 

    logger.info(net)

    for name, value in net.named_parameters():
        logger.info(f'name: {name} | shape: {list(value.size())}')
    
    if args.fine_tune:
        new_state_dict = OrderedDict()
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model_dict = checkpoint['state_dict']

        for k, v in model_dict.items():
            name = k
            new_state_dict[name] = v
        
        state_tmp = net.state_dict()
        state_tmp.update(new_state_dict)
        
        net.load_state_dict(state_tmp)
        logger.info("=> loaded checkpoint '{}' best acc = {}".format(args.resume, checkpoint['acc']))

    if args.use_cuda:
        if args.ngpu > 1:
            net = torch.nn.DataParallel(net)

    # Loss function
    net = net.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    
    # Evaluate
    if args.evaluate:
        # set the clipping parameter to the Qconv2d Layer
        count = 0
        for m in net.modules():
            if isinstance(m, RRAMConv2d):
                m.layer_idx = count
                count += 1

        test_acc, val_loss = test(testloader, net, criterion, 0)
        logger.info(f'Test accuracy: {test_acc}')
        
        exit()
    else:
        raise ValueError("For training and normal test, please use train.py under the same directory!")

if __name__ == '__main__':
    main()
