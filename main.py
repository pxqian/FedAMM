#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from sampling import  get_train_data,Divide_groups
from options import args_parser
from Update import *
from util import *
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from model import CNNCifar,vgg16
from torch.utils.data import DataLoader, Dataset
from resnet import *
from picture import *
import time
if __name__ == '__main__':
    # parse args

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    setup_seed(args.seed)

    # log
    current_time = datetime.now().strftime('%b.%d_%H.%M.%S')
    TAG = 'exp/final_acc/{}_seed{}_wlr{}_{}_C{}_{}_user{}_{}'.format( args.dataset,args.seed,
                                                                                args.num_meta, args.epochs, args.frac,
                                                                                args.imb_factor, args.num_users,
                                                                                current_time)
    logdir = f'runs/{TAG}' if not args.debug else f'runs2/{TAG}'
    writer = SummaryWriter(logdir)

    # load dataset and split users
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transform_test)

    # train_groups, idx_to_meta, img_num_list = get_train_data(train_dataset, args)

    train_groups, idx_to_meta,dict_per_cls,img_num_list = get_train_data(train_dataset,args.dataset, args)
    validloader = DataLoader(DatasetSplit(train_dataset, idx_to_meta), batch_size=len(idx_to_meta), shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=args.test_bs, shuffle=False)
    dict_users = Divide_groups(train_dataset,train_groups, dict_per_cls,args.num_users,args)
    mkdirs("./runs/data_pic")
    pic_data(train_dataset,dict_users,args)


    model = ResNet32(args.num_classes).cuda(args.device)
    model.train()
    vnet = VNet(1, 100, 1).to(args.device)
    vnet.train()
    # copy weights
    w_glob = model.state_dict()
    v_glob=vnet.state_dict()

    loss_train=[]

    for epoch in range(args.epochs):

        w_locals, loss_locals = [], []
        theta_locals=[]

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            pre_idx=np.random.choice(list(dict_users[idx]),args.num_meta*args.num_classes,replace=False)
            pre_idx=list(pre_idx)
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=dict_users[idx],pre_idx=pre_idx)
            w,meta_model,meta_vnet, loss\
                = local_model.train(net=copy.deepcopy(model).to(args.device),
                                    vnet=copy.deepcopy(vnet).to(args.device),
                                    v_glob=v_glob)

            optimizer_w = torch.optim.SGD(meta_vnet.params(), args.v_lr, momentum=args.momentum,
                                          nesterov=args.nesterov, weight_decay=args.weight_decay)

            images_val, labels_val = next(iter(validloader))
            images_val, labels_val = images_val.to(args.device), labels_val.to(args.device)
            y_g_hat = meta_model(images_val)
            l_g_meta = F.cross_entropy(y_g_hat, labels_val)
            meta_vnet.zero_grad()
            l_g_meta.backward()
            optimizer_w.step()

            w_locals.append(w)
            loss_locals.append(loss)
            theta_locals.append(meta_vnet.state_dict())

        # update global weights
        w_glob = FedAvg(w_locals)
        v_glob=FedAvg(theta_locals)

        vnet.load_state_dict(v_glob)
        model.load_state_dict(w_glob)


        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Train loss {:.3f}'.format(epoch+1, loss_avg))
        loss_train.append(loss_avg)
        writer.add_scalar('train_loss', loss_avg,epoch+1)
        test_acc, test_loss = test_img(model, test_dataset, args)
        writer.add_scalar('test_loss', test_loss, epoch + 1)
        writer.add_scalar('test_acc', test_acc, epoch + 1)

    # testing
    model.eval()
    acc_train, loss_train = test_img(model, train_dataset, args)
    acc_test, loss_test = test_img(model, test_dataset, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    writer.close()
