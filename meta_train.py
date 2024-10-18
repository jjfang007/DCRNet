import argparse
import logging
import os

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR, StepLR
import torch.nn as nn

from backbone.Network import Network
from data.datamgr import SetDataManager
from eval.meta_eval import meta_test
from models.model import get_model
from utils import load_model


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # general
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--seed', type=int, default=31)
    parser.add_argument('--use_gpu', action='store_true', help='use gpu to train')
    parser.add_argument('--convnet_name', type=str, default='conv64f', choices=['resnet12', 'conv64f'],
                        help='ConVet Net name')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('-step_size', type=int, default=10)
    parser.add_argument('-gamma', type=float, default=0.5)
    parser.add_argument('--image_size', type=int, default=84, help='image size')

    # dataset and model
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                        choices=['miniImageNet', 'tieredImageNet', 'CUB', 'StanfordCar','StanfordDog'])

    # specify folder
    parser.add_argument('--pre_trained', type=str,
                        default='output_dir/pretrain/classification/miniImageNet/resnet12/max_acc.pth',
                        help='cnn path')
    parser.add_argument('--data_root', type=str, default='/data/qq/Datasets', help='path to data root')
    # parser.add_argument('--data_root', type=str, default='/public/home/lfzh/FJJ/data', help='path to data root')
    parser.add_argument('--save_root', type=str, default='./output_dir', help='path to save output')
    parser.add_argument('--metatrain_path', type=str, default='metatrain', help='path to save model')
    parser.add_argument('--metatrain_log_path', type=str, default='metatrain_log', help='path to save log')
    parser.add_argument('--model_name', type=str, default='ATL', help='model name')
    parser.add_argument('--continue_train', action='store_true', help='continue to train model')
    # parser.add_argument('--model_trained_path', default='./output_dir/metatrain/ATL_mini_rand_5w5s_best.pth',
    #                     help='continue to train model')

    # setting for meta-learning
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=5, metavar='N', help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N', help='Number of query in test')
    parser.add_argument('--temperature_p', default=1, type=int, help='the temperature for contract prototype')

    opt = parser.parse_args()
    if opt.convnet_name == 'resnet12':
        opt.inplanes = 640
    elif opt.convnet_name == 'conv64f':
        opt.inplanes = 64
    else:
        raise ValueError('Unknown Convnet')
    if torch.cuda.is_available() and opt.use_gpu:
        opt.device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        opt.device = torch.device("cpu")
    opt.save_name = '{}_{}_rand_size_{}_{}w{}s'.format(opt.model_name, opt.dataset[:4], opt.image_size, opt.n_ways,
                                                       opt.n_shots)

    opt.log_name = '{}_{}_rand_size_{}_{}w{}s.log'.format(opt.model_name, opt.dataset[:4], opt.image_size, opt.n_ways,
                                                          opt.n_shots)
    opt.model_save_path = os.path.join(opt.save_root, opt.metatrain_path)
    if not os.path.isdir(opt.model_save_path):
        os.makedirs(opt.model_save_path)
    opt.log_save_path = os.path.join(opt.save_root, opt.metatrain_log_path)
    if not os.path.isdir(opt.log_save_path):
        os.makedirs(opt.log_save_path)

    opt.n_gpu = torch.cuda.device_count()

    return opt


def main():
    opt = parse_option()
    print(opt)
    logging.basicConfig(filename=os.path.join(opt.log_save_path, opt.log_name), filemode='w',
                        level=logging.INFO)
    print("Starting Meta-Training ..... \n\n")

    # dataloader
    base_file = os.path.join(opt.data_root, opt.dataset, 'base.json')
    val_file = os.path.join(opt.data_root, opt.dataset, 'novel.json')
    train_datamgr = SetDataManager(opt.image_size, n_query=opt.n_queries, n_way=opt.n_ways, n_support=opt.n_shots)
    meta_trainloader, cls_num = train_datamgr.get_data_loader(base_file, aug=False)
    val_datamgr = SetDataManager(opt.image_size, n_query=opt.n_queries, n_way=opt.n_ways, n_support=opt.n_shots,
                                 n_eposide=100)
    meta_valloader, _ = val_datamgr.get_data_loader(val_file, aug=False)
    opt.class_num = cls_num
    # print("load {model_path_pre}".format(model_path_pre=opt.model_path_pre))
    convnet = Network(opt, 'meta_train')
    # load contrastive pre train model
    # paras = torch.load(opt.pre_trained)
    # convnet.load_state_dict(paras, strict=False)
    # load classification pre train model
    load_model(convnet, opt.pre_trained)
    criterion = nn.CrossEntropyLoss()
    model = get_model(opt, convnet=convnet)
    if opt.continue_train:
        model = torch.load(opt.model_trained_path)
    model = model.to(opt.device)

    # optimizer
    if opt.convnet_name == 'resnet12':
        opt.learning_rate = 5e-4
        optimizer = torch.optim.SGD(model.parameters(), nesterov=True, momentum=0.9, lr=opt.learning_rate)
    elif opt.convnet_name == 'conv64f':
        opt.learning_rate = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    lr_scheduler = StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)
    # mate train routine
    max_acc = 0
    for epoch in range(1, opt.epochs + 1):
        train(epoch, meta_trainloader, model, criterion, optimizer, opt)
        lr_scheduler.step()
        # validation
        all_acc, eval_loss = meta_test(model, dataloader=meta_valloader, opt=opt)
        acc_mean = np.mean(all_acc)
        acc_std = np.std(all_acc)
        if acc_mean > max_acc:
            max_acc = acc_mean
            print("---epoch %d saving best model max acc %4.2f%% +- %4.2f%% ---" % (
                epoch, max_acc, 1.96 * acc_std / np.sqrt(len(meta_valloader))))
            bestpath = os.path.join(opt.model_save_path, opt.save_name + "_best.pth")
            torch.save(model, bestpath)
        print('--- Loss = %.6f ---' % (eval_loss / len(meta_valloader)))
        print('--- Best accuracy %4.2f%% ---' % max_acc)
        print('--- epoch %d Test Acc = %4.2f%% +- %4.2f%% ---' % (
            epoch, acc_mean, 1.96 * acc_std / np.sqrt(len(meta_valloader))))

        logging.info('--- epoch %d Test Acc = %4.2f%% +- %4.2f%% ---' % (
            epoch, acc_mean, 1.96 * acc_std / np.sqrt(len(meta_valloader))))
    logging.info('--- the best Acc = %4.2f%% ---' % (max_acc))


def train(epoch, meta_trainloader, model, criterion, optimizer, opt):
    """One epoch training"""
    model.train()
    avg_loss = 0.
    for i, (x, _) in enumerate(meta_trainloader):
        # ===================forward=====================
        x = x.to(opt.device)
        support_x = x[:, :opt.n_shots].contiguous().view(-1, *x.size()[2:])
        query_x = x[:, opt.n_shots:].contiguous().view(-1, *x.size()[2:])
        support_y = torch.arange(opt.n_ways).reshape(opt.n_ways, 1).repeat(1, opt.n_shots).view(
            -1).to(opt.device)
        query_y = torch.arange(opt.n_ways).reshape(opt.n_ways, 1).repeat(1, opt.n_queries).view(
            -1).to(opt.device)
        optimizer.zero_grad()
        if opt.model_name == 'GLOSENet' or opt.model_name == 'TCDSNet' or opt.model_name == 'TSDCNet' or opt.model_name == 'GSLNet':
            loss2, score = model(support_x, support_y, query_x, query_y)
            loss = 0.01 * loss2 + 0.99 * criterion(score, query_y)
        elif opt.model_name == 'GSLSANet':
            loss1, loss2, score = model(support_x, support_y, query_x, query_y)
            loss = 0.1 * loss1 + 0.1 * loss2 + criterion(score, query_y)
        else:
            score = model(support_x, support_y, query_x, query_y)
            loss = criterion(score, query_y)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        if (i + 1) % 10 == 0:
            print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i + 1, len(meta_trainloader),
                                                                    avg_loss / float(i + 1)))


if __name__ == '__main__':
    main()
