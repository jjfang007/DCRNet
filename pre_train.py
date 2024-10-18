import argparse
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

import loaddataset
from backbone.Network import Network
from utils import set_seed, count_acc, Averager


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # general
    parser.add_argument('--save_freq', type=int, default=100, help='meta-eval frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--seed', type=int, default=31)
    parser.add_argument('--image_size', type=int, default=84, help='image size')
    parser.add_argument('--use_gpu', action='store_true', help='use gpu to train')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-1, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('-step_size', type=int, default=30)

    # dataset
    parser.add_argument('--dataset', type=str, default='StanfordCar',
                        choices=['miniImageNet', 'tieredImageNet', 'CUB', 'StanfordCar','StanfordDog'])

    # specify folder
    parser.add_argument('--save_root', type=str, default='./output_dir', help='path to save model')
    parser.add_argument('--pretrain_path', type=str, default='pretrain', help='path to save model')
    parser.add_argument('--pretrain_log_path', type=str, default='pretrain_log', help='path to save log')
    parser.add_argument('--data_root', type=str, default='/data/qq/Datasets', help='path to data root')
    # parser.add_argument('--data_root', type=str, default='/public/home/lfzh/FJJ/data', help='path to data root')
    parser.add_argument('--convnet_name', type=str, default='conv64f', choices=['resnet12', 'conv64f'],
                        help='ConVet Net name')

    # contrastive learning
    parser.add_argument('--train_aug', action='store_true', help='perform data augmentation or not during training ')

    # contrastive loss
    parser.add_argument('--use_selfsup_loss', action='store_true',
                        help='use the standard unsupervised contrastive loss')
    # parse & define standard parameters
    opt = parser.parse_args()
    if opt.convnet_name == 'resnet12':
        opt.inplanes = 640
    elif opt.convnet_name == 'conv64f':
        opt.inplanes = 64
    else:
        raise ValueError('Unknown Convnet')
    opt.n_gpu = torch.cuda.device_count()

    opt.model_save_path = os.path.join(opt.save_root, opt.pretrain_path, 'classification', opt.dataset, opt.convnet_name)
    if not os.path.isdir(opt.model_save_path):
        os.makedirs(opt.model_save_path)
    opt.log_save_path = os.path.join(opt.save_root, opt.pretrain_log_path, 'classification', opt.dataset, opt.convnet_name)
    if not os.path.isdir(opt.log_save_path):
        os.makedirs(opt.log_save_path)

    return opt


def train():
    opt = parse_option()
    print(opt)
    set_seed(opt.seed)
    if torch.cuda.is_available() and opt.use_gpu:
        DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")
    print("Starting Pre-Training ..... \n\n")

    # create dataloader
    # base_file = os.path.join("/HOME/scz0bj1/run/data", opt.dataset, 'base.json')
    base_file = os.path.join(opt.data_root, opt.dataset, 'base.json')
    val_file = os.path.join(opt.data_root, opt.dataset, 'val.json')
    train_dataset_manger = loaddataset.SimpleDataManager(opt.image_size, batch_size=opt.batch_size)
    train_dataset, train_data_loader, cls_num = train_dataset_manger.get_data_loader(base_file, aug=False)
    val_dataset, val_data_loader, _ = train_dataset_manger.get_data_loader(val_file, aug=False)
    opt.num_class = cls_num
    criterion = nn.CrossEntropyLoss()

    # create model
    model = Network(opt, mode='pre_train')
    model = model.to(DEVICE)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    lr_scheduler = StepLR(optimizer, step_size=opt.step_size, gamma=opt.lr_decay_rate)
    if opt.n_gpu > 1:
        model = nn.DataParallel(model)
    best_acc = 0

    # training routine
    for epoch in range(1, opt.epochs + 1):
        model.train()
        total_loss = 0
        for batch, (input, target) in enumerate(train_data_loader):

            input, target = input.to(DEVICE), target.to(DEVICE)
            logits = model(input)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()

            if (batch + 1) % 100 == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch,
                                                                        batch + 1,
                                                                        len(train_data_loader),
                                                                        total_loss / float(
                                                                            batch + 1),

                                                                        ))

        print("Epoch {:d} Finally Loss {:f}".format(epoch, total_loss / len(train_dataset) * opt.batch_size))

        with open(os.path.join(opt.log_save_path, "pretrain_loss.txt"), "a") as f:
            f.write(str(total_loss / len(train_dataset) * opt.batch_size) + " ")
        vl = Averager()
        va = Averager()
        with torch.no_grad():
            for batch, (input, target) in enumerate(train_data_loader):
                input, target = input.to(DEVICE), target.to(DEVICE)
                logits = model(input)
                loss = criterion(logits, target)
                acc = count_acc(logits, target)
                vl.add(loss.item())
                va.add(acc)
            vl = vl.item()
            va = va.item()
            print("Epoch {:d} evaluation Loss {:f} Acc {:f}".format(epoch, float(vl), float(va)))
        if va > best_acc:
            torch.save(dict(params=model.state_dict()), os.path.join(opt.model_save_path, 'max_acc.pth'))
            print("Best Pre Train Model Saved")
        if epoch % opt.save_freq == 0:
            torch.save(dict(parms=model.state_dict()), os.path.join(opt.model_save_path, 'model_pretrain_epoch' + str(epoch) + '.pth'))
        lr_scheduler.step()


if __name__ == '__main__':
    train()
