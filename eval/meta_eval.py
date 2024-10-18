from __future__ import print_function
from utils import euclidean_dist
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from torch import nn
from scipy.stats import t
import numpy as np
import torch
import scipy
import torch.nn as nn


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1 + confidence) / 2., n - 1)
    return m, h


def meta_test(model, dataloader, opt):
    """
        meta testing loop using pytorch implementation of the classifier (might give slightly worse results)
        """

    model = model.eval()
    eval_loss = 0
    acc_all = []
    for i, (x, _) in enumerate(dataloader):
        # fetch the data
        x = x.to(opt.device)
        support_x = x[:, :opt.n_shots].contiguous().view(-1, *x.size()[2:])
        query_x = x[:, opt.n_shots:].contiguous().view(-1, *x.size()[2:])
        support_y = torch.arange(opt.n_ways).reshape(opt.n_ways, 1).repeat(1, opt.n_shots).view(-1).to(opt.device)
        query_y = torch.arange(opt.n_ways).reshape(opt.n_ways, 1).repeat(1, opt.n_queries).view(-1).to(opt.device)
        # local_scores, support_patch, query_patch = set_forward(x, model, opt)
        with torch.no_grad():
            if opt.model_name == 'GLOSENet' or opt.model_name == 'TCDSNet' or opt.model_name == 'TSDCNet' or opt.model_name == 'GSLNet':
                _, score = model(support_x, support_y, query_x, query_y)
            elif opt.model_name == 'GSLSANet':
                _, _, score = model(support_x, support_y, query_x, query_y)
            else:
                score = model(support_x, support_y, query_x, query_y)
            loss = nn.CrossEntropyLoss()(score, query_y)
            eval_loss += loss
            _, predict_labels = torch.max(score, 1)
            rewards = [1 if predict_labels[j] == query_y[j].to(predict_labels.device) else 0 for j in
                       range(len(query_y))]
            rewards = torch.tensor(rewards)
            # topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
            # topk_ind = topk_labels.cpu().numpy()
            # top1_correct = np.sum(topk_ind[:, 0] == y_query)
            acc = rewards.sum(-1) / rewards.size(-1) * 100
            acc_all.append(acc)
    return acc_all, eval_loss
