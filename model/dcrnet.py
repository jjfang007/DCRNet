import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def weights_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EnLayer(nn.Module):
    def __init__(self, in_channel=64):
        super(EnLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.enlayer(x)
        return x


class LatLayer(nn.Module):
    def __init__(self, in_channel):
        super(LatLayer, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.convlayer(x)
        return x


class DSLayer(nn.Module):
    def __init__(self, in_channel=64):
        super(DSLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.predlayer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0))#, nn.Sigmoid())

    def forward(self, x):
        x = self.enlayer(x)
        x = self.predlayer(x)
        return x


class half_DSLayer(nn.Module):
    def __init__(self, in_channel=512):
        super(half_DSLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, int(in_channel/4), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.predlayer = nn.Sequential(
            nn.Conv2d(int(in_channel/4), 1, kernel_size=1, stride=1, padding=0)) #, nn.Sigmoid())

    def forward(self, x):
        x = self.enlayer(x)
        x = self.predlayer(x)
        return x

class AttLayer(nn.Module):
    def __init__(self, n_way, n_shot, n_query, input_channels=640, ):
        super(AttLayer, self).__init__()
        self.query_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.key_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.scale = 1.0 / (input_channels ** 0.5)
        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.input_channels = input_channels

    def correlation(self, x5, seeds):
        N, K, C, H5, W5 = x5.size()
        # x5 = x5.view(-1, C, H5, W5)
        # seeds = seeds.view(-1, C, 1, 1)
        correlation_maps = F.conv3d(x5, weight=seeds)  # B,B,H,W
        correlation_maps = correlation_maps.unsqueeze(2).view(N, K, -1)
        min_value = torch.min(correlation_maps, dim=2, keepdim=True)[0]
        max_value = torch.max(correlation_maps, dim=2, keepdim=True)[0]
        correlation_maps = (correlation_maps - min_value) / (max_value - min_value + 1e-12)  # shape=[B, HW]
        correlation_maps = correlation_maps.view(N, K, 1, H5, W5)  # shape=[B, 1, H, W]
        return correlation_maps

    def forward(self, x5):
        # x: B,C,H,W
        x5 = self.conv(x5)+x5
        B, C, H, W = x5.size()
        x5 = self.query_transform(x5)
        x5_att = x5.view(self.n_way, self.n_shot, C, H, W).permute(0, 2, 1, 3, 4).contiguous().view(self.n_way, C, -1)
        x_query = self.query_transform(x5).view(B, C, -1).contiguous().view(self.n_way, self.n_shot, C, H*W)
        # x_query: N,K,HW,C
        x_query = torch.transpose(x_query, 2, 3).contiguous().view(self.n_way, self.n_shot*H*W, C)  # N, KHW, C
        # x_key: N,K,C,HW
        x_key = self.key_transform(x5).view(B, C, -1).contiguous().view(self.n_way, self.n_shot, C, H*W)
        x_key = torch.transpose(x_key, 1, 2).contiguous().view(self.n_way, C, -1)  # N, C, KHW
        # W = Q^T K: B,HW,HW
        x_w1 = torch.matmul(x_query, x_key) * self.scale  # N, KHW, KHW
        # x_w1 = torch.matmul(x5_att.permute(0, 2, 1), x5_att) * self.scale # N, KHW, KHW
        x_w = x_w1.view(self.n_way, self.n_shot * H * W, self.n_shot, H * W)
        x_w = torch.max(x_w, -1).values  # N, KHW, K
        x_w = x_w.mean(-1)
        x_w = x_w.view(self.n_way, self.n_shot, -1)   # N, K, HW
        x_w = F.softmax(x_w, dim=-1)  # N, K, HW
        #####  mine ######
        # x_w_max = torch.max(x_w, -1)
        # max_indices0 = x_w_max.indices.unsqueeze(-1).unsqueeze(-1)
        norm0 = F.normalize(x5, dim=1).view(self.n_way, self.n_shot, C, H, W)
        # norm = norm0.view(B, C, -1)
        # max_indices = max_indices0.expand(B, C, -1)
        # seeds = torch.gather(norm, 2, max_indices).unsqueeze(-1)
        x_w = x_w.unsqueeze(2)
        x_w_max = torch.max(x_w, -1).values.unsqueeze(3).expand_as(x_w)
        mask = torch.zeros_like(x_w).cuda()
        mask[x_w == x_w_max] = 1
        mask = mask.view(self.n_way, self.n_shot, 1, H, W)
        seeds = norm0 * mask
        seeds = seeds.sum(4).sum(3).unsqueeze(3).unsqueeze(4)
        cormap = self.correlation(norm0, seeds).view(-1, 1, H, W)
        x51 = (x5 * cormap).view(self.n_way, self.n_shot, self.input_channels, H, W)
        proto1 = torch.mean(x51, (1, 3, 4), True).squeeze(1).squeeze(2).squeeze(2)
        return proto1

class AEAModule(nn.Module):
    def __init__(self, inplanes, scale_value=50, from_value=0.4, value_interval=0.5):
        super(AEAModule, self).__init__()
        self.inplanes = inplanes
        self.scale_value = scale_value
        self.from_value = from_value
        self.value_interval = value_interval

        self.f_psi = nn.Sequential(
            nn.Linear(self.inplanes, self.inplanes // 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.inplanes // 16, 1),
            nn.Sigmoid()
        )

    def forward(self, x, f_x):
        # the Eq.(7) should be an approximation of Step Function with the adaptive threshold,
        # please refer to https://github.com/LegenDong/ATL-Net/pdf/ATL-Net_Update.pdf
        b, hw, c = x.size()
        clamp_value = self.f_psi(x.view(b * hw, c)) * self.value_interval + self.from_value
        clamp_value = clamp_value.view(b, hw, 1)
        clamp_fx = torch.sigmoid(self.scale_value * (f_x - clamp_value))
        attention_mask = F.normalize(clamp_fx, p=1, dim=-1)

        return attention_mask

class ATLModule(nn.Module):
    def __init__(self, opt, inplanes, transfer_name='W', scale_value=30, atten_scale_value=50, from_value=0.5,
                 value_interval=0.3):
        super(ATLModule, self).__init__()

        self.inplanes = inplanes
        self.scale_value = scale_value

        if transfer_name == 'W':
            self.W = nn.Sequential(
                nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.inplanes),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            raise RuntimeError

        self.attention_layer = AEAModule(self.inplanes, atten_scale_value, from_value, value_interval)

    def forward(self, query_data, support_data):
        q, c, h, w = query_data.size()
        s, _, _, _ = support_data.size()
        support_data = support_data.unsqueeze(0).expand(q, -1, -1, -1, -1).contiguous().view(q * s, c, h, w)

        w_query = query_data.view(q, c, h * w)
        w_query = w_query.permute(0, 2, 1).contiguous()
        w_support = support_data.view(q, s, c, h * w).permute(0, 2, 1, 3).contiguous().view(q, c, s * h * w)
        w_query = F.normalize(w_query, dim=2)
        w_support = F.normalize(w_support, dim=1)
        f_x = torch.matmul(w_query, w_support)
        attention_score = self.attention_layer(w_query, f_x)

        query_data = query_data.view(q, c, h * w).permute(0, 2, 1)
        support_data = support_data.view(q, s, c, h * w).permute(0, 2, 1, 3).contiguous().view(q, c, s * h * w)
        query_data = F.normalize(query_data, dim=2)
        support_data = F.normalize(support_data, dim=1)

        match_score = torch.matmul(query_data, support_data)
        attention_match_score = torch.mul(attention_score, match_score).view(q, h * w, s, h * w).permute(0, 2, 1, 3)

        final_local_score = torch.sum(attention_match_score.contiguous().view(q, s, h * w, h * w), dim=-1)
        final_score = torch.mean(final_local_score, dim=-1) * self.scale_value

        return final_score, final_local_score

class GSLSAModule(nn.Module):
    def __init__(self, inplanes, n_way, n_shot, n_query):
        super(GSLSAModule, self).__init__()
        self.inplanes = inplanes
        self.g = nn.Sequential(
            nn.Linear(self.inplanes, self.inplanes * 2, bias=False),
            nn.Dropout(),
            # nn.BatchNorm1d(self.inplanes * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.inplanes * 2, self.inplanes, bias=False))
        self.snet = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.LeakyReLU(0.2, inplace=True))
        self.cdsnet = CDSNet(top_lr=0.8, inplanes=inplanes, n_way=n_way, n_shot=n_shot, n_query=n_query)
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.attlayer = AttLayer(n_way=n_way, n_shot=n_shot, n_query=n_query, input_channels=inplanes)
        self.n_way = n_way
        self.n_query = n_query
        self.n_shot = n_shot

    def forward(self, local_feature):
        s, c, h, w = local_feature.size()
        global_proto = self.attlayer(local_feature).unsqueeze(1).expand(-1, self.n_shot, -1).contiguous().view(-1, self.inplanes)

        global_proto_att = global_proto.unsqueeze(1).expand(-1, h * w, -1)
        # local_feature = self.snet(local_feature)

        contrastive_loss, pos_index, neg_index = self.cdsnet(local_feature)
        pos_local_feature = local_feature * pos_index
        neg_local_feature = local_feature * neg_index
        pos_local_feature_x = pos_local_feature.view(s, c, h * w).contiguous().permute(0, 2, 1)
        pos_local_feature_x = F.normalize(pos_local_feature_x, dim=-1)
        neg_local_feature_x = neg_local_feature.view(s, c, h * w).contiguous().permute(0, 2, 1)
        neg_local_feature_x = F.normalize(neg_local_feature_x, dim=-1)
        pos_value = torch.exp(pos_local_feature_x * global_proto_att).sum(-1).mean(-1).mean(-1)
        neg_value = torch.exp(neg_local_feature_x * global_proto_att).sum(-1).mean(-1).mean(-1)
        ctc_loss = -torch.log(pos_value / neg_value)
        return contrastive_loss, ctc_loss, pos_index


class CDSNet(nn.Module):
    def __init__(self, top_lr, inplanes, n_way, n_shot, n_query):
        super(CDSNet, self).__init__()
        self.top_lr = top_lr
        self.sigma = nn.Sigmoid()
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.inplanes = inplanes
        self.query_transform = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0)
        self.key_transform = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0)
        self.scale = 1.0 / (inplanes ** 0.5)


    def forward(self, local_feature):
        B, C, H, W = local_feature.size()
        local_feature_copy = local_feature
        local_feature = F.normalize(local_feature, dim=1)
        local_feature = local_feature.view(B, C, H * W).permute(0, 2, 1).contiguous()

        mask_intra = (torch.ones(self.n_way * H * W) - torch.eye(self.n_way * H * W)).unsqueeze(0).expand(self.n_way,
                                                                                                          -1, -1).to(
            local_feature.device)
        local_feature_intra = local_feature.view(self.n_way, self.n_shot, H * W, C).contiguous().view(self.n_way,
                                                                                                      self.n_shot * H * W,
                                                                                                      C).contiguous()

        ones = torch.ones(B * H * W).to(local_feature.device)
        eyes = torch.eye(B).unsqueeze(2).expand(-1, -1, H * W).unsqueeze(3).expand(-1, -1, -1, H * W)
        eyes = eyes.permute(0, 2, 1, 3).contiguous().view(B * H * W, B * H * W).to(local_feature_intra.device)
        mask_inter = ones - eyes
        local_feature_inter = local_feature.view(B * H * W, C)
        d_intra = torch.bmm(local_feature_intra, local_feature_intra.permute(0, 2, 1)) * mask_intra
        d_intra = d_intra.mean(2).view(self.n_way * self.n_shot * H * W)
        d_inter = torch.mm(local_feature_inter, local_feature_inter.T) * mask_inter
        d_inter = d_inter.mean(1)
        cds = self.sigma(d_intra / d_inter)
        cds = cds.view(self.n_way, -1)
        _, select_index = torch.topk(cds, k=int(self.n_shot * H * W * self.top_lr))
        ones = torch.zeros([self.n_way, self.n_shot * H * W]).to(local_feature_intra.device)
        select_index = ones.scatter(1, select_index, 1).view(self.n_way, self.n_shot, H * W).contiguous().view(
            self.n_way, self.n_shot, H,
            W)
        select_index = select_index.contiguous().view(self.n_way * self.n_shot, H, W).unsqueeze(1).expand(-1, C, -1, -1)
        select_local_features = local_feature_copy * select_index
        select_local_features = F.normalize(select_local_features.view(B, C, H * W), dim=1).permute(0, 2,
                                                                                                    1).contiguous().view(
            self.n_way, self.n_shot, H * W, C).contiguous()
        l_intra = select_local_features.view(self.n_way, self.n_shot * H * W, C).contiguous()
        l_intra = torch.mm(l_intra.mean(1), l_intra.mean(1).permute(1, 0))
        l_inter = torch.bmm(select_local_features.mean(2), select_local_features.mean(2).permute(0, 2, 1))
        l_intra_mask = (torch.ones(self.n_way) - torch.eye(self.n_way)).to(local_feature)
        l_inter_mask = (torch.ones(self.n_shot) - torch.eye(self.n_shot)).unsqueeze(0).expand(self.n_way, -1, -1).to(
            local_feature.device)
        l_intra = l_intra * l_intra_mask
        l_inter = l_inter * l_inter_mask
        loss = torch.exp(torch.mean(l_inter) / torch.mean(l_intra))
        _, select_index = torch.topk(cds, k=int(self.n_shot * H * W * self.top_lr))
        zeros = torch.zeros([self.n_way, self.n_shot * W * H]).to(local_feature_intra.device)
        pos_index = zeros.scatter(1, select_index, 1).view(self.n_way, self.n_shot, H * W).contiguous().view(self.n_way,
                                                                                                             self.n_shot,
                                                                                                             H,
                                                                                                             W)
        pos_index = pos_index.contiguous().view(self.n_way * self.n_shot, H, W).unsqueeze(1).expand(-1, C, -1, -1)
        neg_index = torch.ones([self.n_way * self.n_shot, C, H, W]).to(local_feature_intra.device) - pos_index

        return loss, pos_index, neg_index


class DCRNet(nn.Module):
    def __init__(self, opt, cnn, inplanes):
        super(DCRNet, self).__init__()

        self.features = cnn
        self.n_ways = opt.n_ways
        self.n_shots = opt.n_shots
        self.n_queries = opt.n_queries
        self.metric_layer = ATLModule(opt, inplanes)
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = opt.temperature_p
        self.opt = opt
        self.glosemodule = GSLSAModule(inplanes, self.n_ways, self.n_shots, self.n_queries)

    def forward(self, support_x, support_y, query_x, query_y):
        query_feature = self.features(query_x)
        support_feature = self.features(support_x)
        contrastive_loss, ctc_loss, select_index = self.glosemodule(support_feature)
        support_feature = support_feature * select_index
        scores, local_score = self.metric_layer(query_feature, support_feature)
        scores = torch.mean(scores.view(-1, self.opt.n_ways, self.opt.n_shots), dim=2)
        return contrastive_loss, ctc_loss, scores
