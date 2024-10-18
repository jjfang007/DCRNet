import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.resnet import ResNet
from backbone.cnn import CNNNet

class Projection(nn.Module):
    """
    projection head
    """

    def __init__(self, dim, projection_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size, bias=False),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size, bias=False))

    def forward(self, x):
        return self.net(x)

class Network(nn.Module):

    def __init__(self, args, mode='meta', train_way='classification'):
        super().__init__()

        self.mode = mode
        self.train_way = train_way
        self.args = args
        if args.convnet_name == 'resnet12':
            self.encoder = ResNet(args=args)
        elif args.convnet_name == 'conv64f':
            self.encoder = CNNNet()
        else:
            raise RuntimeError

        if self.mode == 'pre_train':
            if train_way == 'contrastive':
                self.head = Projection(dim=args.inplanes, projection_size=args.inplanes//4, hidden_size=args.inplanes//2)
            elif train_way == 'classification':
                self.fc = nn.Linear(args.inplanes, self.args.num_class)

    def forward(self, input):
        if self.mode == 'meta_train':
            return self.meta_train_forward(input)

        elif self.mode == 'pre_train':
            return self.pre_train_forward(input)
        else:
            raise ValueError('Unknown mode')

    def meta_train_forward(self, x):
        x = self.encoder(x)
        return x

    def pre_train_forward(self, input):
        x = self.encoder(input)
        x = F.adaptive_avg_pool2d(x, 1)
        if self.train_way == 'contrastive':
            return F.normalize(self.head(x.squeeze(-1).squeeze(-1)),dim=-1)
        elif self.train_way == 'classification':
            return self.fc(x.squeeze(-1).squeeze(-1))
        else:
            raise ValueError('Unknown train way')