#---------------------------------------------------
# Imports
#---------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# TODO: figure out architecture

cfg_features = {
    'vgg5' : [64, 'avg', 128, 128, 'avg'],
    'vgg9':  [64, 'avg', 128, 256, 'avg', 256, 512, 'avg', 512, 'avg', 512],
    'vgg11': [64, 'avg', 128, 256, 'avg', 512, 512, 'avg', 512, 'avg', 512, 512],
    'vgg13': [64, 64, 'avg', 128, 128, 'avg', 256, 256, 'avg', 512, 512, 512, 'avg', 512],
    'vgg16': [64, 64, 'avg', 128, 128, 'avg', 256, 256, 256, 'avg', 512, 512, 512, 'avg', 512, 512, 512],
    'vgg19': [64, 64, 'avg', 128, 128, 'avg', 256, 256, 256, 256, 'avg', 512, 512, 512, 512, 'avg', 512, 512, 512, 512]
}

cfg_classifier = {
    'vgg5' : [1024, 1024, 'output'],
    'vgg9':  [4096, 4096, 'output'],
}


class ANN_VGG(nn.Module):

    def __init__(self, config, init='xavier'):
        super(ANN_VGG, self).__init__()

        # Architecture parameters
        self.architecture = config.architecture
        self.bntt = config.bn
        self.img_size = config.img_size
        self.kernel_size = config.kernel_size
        self.dataset = config.dataset
        
        self._make_layers(cfg[self.architecture])
        
        self._init_layers(init)

        if config.pretrained:
            if self.architecture == 'vgg11':
                vgg = torchvision.models.vgg11(pretrained=True)
                state_vgg = vgg.features.state_dict()
                self.features.load_state_dict(state_vgg, strict=False)
            elif self.architecture == 'vgg16':
                vgg = torchvision.models.vgg16(pretrained=True)
                state_vgg = vgg.features.state_dict()
                self.features.load_state_dict(state_vgg, strict=False)

    
    def _make_layers(self, cfg):
        affine_flag = True
        bias_flag = False
        stride = 1
        padding = (self.ksize-1)//2

        in_channels = self.dataset.input_dim
        layer = 0
        divisor = 1
        layers, self.pool_features, relu_layers = [], {}, []
        
        for x in (cfg_features[self.architecture]):
            if isinstance(x, str) and 'avg' in x:
                self.pool_features[str(layer-1)] = nn.AvgPool2d(kernel_size=2, stride=2)
                divisor *= 2
            elif isinstance(x, str) and 'max' in x:
                self.pool_features[str(layer-1)] = nn.MaxPool2d(kernel_size=2, stride=2)
                divisor *= 2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=self.ksize, padding=padding, stride=stride, bias=bias_flag)]
                relu_layers += [nn.ReLU(inplace=True)]
                in_channels = x
                layer += 1

        self.features = nn.ModuleList(layers)
        self.pool_features = nn.ModuleDict(self.pool_features)
        self.relu_features = nn.ModuleList(relu_layers)
        
        layer = 0
        layers, self.pool_classifier, relu_layers = [], {}, []

        scale = self.img_size[0]//divisor
        in_channels = in_channels*scale*scale

        for x in (cfg_classifier[self.architecture]):
            if isinstance(x, str) and 'avg' in x:
                self.pool_classifier[str(layer-1)] = nn.AvgPool2d(kernel_size=2, stride=2)
            elif isinstance(x, str) and 'max' in x:
                self.pool_classifier[str(layer-1)] = nn.MaxPool2d(kernel_size=2, stride=2)
            elif isinstance(x, str) and x == 'output':
                layers += [nn.Linear(in_channels, self.dataset.num_cls, bias=bias_flag)]
                break
            else:
                layers += [nn.Linear(in_channels, x, bias=bias_flag)]
                relu_layers += [nn.ReLU(inplace=True)]
                in_channels = x
                layer += 1

        self.classifier = nn.ModuleList(layers)
        self.pool_classifier = nn.ModuleDict(self.pool_classifier)
        self.relu_classifier = nn.ModuleList(relu_layers)

    def _init_layers(self, init):
        # Initialize the firing thresholds and weights of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                if init == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif init == 'xavier':
                    torch.nn.init.xavier_uniform_(m.weight, gain=2)

                if m.bias is not None:
                    m.bias.data.zero_()

            elif (isinstance(m, nn.Linear)):
                if init == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif init == 'xavier':
                    torch.nn.init.xavier_uniform_(m.weight, gain=2)

                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = x

        for k in range(len(self.features)):
            out = self.features[k](out)
            out = self.relu_features[k](out)

            if str(k) in self.pool_features.keys():
                out = self.pool_features[str(k)](out)

        for k in range(len(self.classifier) - 1):
            out = self.classifier[k](out)
            out = self.relu_classifier[k](out)

            if str(k) in self.pool_classifier.keys():
                out = self.pool_classifier[str(k)](out)

        out = self.classifier[k+1](out)

        return out