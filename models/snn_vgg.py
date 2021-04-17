#---------------------------------------------------
# Imports
#---------------------------------------------------
import torch
import torch.nn as nn
import torchvision

from .spikes import *
from .net_utils import AverageMeterNetwork

cfg_features = {
    'vgg5' : [64, 'avg', 128, 128, 'avg'],
    'vgg9':  [64, 64, 'avg', 128, 128, 'avg', 256, 256, 256, 'avg'],
    'vgg11': [64, 'avg', 128, 256, 'avg', 512, 512, 'avg', 512, 'avg', 512, 512],
    'vgg13': [64, 64, 'avg', 128, 128, 'avg', 256, 256, 'avg', 512, 512, 512, 'avg', 512],
    'vgg16': [64, 64, 'avg', 128, 128, 'avg', 256, 256, 256, 'avg', 512, 512, 512, 'avg', 512, 512, 512],
    'vgg19': [64, 64, 'avg', 128, 128, 'avg', 256, 256, 256, 256, 'avg', 512, 512, 512, 512, 'avg', 512, 512, 512, 512]
}

cfg_classifier = {
    'vgg5' : [1024, 1024, 'output'],
    'vgg9':  [1024, 1024, 'output'],
    'vgg11':  [4096, 4096, 'output'],
    'vgg13':  [4096, 4096, 'output'],
    'vgg16':  [4096, 4096, 'output'],
    'vgg19':  [4096, 4096, 'output']
}


class SNN_VGG(nn.Module):

    def __init__(self, config, grad_type='Linear', init='xavier'):
        super(SNN_VGG, self).__init__()

        # Architecture parameters
        self.architecture = config.architecture
        self.dataset = config.dataset
        self.img_size = config.img_size
        self.kernel_size = config.kernel_size
        self.init = init

        # SNN simulation parameters
        self.timesteps = config.timesteps
        self.leak_mem = torch.tensor(config.leak_mem)
        self.def_threshold = config.def_threshold
        self.spike_fn       = init_spike_fn(grad_type)
        self.input_layer 	= PoissonGenerator()

        self.count_spikes = config.count_spikes
        
        self._make_layers()
        self._init_layers()

        if config.pretrained:
            if self.architecture == 'vgg11':
                vgg = torchvision.models.vgg11(pretrained=True)
                state_vgg = vgg.features.state_dict()
                self.features.load_state_dict(state_vgg, strict=False)
            elif self.architecture == 'vgg16':
                vgg = torchvision.models.vgg16(pretrained=True)
                state_vgg = vgg.features.state_dict()
                self.features.load_state_dict(state_vgg, strict=False)

        if self.count_spikes:
            # Make AverageMeterNetwork for measuring spikes
            model_length = len(self.features) + len(self.classifier) - 1
            self.spikes = AverageMeterNetwork(model_length)

            mem_features, mem_classifier = self.init_mems(1)
            for i in range(model_length):
                if i < len(mem_features):
                    layer_shape = mem_features[i].shape
                    neurons = layer_shape[1] * layer_shape[2] * layer_shape[3]
                else:
                    layer_shape = mem_classifier[i-len(mem_features)].shape
                    neurons = layer_shape[1]
                self.spikes.updateUnits(i, neurons)
    
    def _make_layers(self):
        affine_flag = True
        bias_flag = False
        stride = 1
        padding = (self.kernel_size-1)//2

        in_channels = self.dataset['input_dim']
        layer = 0
        divisor = 1
        layers, self.pool_features = [], {}
        
        for x in (cfg_features[self.architecture]):
            if isinstance(x, str) and 'avg' in x:
                self.pool_features[str(layer-1)] = nn.AvgPool2d(kernel_size=2, stride=2)
                divisor *= 2
                continue
            elif isinstance(x, str) and 'max' in x:
                self.pool_features[str(layer-1)] = nn.MaxPool2d(kernel_size=2, stride=2)
                divisor *= 2
                continue
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=padding, stride=stride, bias=bias_flag)]
                in_channels = x
            layer += 1

        self.features = nn.ModuleList(layers)
        self.pool_features = nn.ModuleDict(self.pool_features)
        
        layer = 0
        layers, self.pool_classifier = [], {}

        scale = self.img_size[0]//divisor
        in_channels = in_channels*scale*scale

        for x in (cfg_classifier[self.architecture]):
            if isinstance(x, str) and 'avg' in x:
                self.pool_classifier[str(layer-1)] = nn.AvgPool2d(kernel_size=2, stride=2)
                continue
            elif isinstance(x, str) and 'max' in x:
                self.pool_classifier[str(layer-1)] = nn.MaxPool2d(kernel_size=2, stride=2)
                continue
            elif isinstance(x, str) and x == 'output':
                layers += [nn.Linear(in_channels, self.dataset['num_cls'], bias=bias_flag)]
                break
            else:
                layers += [nn.Linear(in_channels, x, bias=bias_flag)]
                in_channels = x
            layer += 1

        self.classifier = nn.ModuleList(layers)
        self.pool_classifier = nn.ModuleDict(self.pool_classifier)

    def _init_layers(self):
        # Initialize the firing thresholds and weights of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = self.def_threshold

                if self.init == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight, a=1)
                elif self.init == 'xavier':
                    torch.nn.init.xavier_uniform_(m.weight, gain=2)

                if m.bias is not None:
                    m.bias.data.zero_()

            elif (isinstance(m, nn.AvgPool2d)):
                m.threshold = self.def_threshold*0.75

            elif (isinstance(m, nn.Linear)):
                m.threshold = self.def_threshold

                if self.init == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight, a=1)
                elif self.init == 'xavier':
                    torch.nn.init.xavier_uniform_(m.weight, gain=2)

                if m.bias is not None:
                    m.bias.data.zero_()

    def threshold_update(self, scaling_factor=1.0, thresholds=[]):
        # Initialize thresholds
        self.scaling_factor = scaling_factor
        
        for m in self.modules():
            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)) and thresholds:
                v_th = thresholds.pop(0)
                m.threshold = torch.tensor(v_th*self.scaling_factor)

    def init_mems(self, N):
        height = self.img_size[0]
        width = self.img_size[1]

        # Initialize the neuronal membrane potentials
        layers = []
        divisor = 1
        for x in (cfg_features[self.architecture]):
            if isinstance(x, str) and ('avg' in x or 'max' in x):
                divisor *= 2
                continue
            elif not isinstance(x, int):
                continue
            layers += [torch.zeros(N, x, height//divisor, width//divisor).cuda()]
        mem_features = layers

        layers = []
        for x in (cfg_classifier[self.architecture]):
            if isinstance(x, str) and x == 'output':
                x = self.dataset['num_cls']
            elif not isinstance(x, int):
                continue
            layers += [torch.zeros(N, x).cuda()]
        mem_classifier = layers

        return mem_features, mem_classifier

    def forward(self, x, find_max_mem=False, max_mem_layer=0):
        N, C, H, W = x.size()
        # print('input size', N, C, H, W)

        if self.count_spikes:
            self.spikes.updateCount(N)

        mem_features, mem_classifier = self.init_mems(N)

        max_mem = 0.0

        for t in range(self.timesteps):
            out_prev = self.input_layer(x)

            for k in range(len(self.features)):

                if find_max_mem and k == max_mem_layer:
                    ts_max = (self.features[k](out_prev)).max()
                    max_mem = ts_max if ts_max > max_mem else max_mem
                    break

                # Compute the conv outputs
                mem_features[k]     = (self.leak_mem * mem_features[k] + (self.features[k](out_prev)))
                mem_thr             = (mem_features[k]/self.features[k].threshold) - 1.0
                out                 = self.spike_fn(mem_thr)
                rst                 = torch.zeros_like(mem_features[k]).cuda()
                rst                 = (mem_thr > 0) * self.features[k].threshold
                mem_features[k]     = mem_features[k] - rst
                out_prev            = out.clone()

                if self.count_spikes:
                    self.spikes.updateSum(k, torch.sum(out.detach().clone()).item())

                if str(k) in self.pool_features.keys():
                    out = self.pool_features[str(k)](out_prev)
                    out_prev = out.clone()

            if find_max_mem and max_mem_layer < len(self.features):
                continue

            out_prev = out_prev.reshape(N, -1)
            prev = len(self.features)

            for k in range(len(self.classifier) - 1):
                
                if find_max_mem and (prev + k) == max_mem_layer:
                    ts_max = (self.classifier[k](out_prev)).max()
                    max_mem = ts_max if ts_max > max_mem else max_mem
                    break

                mem_classifier[k]   = (self.leak_mem * mem_classifier[k] + (self.classifier[k](out_prev)))
                mem_thr             = (mem_classifier[k]/self.classifier[k].threshold) - 1.0
                out                 = self.spike_fn(mem_thr)
                rst                 = torch.zeros_like(mem_classifier[k]).cuda()
                rst                 = (mem_thr > 0) * self.classifier[k].threshold
                mem_classifier[k]   = mem_classifier[k] - rst
                out_prev            = out.clone()

                if self.count_spikes:
                    self.spikes.updateSum((prev+k), torch.sum(out.detach().clone()).item())

                if str(k) in self.pool_classifier.keys():
                    out = self.pool_classifier[str(k)](out_prev)
                    out_prev = out.clone()

            # compute last conv
            if not find_max_mem:
                mem_classifier[k+1] = (1 * mem_classifier[k+1] + self.classifier[k+1](out_prev))
            
            # print('timesteps: {}/{}'.format(t, total_timesteps), end='\r', flush=True)

        if find_max_mem:
            return max_mem

        out_voltage = mem_classifier[k+1]
        out_voltage = (out_voltage) / self.timesteps

        return out_voltage