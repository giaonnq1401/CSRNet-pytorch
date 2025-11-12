import torch.nn as nn
import torch
from torchvision import models

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        # Initialize all weights first
        self._initialize_weights()
        # Optionally load VGG16 Frontend weights (skip when load_weights=True)
        if not load_weights:
            try:
                # Newer torchvision API
                mod = models.vgg16(weights=getattr(models, 'VGG16_Weights', None).IMAGENET1K_FEATURES) if hasattr(models, 'VGG16_Weights') else models.vgg16(pretrained=True)
            except Exception:
                # Fallback to deprecated pretrained flag
                mod = models.vgg16(pretrained=True)
            # Copy conv weights from VGG16.features to our frontend conv layers
            vgg_convs = [m for m in mod.features if isinstance(m, nn.Conv2d)]
            fe_convs = [m for m in self.frontend if isinstance(m, nn.Conv2d)]
            for src, dst in zip(vgg_convs, fe_convs):
                if dst.weight.shape == src.weight.shape:
                    dst.weight.data.copy_(src.weight.data)
                    if dst.bias is not None and src.bias is not None:
                        dst.bias.data.copy_(src.bias.data)
    def forward(self,x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
                
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)                