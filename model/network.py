import torch
import torch.nn as nn
from functools import partial

from model.backbone.resnext.resnext101_regular import ResNeXt101
from model.backbone.efficientnet_pytorch.model import EfficientNet
from model.backbone.convnext import convnext_small, convnext_base
from model.backbone.PVT.pvt_v2 import PyramidVisionTransformerV2


class LayerConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, relu):
        super(LayerConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)

        return x


class Weighted_BCE_Loss(nn.Module):
    def __init__(self, mu=0.5, eps=1e-5):
        super(Weighted_BCE_Loss, self).__init__()
        self.mu = mu
        self.eps = eps

    def forward(self, result, gts):
        map = self.mu * gts * torch.log(result + self.eps) + (1 - self.mu) * (1 - gts) * torch.log(
            1 - result + self.eps)
        Loss = - torch.sum(map)
        return Loss


class Network(nn.Module):
    def __init__(self,
                 input_resolution,
                 backbone='ResNeXt',
                 backbone_ckpt=None
                 ):

        super(Network, self).__init__()
        self.input_resolution = input_resolution
        self.backbone = backbone

        if backbone == 'ResNeXt':
            resnext = ResNeXt101(backbone_path=backbone_ckpt)
            self.layer0 = resnext.layer0
            self.layer1 = resnext.layer1
            self.layer2 = resnext.layer2
            self.layer3 = resnext.layer3
            self.layer4 = resnext.layer4

            self.layer4_transposed_conv = nn.ConvTranspose2d(2048, 1024, 2, 2)
            self.layer3_transposed_conv = nn.ConvTranspose2d(1024, 512, 2, 2)
            self.layer2_transposed_conv = nn.ConvTranspose2d(512, 256, 2, 2)
            self.layer1_transposed_conv = nn.ConvTranspose2d(256, 64, 2, 2)

            self.relu = nn.ReLU()

            self.cat_43 = nn.Sequential(
                nn.Conv2d(2048, 1024, 3, 1, 1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.Conv2d(1024, 1024, 3, 1, 1),
                nn.BatchNorm2d(1024),
                nn.ReLU()
            )
            self.cat_32 = nn.Sequential(
                nn.Conv2d(1024, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU()
            )
            self.cat_21 = nn.Sequential(
                nn.Conv2d(512, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
            self.cat_10 = nn.Sequential(
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )

            self.output = nn.Sequential(
                nn.ConvTranspose2d(64, 64, 2, 2),
                nn.Conv2d(64, 1, 1, 1),
                nn.Sigmoid()
            )

        elif backbone.split('-')[0] == 'efficientnet':
            net = EfficientNet.from_pretrained(backbone, in_channels=3, weights_path=backbone_ckpt)
            self.drop_connect_rate = net._global_params.drop_connect_rate

            if backbone == 'efficientnet-b3':
                self.encoder_idx = [2, 5, 8, 18, 26]
                self.decoder_dim = [24, 32, 48, 136, 1536]
            elif backbone == 'efficientnet-b7':
                self.encoder_idx = [4, 11, 18, 38, 55]
                self.decoder_dim = [32, 48, 80, 224, 2560]
            elif backbone == 'efficientnet-b8':
                self.encoder_idx = [4, 12, 20, 42, 61]
                self.decoder_dim = [32, 56, 88, 248, 2816]

            self.layer0_a = nn.Sequential(
                net._conv_stem,
                net._bn0,
                net._swish
            )
            self.layer0_b = nn.Sequential(
                *net._blocks[0:self.encoder_idx[0]]
            )
            self.layer1 = nn.Sequential(
                *net._blocks[self.encoder_idx[0]:self.encoder_idx[1]]
            )
            self.layer2 = nn.Sequential(
                *net._blocks[self.encoder_idx[1]:self.encoder_idx[2]]
            )
            self.layer3 = nn.Sequential(
                *net._blocks[self.encoder_idx[2]:self.encoder_idx[3]]
            )
            self.layer4_a = nn.Sequential(
                *net._blocks[self.encoder_idx[3]:self.encoder_idx[4]]
            )
            self.layer4_b = nn.Sequential(
                net._conv_head,
                net._bn1,
                net._swish
            )

            self.layer4_transposed_conv = nn.ConvTranspose2d(self.decoder_dim[4], self.decoder_dim[3], 2, 2)
            self.layer3_transposed_conv = nn.ConvTranspose2d(self.decoder_dim[3], self.decoder_dim[2], 2, 2)
            self.layer2_transposed_conv = nn.ConvTranspose2d(self.decoder_dim[2], self.decoder_dim[1], 2, 2)
            self.layer1_transposed_conv = nn.ConvTranspose2d(self.decoder_dim[1], self.decoder_dim[0], 2, 2)

            self.relu = nn.ReLU()

            self.cat_43 = nn.Sequential(
                nn.Conv2d(self.decoder_dim[3] * 2, self.decoder_dim[3], 3, 1, 1),
                nn.BatchNorm2d(self.decoder_dim[3]),
                nn.ReLU(),
                nn.Conv2d(self.decoder_dim[3], self.decoder_dim[3], 3, 1, 1),
                nn.BatchNorm2d(self.decoder_dim[3]),
                nn.ReLU()
            )
            self.cat_32 = nn.Sequential(
                nn.Conv2d(self.decoder_dim[2] * 2, self.decoder_dim[2], 3, 1, 1),
                nn.BatchNorm2d(self.decoder_dim[2]),
                nn.ReLU(),
                nn.Conv2d(self.decoder_dim[2], self.decoder_dim[2], 3, 1, 1),
                nn.BatchNorm2d(self.decoder_dim[2]),
                nn.ReLU()
            )
            self.cat_21 = nn.Sequential(
                nn.Conv2d(self.decoder_dim[1] * 2, self.decoder_dim[1], 3, 1, 1),
                nn.BatchNorm2d(self.decoder_dim[1]),
                nn.ReLU(),
                nn.Conv2d(self.decoder_dim[1], self.decoder_dim[1], 3, 1, 1),
                nn.BatchNorm2d(self.decoder_dim[1]),
                nn.ReLU()
            )
            self.cat_10 = nn.Sequential(
                nn.Conv2d(self.decoder_dim[0] * 2, self.decoder_dim[0], 3, 1, 1),
                nn.BatchNorm2d(self.decoder_dim[0]),
                nn.ReLU(),
                nn.Conv2d(self.decoder_dim[0], self.decoder_dim[0], 3, 1, 1),
                nn.BatchNorm2d(self.decoder_dim[0]),
                nn.ReLU()
            )

            self.output = nn.Sequential(
                nn.ConvTranspose2d(self.decoder_dim[0], self.decoder_dim[0], 2, 2),
                nn.Conv2d(self.decoder_dim[0], 1, 1, 1),
                nn.Sigmoid()
            )

        elif backbone == 'convnext-small':
            self.encoder = convnext_small(pretrained=True, in_22k=False)

            self.layer3_transposed_conv = nn.ConvTranspose2d(768, 384, 2, 2)
            self.layer2_transposed_conv = nn.ConvTranspose2d(384, 192, 2, 2)
            self.layer1_transposed_conv = nn.ConvTranspose2d(192, 96, 2, 2)

            self.relu = nn.ReLU()

            self.cat_32 = nn.Sequential(
                nn.Conv2d(768, 384, 3, 1, 1),
                nn.BatchNorm2d(384),
                nn.ReLU(),
                nn.Conv2d(384, 384, 3, 1, 1),
                nn.BatchNorm2d(384),
                nn.ReLU()
            )
            self.cat_21 = nn.Sequential(
                nn.Conv2d(384, 192, 3, 1, 1),
                nn.BatchNorm2d(192),
                nn.ReLU(),
                nn.Conv2d(192, 192, 3, 1, 1),
                nn.BatchNorm2d(192),
                nn.ReLU()
            )
            self.cat_10 = nn.Sequential(
                nn.Conv2d(192, 96, 3, 1, 1),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.Conv2d(96, 96, 3, 1, 1),
                nn.BatchNorm2d(96),
                nn.ReLU()
            )

            self.output = nn.Sequential(
                nn.ConvTranspose2d(96, 48, 2, 2),
                nn.ConvTranspose2d(48, 24, 2, 2),
                nn.Conv2d(24, 1, 1, 1),
                nn.Sigmoid()
            )

        elif backbone == 'convnext-base':
            self.encoder = convnext_base(pretrained=True, in_22k=False)

            self.layer3_transposed_conv = nn.ConvTranspose2d(1024, 512, 2, 2)
            self.layer2_transposed_conv = nn.ConvTranspose2d(512, 256, 2, 2)
            self.layer1_transposed_conv = nn.ConvTranspose2d(256, 128, 2, 2)

            self.relu = nn.ReLU()

            self.cat_32 = nn.Sequential(
                nn.Conv2d(1024, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU()
            )
            self.cat_21 = nn.Sequential(
                nn.Conv2d(512, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
            self.cat_10 = nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )

            self.output = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 2, 2),
                nn.ConvTranspose2d(64, 32, 2, 2),
                nn.Conv2d(32, 1, 1, 1),
                nn.Sigmoid()
            )

        elif backbone.split('-')[0] == 'PVT':
            embed_dims = [64, 128, 320, 512]

            if backbone == 'PVT-b3':
                self.encoder = PyramidVisionTransformerV2(
                    patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1])
            elif backbone == 'PVT-b5':
                self.encoder = PyramidVisionTransformerV2(
                    patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1])
            else:
                ValueError('wrong pvt backbone')

            self.encoder.load_state_dict(torch.load(backbone_ckpt))

            self.layer3_transposed_conv = nn.ConvTranspose2d(embed_dims[3], embed_dims[2], 2, 2)
            self.layer2_transposed_conv = nn.ConvTranspose2d(embed_dims[2], embed_dims[1], 2, 2)
            self.layer1_transposed_conv = nn.ConvTranspose2d(embed_dims[1], embed_dims[0], 2, 2)

            self.cat_32 = nn.Sequential(
                nn.Conv2d(2 * embed_dims[2], embed_dims[2], 3, 1, 1),
                nn.BatchNorm2d(embed_dims[2]),
                nn.ReLU(),
                nn.Conv2d(embed_dims[2], embed_dims[2], 3, 1, 1),
                nn.BatchNorm2d(embed_dims[2]),
                nn.ReLU()
            )

            self.cat_21 = nn.Sequential(
                nn.Conv2d(2 * embed_dims[1], embed_dims[1], 3, 1, 1),
                nn.BatchNorm2d(embed_dims[1]),
                nn.ReLU(),
                nn.Conv2d(embed_dims[1], embed_dims[1], 3, 1, 1),
                nn.BatchNorm2d(embed_dims[1]),
                nn.ReLU()
            )

            self.cat_10 = nn.Sequential(
                nn.Conv2d(2 * embed_dims[0], embed_dims[0], 3, 1, 1),
                nn.BatchNorm2d(embed_dims[0]),
                nn.ReLU(),
                nn.Conv2d(embed_dims[0], embed_dims[0], 3, 1, 1),
                nn.BatchNorm2d(embed_dims[0]),
                nn.ReLU()
            )

            self.output = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 2, 2),
                nn.ConvTranspose2d(32, 16, 2, 2),
                nn.Conv2d(16, 1, 1, 1),
                nn.Sigmoid()
            )

        else:
            ValueError('wrong backbone')

    def forward(self, x):
        if self.backbone == 'ResNeXt':
            layer0 = self.layer0(x)
            layer1 = self.layer1(layer0)
            layer2 = self.layer2(layer1)
            layer3 = self.layer3(layer2)
            layer4 = self.layer4(layer3)

            layer4_up = self.layer4_transposed_conv(layer4)
            layer3_up = self.layer3_transposed_conv(self.cat_43(torch.cat((layer3, layer4_up), dim=1)))
            layer2_up = self.layer2_transposed_conv(self.cat_32(torch.cat((layer2, layer3_up), dim=1)))
            layer1_up = self.layer1_transposed_conv(self.cat_21(torch.cat((layer1, layer2_up), dim=1)))
            layer0_up = self.cat_10(torch.cat((layer0, layer1_up), dim=1))

            output = self.output(layer0_up)

            return output

        elif self.backbone.split('-')[0] == 'efficientnet':
            layer0 = self.layer0_a(x)
            for idx, block in enumerate(self.layer0_b):
                drop_connect_rate = self.drop_connect_rate
                drop_connect_rate *= (float(idx) + 0) / self.encoder_idx[-1]
                layer0 = block(layer0, drop_connect_rate=drop_connect_rate)

            layer1 = layer0
            for idx, block in enumerate(self.layer1):
                drop_connect_rate = self.drop_connect_rate
                drop_connect_rate *= (float(idx) + self.encoder_idx[0]) / self.encoder_idx[-1]
                layer1 = block(layer1, drop_connect_rate=drop_connect_rate)

            layer2 = layer1
            for idx, block in enumerate(self.layer2):
                drop_connect_rate = self.drop_connect_rate
                drop_connect_rate *= (float(idx) + self.encoder_idx[1]) / self.encoder_idx[-1]
                layer2 = block(layer2, drop_connect_rate=drop_connect_rate)

            layer3 = layer2
            for idx, block in enumerate(self.layer3):
                drop_connect_rate = self.drop_connect_rate
                drop_connect_rate *= (float(idx) + self.encoder_idx[2]) / self.encoder_idx[-1]
                layer3 = block(layer3, drop_connect_rate=drop_connect_rate)

            layer4 = layer3
            for idx, block in enumerate(self.layer4_a):
                drop_connect_rate = self.drop_connect_rate
                drop_connect_rate *= (float(idx) + self.encoder_idx[3]) / self.encoder_idx[-1]
                layer4 = block(layer4, drop_connect_rate=drop_connect_rate)
            layer4 = self.layer4_b(layer4)

            layer4_up = self.layer4_transposed_conv(layer4)
            layer3_up = self.layer3_transposed_conv(self.cat_43(torch.cat((layer3, layer4_up), dim=1)))
            layer2_up = self.layer2_transposed_conv(self.cat_32(torch.cat((layer2, layer3_up), dim=1)))
            layer1_up = self.layer1_transposed_conv(self.cat_21(torch.cat((layer1, layer2_up), dim=1)))
            layer0_up = self.cat_10(torch.cat((layer0, layer1_up), dim=1))

            output = self.output(layer0_up)

            return output

        elif self.backbone == 'convnext-small' or self.backbone == 'convnext-base':

            layer0 = self.encoder.downsample_layers[0](x)
            layer0 = self.encoder.stages[0](layer0)

            layer1 = self.encoder.downsample_layers[1](layer0)
            layer1 = self.encoder.stages[1](layer1)

            layer2 = self.encoder.downsample_layers[2](layer1)
            layer2 = self.encoder.stages[2](layer2)

            layer3 = self.encoder.downsample_layers[3](layer2)
            layer3 = self.encoder.stages[3](layer3)

            layer3_up = self.layer3_transposed_conv(layer3)
            layer2_up = self.layer2_transposed_conv(self.cat_32(torch.cat((layer2, layer3_up), dim=1)))
            layer1_up = self.layer1_transposed_conv(self.cat_21(torch.cat((layer1, layer2_up), dim=1)))
            layer0_up = self.cat_10(torch.cat((layer0, layer1_up), dim=1))

            output = self.output(layer0_up)

            return output

        elif self.backbone.split('-')[0] == 'PVT':
            layer0, layer1, layer2, layer3 = self.encoder.forward_features(x)

            layer3_up = self.layer3_transposed_conv(layer3)
            layer2_up = self.layer2_transposed_conv(self.cat_32(torch.cat((layer2, layer3_up), dim=1)))
            layer1_up = self.layer1_transposed_conv(self.cat_21(torch.cat((layer1, layer2_up), dim=1)))
            layer0_up = self.cat_10(torch.cat((layer0, layer1_up), dim=1))

            output = self.output(layer0_up)

            return output
        else:
            ValueError('wrong backbone')

