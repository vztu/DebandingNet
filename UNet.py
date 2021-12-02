import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, num_res=2, base_channel=32):
        super(UNet, self).__init__()

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.CatConv = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.feat_up = nn.ModuleList([
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
        ])

        self.ConvOut = BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)

    def forward(self, x):

        # encoder
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        x_ = self.feat_extract[1](res1)
        res2 = self.Encoder[1](x_)

        z = self.feat_extract[2](res2)
        z = self.Encoder[2](z)

        # decoder
        z = self.Decoder[0](z)
        z = self.feat_up[0](z)

        z = torch.cat([z, res2], dim=1)
        z = self.CatConv[0](z)
        z = self.Decoder[1](z)
        z = self.feat_up[1](z)
        
        z = torch.cat([z, res1], dim=1)
        z = self.CatConv[1](z)
        z = self.Decoder[2](z)
        z = self.ConvOut(z)

        outputs = z + x  # residual learning

        return outputs


def model(model_name):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg

    if model_name == "UNet-32":
        # 1.82M, 19.84 GMAC
        return UNet(num_res=2, base_channel=32)
    elif model_name == "UNet-64":
        # 7.27M, 79.01 GMAC
        return UNet(num_res=2, base_channel=64)
    elif model_name == "UNet-96":
        # 30.28M, 308.09 GMAC
        return UNet(num_res=4, base_channel=96)
    raise ModelError('Wrong Model!\nYou should choose MIMO-UNetPlus or MIMO-UNet.')


if __name__ == "__main__":

    model = model("UNet-32")
    model.cuda()

    x = torch.randn(1, 3, 256, 256)
    x = x.cuda()

    outputs = model(x)
    print(outputs.shape)