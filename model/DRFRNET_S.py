import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, dilation=1):
        super(ConvX, self).__init__()
        if dilation == 1:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, bias=False)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, dilation=dilation,
                                  padding=dilation, bias=False)
        self.bn = BatchNorm2d(out_planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.keras_init_weight()

    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                # nn.init.xavier_normal_(ly.weight,gain=nn.init.calculate_gain('relu'))
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    # 对比着resnet的residual block看很容易理解
    expansion = 1  # 残差块的扩张因子

    # 为什么每个残差快内都有一个扩展因子  这个扩展因子用于控制残差快内部的卷积层输出通道树相对于输入通道数的比例
    # 当stride!=1时 也就是说有下采样的时候(缩小特征图尺寸) 为了使short能和主路径上的特征图维度能够对齐  需要通过调扩展因子 来保证通道数一致

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # 是否在用一次relu函数
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                                size=[height, width],
                                mode='bilinear')

        return out


class DualResNet(nn.Module):
    # model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=19, planes=32, spp_planes=128, head_planes=64, augment=True)
    def __init__(self, block, layers, num_classes=19, planes=64, spp_planes=128, head_planes=128, augment=True):
        super(DualResNet, self).__init__()
        highres_planes = planes * 2
        self.augment = augment

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=False)  # inplace 表示是否将计算结果直接覆盖到输入变量中 从而节省内存消耗
        # True时会直接修改输入变量的值 而不是创建一个新的变量来存储结果  False则是返回一个新的张量作为输出
        # 只有layer1没有下采样 其余都进行了下采样
        self.layer1 = self._make_layer(block, planes, planes, layers[0])  # layers里面存放着block重复几次
        self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes * 2, planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes * 4, planes * 8, layers[3], stride=2)

        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, highres_planes, kernel_size=1, bias=False),
            BatchNorm2d(highres_planes, momentum=bn_mom),
        )

        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, highres_planes, kernel_size=1, bias=False),
            BatchNorm2d(highres_planes, momentum=bn_mom),
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(planes * 4, momentum=bn_mom),
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(planes * 4, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(planes * 8, momentum=bn_mom),
        )

        self.layer3_ = self._make_layer(block, planes * 2, highres_planes, 2)

        self.layer4_ = self._make_layer(block, highres_planes, highres_planes, 2)

        self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1)

        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)

        if self.augment:
            self.seghead_extra = segmenthead(highres_planes, head_planes, num_classes)

        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # 用于构建ResNet中的一个残差块
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        # 如果步长不是1 并且输入的通道数不等于    planes * block.expansion表示当前残差块最后一个卷积层期望输出的通道数  planes表示每个残差快中最后一个卷积层的输出通道数
        # 在 ResNet 中，每个残差块的输入通道数 inplanes 应该等于输出通道数 planes 乘以 block.expansion
        if stride != 1 or inplanes != planes * block.expansion:
            # 步长等于1不会发生下采样   下采样操作包括一个卷积层一个批归一化层
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )
        # 开始构建多个残差快
        layers = []  # 空列表存储块
        # 添加的就是前面的basicblock
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))  # 最后一个采用relu
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self, x):

        # 计算输出数据的宽度和高度
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        # width_output: 128  原图是1024*1024的
        # height_output: 128
        layers = []
        x = self.conv1(x)
        # connv1x
        # torch.Size([8, 32, 256, 256])

        x = self.layer1(x)
        # layer1x
        # torch.Size([8, 32, 256, 256])
        layers.append(x)  # layers[0]
        x = self.layer2(self.relu(x))
        # layer2x
        # torch.Size([8, 64, 128, 128])
        layers.append(x)  # layers[1]

        x = self.layer3(self.relu(x))
        # layer3x
        # torch.Size([8, 128, 64, 64])
        layers.append(x)  # layers[2]
        x_ = self.layer3_(self.relu(layers[1]))  # 开始构造第二条分支
        x_ = self.dp1(self.relu(x_))
        # layer3x_
        # torch.Size([8, 64, 128, 128])

        # 开始第一次融合
        x = x + self.down3(self.relu(x_))  # 上到下采样
        x_ = x_ + F.interpolate(
            self.compression3(self.relu(layers[2])),
            size=[height_output, width_output],
            mode='bilinear')  # 下到上采样
        # high_to_low_x
        # torch.Size([8, 128, 64, 64])
        # low_to_hith_x_
        # torch.Size([8, 64, 128, 128])
        if self.augment:
            temp = x_

        x = self.layer4(self.relu(x))  # layer4 的输入是128*64*64
        layers.append(x)
        x_ = self.layer4_(self.relu(x_))
        x_ = self.dp2(self.relu(x_))
        # layer4x
        # torch.Size([8, 256, 32, 32])
        # layer4x_
        # torch.Size([8, 256, 32, 32])
        # 开始第二次融合
        x = x + self.down4(self.relu(x_))  # 上到下采样
        x_ = x_ + F.interpolate(
            self.compression4(self.relu(layers[3])),
            size=[height_output, width_output],
            mode='bilinear')  # 下到上采样
        # high_to_low_x
        # torch.Size([8, 256, 32, 32])
        # low_to_hith_x_
        # torch.Size([8, 64, 128, 128])
        x_ = self.layer5_(self.relu(x_))
        # layer5x_
        # torch.Size([8, 128, 128, 128])

        x = F.interpolate(
            self.spaspp(self.layer5(self.relu(x))),  # layer5 要求输入256 32 32
            size=[height_output, width_output],
            mode='bilinear')
        x = self.sfc(x_, x)
        x_ = self.final_layer(x + x_)

        if self.augment:
            x_extra = self.seghead_extra(temp)
            return [x_extra, x_]
        else:
            return x_
        # 创建一个在imagenet上预训练的model并返回


def DualResNet_imagenet(pretrained=False):
    model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=19, planes=32, spp_planes=128, head_planes=64,
                       augment=True)
    # 模型使用的基础快类型   每个stage中堆叠的基本快数量  输出类别数  初始通道数   用于spp模块的通道数  模型头部的通道数  是否使用数据增强
    # if pretrained:
    #     pretrained_state = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
    #     # 加载预训练权重
    #     model_dict = model.state_dict()
    #     # 获取模型当前参数字典
    #     pretrained_state = {k: v for k, v in pretrained_state.items() if
    #                         (k in model_dict and v.shape == model_dict[k].shape)}
    #     # 通过循环 将预训练权重字典  中与当前模型参数形状匹配的部分筛选出来
    #     model_dict.update(pretrained_state)
    #     # 将删选后的权重更新到当前模型参数字典中
    #
    #     model.load_state_dict(model_dict, strict=False)  # 加载更新后的参数字典  strict表示允许加载不严格匹配的参数
    return model


def get_seg_model_s(**kwargs):
    model = DualResNet_imagenet(pretrained=False)
    return model


if __name__ == '__main__':
    x = torch.rand(4, 3, 1024, 1024)
    net = DualResNet_imagenet()
    y = net(x)
    print(y.shape)

