import torch
import torch.nn as nn

class GeneralConv(nn.Module):

    def __init__(self, nin, nout, kernelSize=3, padding=0, dilation=1, maxPoolFlag=True):
        super().__init__()
        self.maxPoolFlag = maxPoolFlag
        self.conv = nn.Conv2d(
            in_channels=nin,
            out_channels=nout,
            kernel_size=kernelSize,
            padding=1,
            dilation=dilation
        )

        self.batchNorm = nn.BatchNorm2d(nout)
        self.relu = nn.ReLU()
        if maxPoolFlag:
            self.maxPool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(x)

        out = self.batchNorm(out)
        out = self.relu(out)
        if self.maxPoolFlag:
            out = self.maxPool(out)
        return out


class DeepWiseConv(nn.Module):

    def __init__(self, nin, nout, kernelSize=3, padding=0, dilation=1, maxPoolFlag=True):
        super().__init__()
        self.maxPoolFlag = maxPoolFlag
        self.deepWiseConv = nn.Conv2d(
            in_channels=nin,
            out_channels=nin,
            kernel_size=kernelSize,
            padding=1,
            groups=nin,
            dilation=dilation
        )
        self.pointConv = nn.Conv2d(
            in_channels=nin,
            out_channels=nout,
            kernel_size=1,
            padding=padding,
            groups=1
        )
        self.batchNorm = nn.BatchNorm2d(nout)
        self.relu = nn.ReLU()
        if maxPoolFlag:
            self.maxPool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.deepWiseConv(x)
        out = self.pointConv(out)
        out = self.batchNorm(out)
        out = self.relu(out)
        if self.maxPoolFlag:
            out = self.maxPool(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.maxPool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3x3 = DeepWiseConv(nin=nin, nout=nout,
                                    kernelSize=3, padding=1)
        self.dilaConv3x3 = DeepWiseConv(nin=nin, nout=nout,
                                        kernelSize=3, padding=2, dilation=2)

    def forward(self, x):
        out1 = self.conv3x3(x)
        # print('out1.shape   ', out1.shape)
        out2 = self.dilaConv3x3(x)
        # print('out2.shape   ', out2.shape)
        out = torch.cat((out1, out2), 1)
        # out = self.maxPool(kernel_size=2, stride=2)
        return out


class Transition(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels=nin,
            out_channels=nout,
            kernel_size=1,
            stride=1,
            padding=0)
        self.maxPool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv1x1(x)
        out = self.maxPool(out)
        return out
