import torch.nn.functional as F
from torch import nn
from torchvision.models import vgg19

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True).features
        self.layers = nn.Sequential(*list(vgg.children())[:35]).eval()
        for param in self.layers.parameters():
            param.requires_grad = False
        # 加载VGG19的预训练网络 并提取其前35层的特征层用于计算感知损失
        # 同时冻结这些层 防止训练过程中这些层被错误更新

    def forward(self, output, target):
        output_features = self.layers(output)
        target_features = self.layers(target)
        return F.mse_loss(output_features, target_features)
    # 提取生成图像和真是图像的高层次特征并计算损失


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
        )
        # 构建残差块

    def forward(self, x):
        return x + self.block(x)
        # 进行残差连接

class Generator(nn.Module):
    def __init__(self, num_residual_blocks = 16):
        super(Generator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        # 初始卷积层

        residual_block = []
        for _ in range(num_residual_blocks):
            residual_block.append(ResidualBlock(64))

        self.residual_blocks = nn.Sequential(*residual_block)
        # 通过循环创建多个残差块 并将它们组合成一个顺序执行的模块
        # 用于在生成器网络中进行深层次的特征提取

        self.upscale = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )
        # 将图像通道数和长宽进行一次映射增加通道数
        # 随后通过上采样 将每四个通道的信息分别整合到[0,0],[0,1],[1,0],[1,1]的像素位置
        # 并扩大两倍分辨率

        self.final = nn.Conv2d(64,3, kernel_size=9, stride=1, padding=4)
        # 末尾卷积层 映射回一个RGB图像

    def forward(self, x):
        x = self.initial(x)
        residual = x
        x = self.residual_blocks(x)
        x = self.upscale(residual + x)
        x = self.final(x)
        return x
        # 将图像传入初始卷积层 随后保存到residual以备后续残差连接使用
        # 经过16个残差块的处理加强特征 最后再与最初保存的residual特征进行最终残差连接
        # 并通过上采样放大分辨率 并输出图像
