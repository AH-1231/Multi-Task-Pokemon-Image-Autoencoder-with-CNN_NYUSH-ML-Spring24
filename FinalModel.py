import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个标准的残差模块（ResBlock）
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        # 残差连接：输出 = 激活(x + 子模块输出)
        return self.activation(x + self.block(x))

# 定义整个模型结构
class Model(nn.Module):
    def __init__(self, input_channels=3, latent_channels=8, num_classes=170):
        super().__init__()
        self.latent_channels = latent_channels  # 表示量化后的通道数

        # 编码器部分（下采样 + 残差块）
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),  # H -> H/2
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            ResBlock(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # H/2 -> H/4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            ResBlock(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 不改变分辨率
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            ResBlock(256)
        )

        # 编码后进一步处理，用于量化之前
        self.pre_quant_resblock = ResBlock(256)
        self.quant_conv = nn.Conv2d(256, latent_channels, kernel_size=1)  # 压缩通道数为 latent_channels

        # 解码前通道恢复
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, kernel_size=1)
        self.post_quant_resblock = ResBlock(256)

        # 解码器部分（上采样 + 残差块）
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  # 保持尺寸
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            ResBlock(128),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # H/4 -> H/2
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            ResBlock(64),

            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),  # H/2 -> H
            nn.Sigmoid()  # 输出范围压到 [0,1]，适用于图像重建
        )

        # 分类器部分（作用于z）
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 平均池化到 1x1 特征图
            nn.Flatten(),  # 展平成向量
            nn.Dropout(0.5),
            nn.Linear(latent_channels, num_classes)  # 输出为类别数
        )

    # 编码函数：从原始图像提取特征并映射到潜在空间z
    def encode(self, x):
        feat = self.encoder(x)
        feat = self.pre_quant_resblock(feat)
        z = self.quant_conv(feat)  # 得到潜在编码 z
        return z

    # 解码函数：将潜在编码z重建为图像
    def decode(self, z):
        feat = self.post_quant_conv(z)
        feat = self.post_quant_resblock(feat)
        recon = self.decoder(feat)
        return recon

    # 前向传播：返回重建图像、分类结果、潜在向量
    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        logits = self.classifier(z)
        return recon, logits, z