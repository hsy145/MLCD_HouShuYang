"""
Airbench96 - 高精度CIFAR-10网络 (96%+ accuracy)
基于 airbench96_faster.py 提取
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    
    def forward(self, x):
        return x * self.scale


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-12, weight=False, bias=True):
        super().__init__(num_features, eps=eps)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])


class ConvGroup(nn.Module):
    """卷积组，支持depth=2或3"""
    def __init__(self, channels_in, channels_out, depth=3):
        super().__init__()
        assert depth in (2, 3)
        self.depth = depth
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        if depth == 3:
            self.conv3 = Conv(channels_out, channels_out)
            self.norm3 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        if self.depth == 3:
            x0 = x
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        if self.depth == 3:
            x = self.conv3(x)
            x = self.norm3(x)
            x = x + x0  # 残差连接
            x = self.activ(x)
        return x


class CifarNet96(nn.Module):
    """
    高精度CIFAR-10网络 (96%+ accuracy)
    比CifarNet更宽更深
    """
    def __init__(self, widths=None, depth=3, scaling_factor=1/9):
        super().__init__()
        if widths is None:
            widths = {'block1': 128, 'block2': 384, 'block3': 512}
        
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size**2
        
        self.whiten = Conv(3, whiten_width, whiten_kernel_size, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, widths['block1'], depth),
            ConvGroup(widths['block1'], widths['block2'], depth),
            ConvGroup(widths['block2'], widths['block3'], depth),
            nn.MaxPool2d(3),
            Flatten(),
        )
        self.head = nn.Linear(widths['block3'], 10, bias=False)
        self.scaling_factor = scaling_factor
        
        # 混合精度设置
        self.half()
        for mod in self.modules():
            if isinstance(mod, BatchNorm):
                mod.float()

    def reset(self):
        """重置网络参数"""
        for m in self.modules():
            if type(m) in (Conv, BatchNorm, nn.Linear):
                m.reset_parameters()

    def init_whiten(self, train_images, eps=5e-4):
        """使用训练图像初始化白化层"""
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = train_images.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()
        patches_flat = patches.view(len(patches), -1)
        est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
        eigenvectors_scaled = eigenvectors.T.reshape(-1,c,h,w).flip(0) / torch.sqrt(eigenvalues.flip(0).view(-1,1,1,1) + eps)
        self.whiten.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

    def forward(self, x, whiten_bias_grad=True):
        """前向传播"""
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        return self.head(x) * self.scaling_factor


class ProxyNet(nn.Module):
    """
    小型代理网络，用于快速筛选难样本
    """
    def __init__(self):
        super().__init__()
        widths = {'block1': 32, 'block2': 64, 'block3': 64}
        depth = 2
        scaling_factor = 1/9
        
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size**2
        
        self.whiten = Conv(3, whiten_width, whiten_kernel_size, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.whiten.bias.requires_grad = False
        
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, widths['block1'], depth),
            ConvGroup(widths['block1'], widths['block2'], depth),
            ConvGroup(widths['block2'], widths['block3'], depth),
            nn.MaxPool2d(3),
            Flatten(),
        )
        self.head = nn.Linear(widths['block3'], 10, bias=False)
        self.scaling_factor = scaling_factor
        
        self.half()
        for mod in self.modules():
            if isinstance(mod, BatchNorm):
                mod.float()

    def reset(self):
        for m in self.modules():
            if type(m) in (Conv, BatchNorm, nn.Linear):
                m.reset_parameters()

    def init_whiten(self, train_images, eps=5e-4):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = train_images.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()
        patches_flat = patches.view(len(patches), -1)
        est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
        eigenvectors_scaled = eigenvectors.T.reshape(-1,c,h,w).flip(0) / torch.sqrt(eigenvalues.flip(0).view(-1,1,1,1) + eps)
        self.whiten.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

    def forward(self, x):
        x = F.conv2d(x, self.whiten.weight, self.whiten.bias)
        x = self.layers(x)
        return self.head(x) * self.scaling_factor


# 数据增强工具函数
def batch_flip_lr(inputs):
    """随机左右翻转"""
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


def batch_crop(images, crop_size=32, pad=4):
    """随机裁剪 - 向量化版本"""
    r = pad
    padded = F.pad(images, (r, r, r, r), mode='reflect')
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    
    # 向量化实现
    for sy in range(-r, r+1):
        for sx in range(-r, r+1):
            mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
            if mask.any():
                images_out[mask] = padded[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    return images_out


def batch_cutout(inputs, size=12):
    """随机遮挡 - 向量化版本 (来自原版airbench96)"""
    n, c, h, w = inputs.shape
    corner_y = torch.randint(0, h-size+1, size=(n,), device=inputs.device)
    corner_x = torch.randint(0, w-size+1, size=(n,), device=inputs.device)
    
    # 向量化生成mask
    corner_y_dists = torch.arange(h, device=inputs.device).view(1, 1, h, 1) - corner_y.view(-1, 1, 1, 1)
    corner_x_dists = torch.arange(w, device=inputs.device).view(1, 1, 1, w) - corner_x.view(-1, 1, 1, 1)
    mask_y = (corner_y_dists >= 0) & (corner_y_dists < size)
    mask_x = (corner_x_dists >= 0) & (corner_x_dists < size)
    cutout_mask = mask_y & mask_x
    
    return inputs.masked_fill(cutout_mask, 0)


def infer_tta(model, images, tta_level=2):
    """测试时增强"""
    if tta_level == 0:
        return model(images)
    elif tta_level == 1:
        return 0.5 * model(images) + 0.5 * model(images.flip(-1))
    else:  # tta_level == 2
        logits = 0.5 * model(images) + 0.5 * model(images.flip(-1))
        padded = F.pad(images, (1, 1, 1, 1), 'reflect')
        trans1 = padded[:, :, 0:32, 0:32]
        trans2 = padded[:, :, 2:34, 2:34]
        logits_t1 = 0.5 * model(trans1) + 0.5 * model(trans1.flip(-1))
        logits_t2 = 0.5 * model(trans2) + 0.5 * model(trans2.flip(-1))
        return 0.5 * logits + 0.25 * logits_t1 + 0.25 * logits_t2
