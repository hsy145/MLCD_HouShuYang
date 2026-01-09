# CIFAR-10 图像分类系统

> 机器学习课程设计 | 天津科技大学 人工智能学院
> 
> 作者：侯舒扬 | 学号：23101204

## 项目简介

本项目基于 CIFAR-10 数据集，实现了多种机器学习与深度学习模型进行图像分类任务，并使用 Streamlit 将最优模型部署为 Web 应用。

CIFAR-10 数据集包含 60,000 张 32×32 彩色图像，分为 10 个类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车。

## 项目结构

```
MLCD_HouShuYang/
├── app.py                          # Streamlit Web应用入口
├── cifar10_23101204侯舒扬.ipynb    # 完整训练流程 Notebook
├── datasets/                        # CIFAR-10 数据集目录
│   ├── data_batch_1 ~ data_batch_5 # 训练数据
│   ├── test_batch                   # 测试数据
│   └── batches.meta                 # 元数据
├── checkpoints/                     # 模型权重保存目录
│   ├── best_xgboost_cifar10.pkl    # XGBoost 模型
│   ├── best_cnn_cifar10.pth        # CNN 模型
│   ├── best_resnet18_cifar10.pth   # ResNet-18 模型
│   ├── best_airbench_cifar10.pth   # CifarNet94 模型
│   ├── best_airbench96_cifar10.pth # CifarNet96 模型 (最优)
│   └── best_eva_cifar10.pth        # EVA-02 模型
├── models/                          # 模型定义
│   ├── Airbench.py                 # CifarNet (94%精度)
│   ├── Airbench96.py               # CifarNet96 (96%精度)
│   ├── airbench96_faster.py        # 原版训练脚本
│   ├── CNN.py                      # 基础CNN网络
│   ├── ResNet.py                   # ResNet-18网络
│   └── Muon.py                     # Muon优化器
└── EVA/                             # EVA-02 Vision Transformer
    ├── __init__.py
    ├── modeling_finetune.py        # EVA模型定义
    └── rope.py                      # 旋转位置编码
```

## 模型介绍

| 模型 | 类型 | 测试集精度 | 说明 |
|------|------|-----------|------|
| XGBoost | 传统ML (sklearn) | ~54% | 梯度提升树 |
| CNN | 深度学习 (PyTorch) | ~76% | 3层卷积网络 |
| ResNet-18 | 深度学习 (PyTorch) | ~84% | 残差网络 |
| CifarNet94 | 深度学习 (PyTorch) | ~93% | Airbench优化网络 |
| CifarNet96 | 深度学习 (PyTorch) | ~96% | Airbench优化网络 |
| **EVA-02** | Vision Transformer | **~99%** | 大型预训练模型 |

### 核心技术亮点

- **白化层 (Whitening)**: 使用 PCA 白化作为首层，加速收敛
- **Muon 优化器**: 基于 Newton-Schulz 迭代的参数白化更新
- **Hard Sample Mining**: 只训练 loss 最高的样本
- **Lookahead EMA**: 指数移动平均提升泛化
- **测试时增强 (TTA)**: 翻转+平移多视角融合

## 环境配置

### 依赖安装

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install streamlit numpy pillow matplotlib scikit-learn xgboost timm einops
```

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.7 (GPU 加速)
- 显存 >= 4GB

## 快速开始

### 1. 准备数据集

将 CIFAR-10 数据集文件放入 `datasets/` 目录：
- `data_batch_1` ~ `data_batch_5`
- `test_batch`
- `batches.meta`

数据集下载: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

### 2. 模型训练

完整训练流程请参考 Jupyter Notebook:

```bash
jupyter notebook cifar10_23101204侯舒扬.ipynb
```

或者直接使用预训练权重。

### 3. 启动 Web 应用

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501` 即可使用图像分类功能。

## Web 应用功能

- **图像上传**: 支持 JPG/PNG/BMP 格式
- **实时预测**: 基于 CifarNet96 模型进行分类
- **概率分布**: 显示 Top-K 类别置信度
- **参数设置**: 可调整显示数量和置信度阈值

## 训练细节

### 数据预处理

```python
# 归一化
x_train = x_train.astype('float32') / 255.0

# CIFAR-10 标准化参数
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
```

### 数据增强

- 随机水平翻转 (Random Flip)
- 随机裁剪 (Random Crop, pad=4)
- 随机遮挡 (Cutout, size=12)

### CifarNet96 超参数

```python
epochs = 45
batch_size = 1024
learning_rate = 9.0
momentum = 0.85
weight_decay = 0.012
label_smoothing = 0.2
```

## 文件说明

### models/Airbench96.py

高精度 CIFAR-10 分类网络，包含:
- `CifarNet96`: 主网络 (128-384-512 通道)
- `ProxyNet`: 代理网络用于难样本筛选
- `batch_flip_lr`, `batch_crop`, `batch_cutout`: 数据增强
- `infer_tta`: 测试时增强

### models/Muon.py

基于 Newton-Schulz 迭代的优化器，对卷积层参数进行白化更新。

### EVA/modeling_finetune.py

EVA-02 Vision Transformer 实现，支持:
- RoPE 旋转位置编码
- SwiGLU 激活函数
- xFormers 高效注意力 (可选)

## 参考资料

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Airbench](https://github.com/KellerJordan/cifar10-airbench)
- [EVA-02](https://github.com/baaivision/EVA)
- [Streamlit Documentation](https://docs.streamlit.io/)
