import streamlit as st
import numpy as np
from PIL import Image
import pickle
import os

# CIFAR-10 类别名称
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

# 中文类别名称
CIFAR10_CLASSES_CN = ['飞机', '汽车', '鸟', '猫', '鹿', 
                      '狗', '青蛙', '马', '船', '卡车']

# 页面配置
st.set_page_config(
    page_title="CIFAR-10图像分类器",
    page_icon="🖼️",
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 10px;
    }
    .student-info {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 30px;
    }
    .section-title {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 15px;
    }
    .result-text {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
    }
    .stImage > img {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<h1 class="main-title">CIFAR-10数据训练10分类图像分类器demo</h1>', unsafe_allow_html=True)
st.markdown('<p class="student-info">学号：23101204，姓名：侯舒扬</p>', unsafe_allow_html=True)

# 分割线
st.markdown("---")

def preprocess_image(image, target_size=(32, 32)):
    """预处理图像为模型输入格式"""
    # 调整大小为32x32
    img = image.resize(target_size)
    # 转换为RGB（如果是RGBA则去除alpha通道）
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # 转换为numpy数组并归一化
    img_array = np.array(img).astype('float32') / 255.0
    return img, img_array

def load_model():
    """加载保存的模型"""
    model_path = 'best_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model, 'sklearn'
    
    # 尝试加载ResNet18 PyTorch模型
    pytorch_model_path = 'best_resnet18_cifar10.pth'
    if os.path.exists(pytorch_model_path):
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            
            # 自定义ResNet18 for CIFAR-10
            class BasicBlock(nn.Module):
                expansion = 1
                def __init__(self, in_planes, planes, stride=1):
                    super(BasicBlock, self).__init__()
                    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
                    self.bn1 = nn.BatchNorm2d(planes)
                    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
                    self.bn2 = nn.BatchNorm2d(planes)
                    self.shortcut = nn.Sequential()
                    if stride != 1 or in_planes != self.expansion * planes:
                        self.shortcut = nn.Sequential(
                            nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(self.expansion * planes)
                        )
                def forward(self, x):
                    out = F.relu(self.bn1(self.conv1(x)))
                    out = self.bn2(self.conv2(out))
                    out += self.shortcut(x)
                    out = F.relu(out)
                    return out
            
            class ResNet(nn.Module):
                def __init__(self, block, num_blocks, num_classes=10):
                    super(ResNet, self).__init__()
                    self.in_planes = 64
                    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                    self.bn1 = nn.BatchNorm2d(64)
                    self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
                    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
                    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
                    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
                    self.linear = nn.Linear(512 * block.expansion, num_classes)
                def _make_layer(self, block, planes, num_blocks, stride):
                    strides = [stride] + [1] * (num_blocks - 1)
                    layers = []
                    for stride in strides:
                        layers.append(block(self.in_planes, planes, stride))
                        self.in_planes = planes * block.expansion
                    return nn.Sequential(*layers)
                def forward(self, x):
                    out = F.relu(self.bn1(self.conv1(x)))
                    out = self.layer1(out)
                    out = self.layer2(out)
                    out = self.layer3(out)
                    out = self.layer4(out)
                    out = F.avg_pool2d(out, 4)
                    out = out.view(out.size(0), -1)
                    out = self.linear(out)
                    return out
            
            model = ResNet(BasicBlock, [2, 2, 2, 2])
            model.load_state_dict(torch.load(pytorch_model_path, map_location='cpu'))
            model.eval()
            return model, 'pytorch'
        except Exception as e:
            st.warning(f"加载PyTorch模型失败: {e}")
    
    return None, None

def predict_sklearn(model, img_array):
    """使用sklearn模型进行预测"""
    # 展平图像数据
    img_flat = img_array.reshape(1, -1)
    # 预测
    prediction = model.predict(img_flat)[0]
    # 获取预测概率（如果模型支持）
    try:
        proba = model.predict_proba(img_flat)[0]
    except:
        proba = None
    return prediction, proba

def predict_pytorch(model, img_array):
    """使用PyTorch模型进行预测"""
    import torch
    import torch.nn.functional as F
    
    # 转换为PyTorch张量 (N, C, H, W)
    img_tensor = torch.from_numpy(img_array).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 32, 32)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        proba = F.softmax(outputs, dim=1).numpy()[0]
        prediction = np.argmax(proba)
    
    return prediction, proba

# 创建两列布局
col1, col2 = st.columns(2)

with col1:
    st.markdown('<h2 class="section-title">上传图像</h2>', unsafe_allow_html=True)
    st.write("Upload an image")
    
    # 文件上传组件
    uploaded_file = st.file_uploader(
        "拖拽文件到此处或点击浏览",
        type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
        help="支持 PNG, JPG, JPEG, BMP, GIF 格式，建议上传32x32的图像以获得最佳效果"
    )
    
    if uploaded_file is not None:
        # 显示上传的图像
        image = Image.open(uploaded_file)
        st.image(image, caption=f'{uploaded_file.name}', use_column_width=True)
        
        # 显示图像信息
        st.info(f"图像尺寸: {image.size[0]} x {image.size[1]} 像素")

with col2:
    st.markdown('<h2 class="section-title">分类结果</h2>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # 预处理图像
        processed_img, img_array = preprocess_image(image)
        
        # 显示预处理后的图像
        st.image(processed_img, caption="预处理后的图像 (32x32)", width=200)
        
        # 加载模型并预测
        model, model_type = load_model()
        
        if model is not None:
            if model_type == 'sklearn':
                prediction, proba = predict_sklearn(model, img_array)
            else:
                prediction, proba = predict_pytorch(model, img_array)
            
            # 显示预测结果
            st.markdown(f'<p class="result-text">预测类别: {CIFAR10_CLASSES[prediction]} ({CIFAR10_CLASSES_CN[prediction]})</p>', unsafe_allow_html=True)
            
            # 如果有概率，显示置信度
            if proba is not None:
                st.write(f"置信度: {proba[prediction]*100:.2f}%")
                
                # 显示前5个预测结果
                st.subheader("Top-5 预测结果")
                top5_idx = np.argsort(proba)[::-1][:5]
                for idx in top5_idx:
                    st.progress(float(proba[idx]))
                    st.write(f"{CIFAR10_CLASSES[idx]} ({CIFAR10_CLASSES_CN[idx]}): {proba[idx]*100:.2f}%")
        else:
            st.warning("⚠️ 未找到训练好的模型文件！")
            st.info("""
            请先训练模型并保存：
            - sklearn模型保存为 `best_model.pkl`
            - PyTorch模型保存为 `best_model.pth`
            
            保存模型示例代码：
            ```python
            # sklearn模型
            import pickle
            with open('best_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            
            # PyTorch模型
            torch.save(model.state_dict(), 'best_model.pth')
            ```
            """)
            
            # 显示演示结果（随机预测）
            st.subheader("演示模式 (随机预测)")
            random_pred = np.random.randint(0, 10)
            random_proba = np.random.dirichlet(np.ones(10))
            
            st.markdown(f'<p class="result-text">预测类别: {CIFAR10_CLASSES[random_pred]} ({CIFAR10_CLASSES_CN[random_pred]})</p>', unsafe_allow_html=True)
            st.write(f"置信度: {random_proba[random_pred]*100:.2f}%")
    else:
        st.info("请在左侧上传一张图像进行分类")

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 20px;">
    <p>CIFAR-10 图像分类器 | 机器学习课程设计</p>
    <p>支持的类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车</p>
</div>
""", unsafe_allow_html=True)
