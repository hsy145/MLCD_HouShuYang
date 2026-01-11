import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
from PIL import Image
import os
import torch
import torch.nn.functional as F
import sys
import random

# 设置全局随机种子
def set_seed(seed=23101204):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(23101204)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from EVA import modeling_finetune
from timm.models import create_model

# -----------------------------------------------------------------------------
# 1. 页面配置
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CIFAR-10 智能分类系统",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. 核心 CSS 样式 (隐藏原生顶栏 + 自定义新顶栏)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* 1. 隐藏 Streamlit 原生顶栏 */
    header[data-testid="stHeader"] {
        display: none;
    }
    
    /* 2. 调整主区域上边距 */
    .block-container {
        padding-top: 80px !important;
    }

    /* 3. 自定义顶部导航栏样式 */
    .custom-navbar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 60px;
        background-color: #ffffff;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        z-index: 999999;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 40px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* 左侧：项目名与个人信息 */
    .navbar-left {
        display: flex;
        align-items: center;
        gap: 30px;
    }
    .project-logo {
        font-size: 18px;
        font-weight: 700;
        color: #0052D9; /* 腾讯蓝 */
        letter-spacing: 0.5px;
    }
    
    /* 学生信息胶囊样式 */
    .student-info {
        display: flex;
        gap: 15px;
        font-size: 13px;
        color: #555;
        background-color: #F5F7FA;
        padding: 6px 12px;
        border-radius: 4px;
        border: 1px solid #E1E4E8;
    }
    .info-label {
        color: #888;
        margin-right: 4px;
    }
    .info-value {
        font-weight: 600;
        color: #333;
    }

    /* 右侧：功能区 */
    .navbar-right {
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .nav-link {
        color: #666;
        text-decoration: none !important;
        font-size: 14px;
        cursor: pointer;
        transition: color 0.2s;
    }
    .nav-link:hover {
        color: #0052D9;
        text-decoration: none !important;
    }
    .nav-link:visited, .nav-link:active, .nav-link:focus {
        text-decoration: none !important;
    }
    /* 注册按钮样式 */
    .nav-btn-register {
        background-color: #0052D9;
        color: white !important;
        padding: 6px 18px;
        border-radius: 2px;
        text-decoration: none !important;
        font-size: 13px;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    .nav-btn-register:hover {
        background-color: #003C9D;
        text-decoration: none !important;
    }

    /* 全局背景与卡片 */
    .stApp {
        background-color: #F7F9FC;
    }
    .main-card {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 40px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
        border: 1px solid #EBEEF5;
    }
    
    /* 预测结果框优化 - 纯净风格 */
    .prediction-box {
        background-color: #F2F5FF;
        border-left: 4px solid #0052D9;
        padding: 20px;
        margin-top: 20px;
        border-radius: 0 4px 4px 0;
    }
    .pred-title {
        color: #333;
        font-size: 16px;
        margin-bottom: 8px;
        font-weight: 600;
    }
    .pred-value {
        color: #0052D9;
        font-size: 24px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. 注入 HTML 自定义导航栏
# -----------------------------------------------------------------------------
st.markdown("""
<nav class="custom-navbar">
    <div class="navbar-left">
        <div class="project-logo">
            CIFAR-10 图像分类系统
        </div>
        <div class="student-info">
            <span><span class="info-label">姓名</span><span class="info-value">侯舒扬</span></span>
            <span style="color:#DDD">|</span>
            <span><span class="info-label">学号</span><span class="info-value">23101204</span></span>
        </div>
    </div>
    <div class="navbar-right">
        <a class="nav-link" href="https://github.com/hsy-tust/MLCD_HouShuYang/blob/main/README.md" target="_blank">帮助文档</a>
        <a class="nav-link" target="_self">登录</a>
        <a class="nav-btn-register" target="_self">注册账号</a>
    </div>
</nav>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. 逻辑代码
# -----------------------------------------------------------------------------

# CIFAR-10 类别
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CIFAR10_CLASSES_CN = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

# 加载模型
# EVA配置
EVA_MEAN = (0.48145466, 0.4578275, 0.40821073)
EVA_STD = (0.26862954, 0.26130258, 0.27577711)

@st.cache_resource
def load_model():
    """加载 EVA-02 模型"""
    model_path = 'checkpoints/best_eva_cifar10.pth'
    if not os.path.exists(model_path):
        return None
    
    model = create_model(
        'eva02_base_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE',
        pretrained=False, num_classes=10,
        drop_rate=0.0, drop_path_rate=0.0, attn_drop_rate=0.0, use_mean_pooling=True
    )
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model

def preprocess_image(image, target_size=(224, 224)):
    """预处理图像 (EVA需要224x224)"""
    if image.mode != 'RGB': 
        image = image.convert('RGB')
    display_img = image.copy()
    img_resized = image.resize(target_size, Image.BILINEAR)
    img_array = np.array(img_resized).astype('float32') / 255.0
    return display_img, img_array

def predict(model, img_array):
    """使用EVA模型进行预测"""
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    # EVA归一化
    mean = torch.tensor(EVA_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(EVA_STD).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    with torch.no_grad():
        outputs = model(img_tensor)
        proba = F.softmax(outputs.float(), dim=1).numpy()[0]
        prediction = np.argmax(proba)
    
    return prediction, proba

# 侧边栏
with st.sidebar:
    st.markdown("### 参数设置")
    top_k = st.slider("显示前 K 个结果", 1, 10, 5)
    conf_threshold = st.slider("置信度阈值", 0.0, 1.0, 0.01)
    
    st.markdown("---")
    st.markdown("**系统说明**")
    st.caption("本系统基于 EVA-02 模型构建，用于 CIFAR-10 数据集的图像分类任务。\n\n天津科技大学 人工智能学院")

# 主界面内容
st.markdown('<h2 style="color:#333; font-weight:600; margin-bottom:10px;">图像分类任务演示</h2>', unsafe_allow_html=True)
st.markdown('<p style="color:#666; font-size:14px; margin-bottom: 30px;">请上传本地图像文件，系统将自动进行预处理与模型推断。</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("上传图片文件 (JPG/PNG)", type=['png', 'jpg', 'jpeg', 'bmp'])

# 加载模型
model = load_model()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    display_img, img_array = preprocess_image(image)
    
    if model is not None:
        with st.spinner('正在识别...'):
            prediction, proba = predict(model, img_array)
    else:
        st.error("未找到模型文件 checkpoints/best_eva_cifar10.pth")
        prediction = 0
        proba = np.zeros(10)
        proba[0] = 1.0

    st.markdown("---")
    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        st.markdown("**原始图像**")
        st.image(display_img, use_container_width=True)
    
    with col2:
        st.markdown("**识别结果**")
        st.markdown(f"""
        <div class="prediction-box">
            <div class="pred-title">预测类别</div>
            <div class="pred-value">{CIFAR10_CLASSES_CN[prediction]} <span style="font-size:16px;color:#666;font-weight:400">({CIFAR10_CLASSES[prediction]})</span></div>
            <div style="margin-top:8px; font-size:13px; color:#666;">置信度：{(proba[prediction]*100):.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>**概率分布**", unsafe_allow_html=True)
        # 进度条
        for i in np.argsort(proba)[::-1][:top_k]:
             val = float(proba[i])
             if val > conf_threshold:
                 st.progress(val, text=f"{CIFAR10_CLASSES_CN[i]} ({val*100:.1f}%)")
